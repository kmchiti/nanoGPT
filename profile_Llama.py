import os
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, LlamaTokenizer
from transformers import set_seed

import torch
import argparse
import numpy as np
import time

from transformers import AdamW
from transformers import get_scheduler
from accelerate import init_empty_weights, dispatch_model, infer_auto_device_map, load_checkpoint_and_dispatch

parser = argparse.ArgumentParser(
    description='Profile different multi-head attention configurations')
parser.add_argument(
    '--batch_size', default=1, type=int,
    help='Mini-batch sizes, default: 1')
parser.add_argument(
    '--num_epochs', default=1, type=int,
    help='number of epochs, default: 1')
parser.add_argument(
    '--context_length', default=1024, type=int,
    help='context length, default: 1024')
parser.add_argument(
    '--seed', default=1347, type=int,
    help='context length, default: 1347')
parser.add_argument(
    '--warmup_steps', default=5, type=int,
    help='warmup steps for profiler, default: 5')
parser.add_argument(
    '--active_steps', default=1, type=int,
    help='cactive steps for profiler, default: 1')
parser.add_argument(
    '--repeat', default=1, type=int,
    help='repeat for profiler, default: 1')
parser.add_argument('--torchProf',
                    dest='torchProf',
                    action='store_true',
                    help='run PyTorch profiler.')
parser.add_argument('--cudaProf',
                    dest='cudaProf',
                    action='store_true',
                    help='run PyTorch profiler.')


# def set_profiler(module, profile=True):
#     for name, m in module.named_modules():
#         if isinstance(m, CausalSelfAttention) or isinstance(m, MLP):
#             m.profile = profile

def train(batch, CUDA_profile=False):
    if CUDA_profile: torch.cuda.nvtx.range_push("fetch_data")
    batch = {k: v.to(model.device) for k, v in batch.items()}
    if CUDA_profile: torch.cuda.nvtx.range_pop()
    if CUDA_profile: torch.cuda.nvtx.range_push("forward")
    outputs = model(**batch)
    if CUDA_profile: torch.cuda.nvtx.range_pop()
    loss = outputs.loss
    if CUDA_profile: torch.cuda.nvtx.range_push("backward")
    loss.backward()
    if CUDA_profile: torch.cuda.nvtx.range_pop()

    if CUDA_profile: torch.cuda.nvtx.range_push("optimizer.step")
    optimizer.step()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()
    if CUDA_profile: torch.cuda.nvtx.range_pop()


if __name__ == '__main__':
    hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
    checkpoint = "decapoda-research/llama-7b-hf"
    args = parser.parse_args()
    set_seed(args.seed)

    print("loading model:", checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, dtype=torch.bfloat16)
    # Initializes an empty shell with the model. This is instant and does not use any memory.
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)

    # Initialize the model under the previous context manager breaks the tied weights. So, we need to retie them.
    model.tie_weights()
    device_map = infer_auto_device_map(
        model,
        max_memory={0: "80GB", "cpu": "20GB"},
        no_split_module_classes=["LlamaDecoderLayer"],
        dtype='float16',
    )
    load_checkpoint_and_dispatch(
        model,
        f"{hf_home}/hub/models--decapoda-research--llama-7b-hf/snapshots/5f98eefcc80e437ef68d457ad7bf167c2c6a1348/",
        device_map=device_map,
        dtype='float16',
        offload_folder=f"/Tmp/slurm.3266104.0/offload",
    )
    model.tie_weights()

    print("loading dataset")
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=args.seed).select(range(50000)),
            "valid": ds_valid.shuffle(seed=args.seed).select(range(500))
        }
    )


    def tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=args.context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
            if length == args.context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}


    tokenized_datasets = raw_datasets.map(
        tokenize, batched=True, remove_columns=raw_datasets["train"].column_names, num_proc=4
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    train_dataloader = DataLoader(
        tokenized_datasets["train"], shuffle=True, batch_size=args.batch_size, collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_datasets["valid"], batch_size=args.batch_size, collate_fn=data_collator
    )

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
    if args.torchProf:
        print("starting PyTorch profiler")
        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=args.warmup_steps, active=args.active_steps,
                                                 repeat=args.repeat),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs_Llama_7b'),
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True,
                with_modules=True,
        ) as prof:
            for step, batch_data in enumerate(train_dataloader):
                if step >= (1 + args.warmup_steps + args.active_steps) * args.repeat:
                    break
                train(batch_data)
                prof.step()  # Need to call this at the end of each step

    if args.cudaProf:
        print("starting CUDA profiler")
        # sould run with: nsys profile -w true -t cuda,nvtx,osrt,cudnn,cublas -s cpu  --capture-range=cudaProfilerApi --capture-range-end --cudabacktrace=true -x true -o baseline_LLama_7b python profile_Llama.py
        for step, batch_data in enumerate(train_dataloader):

            if step >= args.warmup_steps + args.active_steps:
                torch.cuda.cudart().cudaProfilerStop()
                break
            elif step == args.warmup_steps:
                torch.cuda.cudart().cudaProfilerStart()
            elif step >= args.warmup_steps:
                train(batch_data, CUDA_profile=True)
            else:
                train(batch_data)
