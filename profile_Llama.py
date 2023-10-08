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
    '--seed', default=42, type=int,
    help='context length, default: 1347')
parser.add_argument(
    '--warmup_steps', default=5, type=int,
    help='warmup steps for profiler, default: 5')
parser.add_argument(
    '--active_steps', default=1, type=int,
    help='cactive steps for profiler, default: 1')
parser.add_argument(
    '--repeat', default=3, type=int,
    help='repeat for profiler, default: 3')


def train(batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


if __name__ == '__main__':
    hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
    checkpoint = "decapoda-research/llama-7b-hf"
    args = parser.parse_args()
    set_seed(args.seed)

    print("loading model:", checkpoint)
    config = AutoConfig.from_pretrained(checkpoint, dtype=torch.float16)
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
        offload_folder=os.path.join(os.environ["SLURM_TMPDIR"], "offload"),
    )
    model.tie_weights()
    model.to('cuda')

    print("random dataset")
    tokenized_datasets = torch.randint(0, 32000, (1000, args.context_length))
    tokenizer = LlamaTokenizer.from_pretrained(checkpoint)
    if tokenizer.pad_token is None:
        print("Add padding token")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        tokenizer.padding_side = "right"
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    train_dataloader = DataLoader(tokenized_datasets, shuffle=True, batch_size=args.batch_size, collate_fn=data_collator)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = args.num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    model.train()
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
    print("finish PyTorch profiler")

