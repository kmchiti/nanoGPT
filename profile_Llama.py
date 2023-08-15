import os
from pathlib import Path

from datasets import load_dataset, DatasetDict
from transformers import DataCollatorForLanguageModeling
from torch.utils.data import DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import set_seed

import torch
import argparse
import numpy as np
import time

from transformers import AdamW
from transformers import get_scheduler

hf_home = Path.home() / "scratch" / "hf_home"  # change this to your own path
hf_home.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(hf_home)

checkpoint = "meta-llama/Llama-2-7b-chat-hf"

parser = argparse.ArgumentParser(
    description='Profile different multi-head attention configurations')
parser.add_argument(
    '--batch_size', default=1, type=int,
    help='Mini-batch sizes, default: 1')
parser.add_argument(
    '--num_epochs', default=1, type=int,
    help='number of epochs, default: 1')
parser.add_argument(
    '--context_length', default=2048, type=int,
    help='context length, default: 2048')
parser.add_argument(
    '--seed', default=1347, type=int,
    help='context length, default: 1347')

args = parser.parse_args()
ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
ds_valid = load_dataset("huggingface-course/codeparrot-ds-valid", split="validation")
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=True)

raw_datasets = DatasetDict(
    {
        "train": ds_train.shuffle(seed=42).select(range(50000)),
        "valid": ds_valid.shuffle(seed=42).select(range(500))
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

set_seed(args.seed)
model = AutoModelForCausalLM.from_pretrained(checkpoint)
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    raise "CUDA not available"

optimizer = AdamW(model.parameters(), lr=5e-5)
num_training_steps = args.num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=num_training_steps,
)


def train(batch):
    batch = {k: v.to(model.device) for k, v in batch.items()}
    outputs = model(**batch)
    loss = outputs.loss
    loss.backward()

    optimizer.step()
    lr_scheduler.step()
    optimizer.zero_grad()


print(num_training_steps)

model.train()
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=5, active=1, repeat=3),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./logs_Llama_7b'),
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_modules=True,
) as prof:
    for step, batch_data in enumerate(train_dataloader):
        if step >= (1 + 5 + 1) * 3:
            break
        train(batch_data)
        prof.step()  # Need to call this at the end of each step

# prof.export_chrome_trace("trace_Llama_7b.json")
