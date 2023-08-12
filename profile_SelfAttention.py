import torch
import torch.nn.functional as F
from model import Block, GPTConfig
from model import CausalSelfAttention, MLP

config = GPTConfig()
config.block_size = 1024
config.vocab_size = 2048
config.n_head = 12
config.n_embd = 768
batch_size = 32
warmup_steps = 10  # warmup_steps for profiler
active_steps = 4  # active_steps for profiler
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    # raise "CUDA not available"
    device = torch.device('cpu')

def set_profiler(module, profile=True):
    for name, m in module.named_modules():
        if isinstance(m, CausalSelfAttention) or isinstance(m, MLP):
            m.profile = profile


decoder = Block(config).to(device)
lm_head = torch.nn.Linear(config.n_embd, config.vocab_size, bias=False).to(device)
decoder.train()
lm_head.train()

hidden_states = torch.randn(batch_size, config.block_size, config.n_embd).to(device)
targets = torch.randint(0, config.vocab_size, size=(batch_size, config.block_size)).to(device)

if __name__ == '__main__':
    for step in range(1000):
        if step >= warmup_steps + active_steps:
            torch.cuda.cudart().cudaProfilerStop()
            quit()
        elif step == warmup_steps:
            set_profiler(decoder, profile=True)
            torch.cuda.cudart().cudaProfilerStart()
        elif step >= warmup_steps:
            out = decoder(hidden_states)
            torch.cuda.nvtx.range_push("lm_head")
            logits = lm_head(out)
            torch.cuda.nvtx.range_pop()
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            torch.cuda.nvtx.range_push("backward")
            loss.backward()
            torch.cuda.nvtx.range_pop()
        else:
            out = decoder(hidden_states)
            logits = lm_head(out)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            loss.backward()
