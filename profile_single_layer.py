import torch
import torch.nn.functional as F
from transformers.models.opt.modeling_opt import OPTDecoderLayer
from transformers.models.opt.configuration_opt import OPTConfig

hidden_size = 768
num_hidden_layers = 12
vocab_size = 2048
batch_size = 256
device = torch.device('cuda')

cfg = OPTConfig(hidden_size=hidden_size, num_hidden_layers=num_hidden_layers, vocab_size=vocab_size)
decoder = OPTDecoderLayer(cfg).to(device)
lm_head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False).to(device)
decoder.train()
lm_head.train()

hidden_states = torch.randn(batch_size, 1, hidden_size).to(device)
targets = torch.randint(0, vocab_size, size=(batch_size, 1)).to(device)


if __name__ == '__main__':
    # training loop wrapped with profiler object
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=5, active=1, repeat=3),
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            with_modules=True,
    ) as prof:
        for step in range(1000):
            if step >= (1 + 5 + 1) * 3:
                break
            out = decoder(hidden_states)[0]
            logits = lm_head(out)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
            loss.backward()
            prof.step()

    prof.export_chrome_trace("single_layer_trace.json")
