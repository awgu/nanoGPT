"""
This is a distributed version of bench.py.

To run on 2 GPUs:
torchrun --standalone --nproc_per_node=2 dist_bench.py
"""
from enum import auto, Enum

import torch

from common_train import benchmark, benchmark_with_profiler, init_benchmark_get_batch_fn
from dist_utils import apply_ac, apply_ddp, apply_fsdp, apply_tp, dist_setup
from model import GPT, GPTConfig


class DistAPI(Enum):
    DDP = auto()
    FSDP = auto()
    TP = auto()


device = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.manual_seed(1337)

batch_size = 1
block_size = 1024
autocast_dtype = torch.bfloat16
compile = True
dist_api = DistAPI.TP
use_ac = False
if use_ac:
    assert not compile, f"`torch.compile()` does not work with eager AC"
    assert (
        dist_api != DistAPI.TP
    ), f"TP should use a different distributed AC implementation, which is not added yet"

profile = False
track_memory = False
use_oom_observer = False
oom_observer_kwargs = {
    "trace_alloc_record_context": False,
}

dist_setup()
if torch.distributed.get_rank() == 0:
    print(f"{dist_api} with compile={compile} and use_ac={use_ac}")
get_batch = init_benchmark_get_batch_fn(
    batch_size, block_size, device, True, use_same_seed=(dist_api == DistAPI.TP)
)

base_kwargs = {"n_layer": 12, "n_head": 12, "n_embd": 768}
xl_kwargs = {"n_layer": 48, "n_head": 25, "n_embd": 1600}
xl_kwargs_tp = {"n_layer": 48, "n_head": 26, "n_embd": 1664}
# TODO (awgu): Add more config settings if needed.
config = GPTConfig(
    block_size=block_size,  # how far back does the model look? i.e. context size
    dropout=0,  # for determinism
    # **base_kwargs,
    # **xl_kwargs,
    **xl_kwargs_tp,
)
model = GPT(config)
model.to(device)
optimizer = model.configure_optimizers(
    weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95)
)

if dist_api == DistAPI.DDP:
    model = apply_ddp(model)
elif dist_api == DistAPI.FSDP:
    # TODO (awgu): If we change `use_orig_params=True` to not clean FQNs in
    # `named_parameters()`, then we should be able to initialize the optimizer
    # after applying FSDP.
    model = apply_fsdp(model)
    if hasattr(model, "sharding_strategy") and torch.distributed.get_rank() == 0:
        print(model.sharding_strategy)
elif dist_api == DistAPI.TP:
    world_size = torch.distributed.get_world_size()
    assert config.n_head % world_size == 0, (
        f"TP requires the number of heads be divisible by the world size but "
        f"got {config.n_head} heads for world size {world_size}"
    )
    assert batch_size == 1, (
        f"Batch size greater than 1 raises error AssertionError: output spec "
        "does not match with output! Expected DTensorSpec, got None."
    )
    model = apply_tp(model)
else:
    raise ValueError(f"Unknown DistAPI: {dist_api}")
if use_ac:
    model = apply_ac(model)

if compile:
    print("Compiling model...")
    model = torch.compile(model)

if profile:
    benchmark_with_profiler(model, optimizer, get_batch, autocast_dtype)
else:
    benchmark(
        model,
        optimizer,
        get_batch,
        autocast_dtype,
        track_memory,
        use_oom_observer,
        oom_observer_kwargs,
    )
