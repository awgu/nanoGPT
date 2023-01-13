"""
Unit tests for sp_model.py.

This test script expects to be run using torchrun. For example:
torchrun --standalone --nproc_per_node=2 test_sp_model.py
"""
import contextlib
import functools
from collections import Counter
from typing import Callable, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from dist_utils import dist_setup
from model import Block, CausalSelfAttention, GPT, GPTConfig, MLP
from sp_model import SPCausalSelfAttention

from torch.distributed._composable import replicate
from torch.nn.parallel import DistributedDataParallel as DDP


PROFILE_SAVE_DIR = "./bench_log"

dist_setup()
world_size = dist.get_world_size()
rank = dist.get_rank()

seed = 1337
batch_size = 1
block_size = 10
n_layer = 2
n_head = 3
n_embd = 6
config = GPTConfig(
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=0.0,  # disable for determinism and due to TP dropout issues
)


def collective_with_count(
    orig_collective: Callable,
    counter: Counter,
    *args,
    **kwargs,
):
    counter[orig_collective] += 1
    return orig_collective(*args, **kwargs)


@contextlib.contextmanager
def patch_device_mesh_all_reduce(new_all_reduce: Callable):
    _orig_all_reduce = torch.distributed._tensor.device_mesh.all_reduce
    torch.distributed._tensor.device_mesh.all_reduce = new_all_reduce
    try:
        yield
    finally:
        torch.distributed._tensor.device_mesh.all_reduce = _orig_all_reduce


@contextlib.contextmanager
def patch_device_mesh_all_gather(new_all_gather: Callable):
    _orig_all_gather = torch.distributed._tensor.device_mesh.all_gather
    torch.distributed._tensor.device_mesh.all_gather = new_all_gather
    try:
        yield
    finally:
        torch.distributed._tensor.device_mesh.all_gather = _orig_all_gather


def _check_cuda() -> None:
    assert torch.cuda.is_available(), f"Requires CUDA"
    assert torch.cuda.device_count() > 1, f"Requires at least 2 GPUs"


@torch.no_grad()
def _init_values(module: nn.Module, sp_module: nn.Module) -> None:
    for (n1, p1), (n2, p2) in zip(
        module.named_parameters(), sp_module.named_parameters()
    ):
        assert n1 == n2, f"{n1} {n2}"
        p2.copy_(p1)


def _get_block_inp() -> Tuple[torch.Tensor]:
    return (torch.randn((batch_size, block_size, n_embd), device="cuda"),)


def _get_gpt_inp() -> Tuple[torch.Tensor]:
    x = torch.randint(50257, (batch_size, block_size), device="cuda")
    y = torch.randint(50257, (batch_size, block_size), device="cuda")
    return (x, y)


def _check_sp_forward_backward(
    module: nn.Module,
    sp_module: nn.Module,
    get_input_fn: Callable,
):
    """
    Checks the sequence parallel ``sp_module`` 's forward against that of
    ``module`` for parity.
    """
    # Use the same seed on all ranks to replicate the input tensor
    torch.manual_seed(seed)
    inp = get_input_fn()
    targets = get_input_fn()[0]
    out = module(*inp)
    # Manually shard the input on the sequence dimension
    world_size = dist.get_world_size()
    sp_inp = (
        torch.chunk(inp[0], world_size, dim=1)[dist.get_rank()],
        *inp[1:],
    )
    sp_out = sp_module(*sp_inp)
    # Manually unshard the input on the sequence dimension
    sp_out_unsharded = torch.empty_like(out)
    dist.all_gather(
        list(torch.chunk(sp_out_unsharded, world_size, dim=1)),
        sp_out,
    )
    torch.testing.assert_close(sp_out_unsharded, out)

    # HACK: Use per-sample loss function. Averaging over the sample would
    # divide by a different value for the non-parallel and parallel versions.
    loss_fn = nn.MSELoss(reduction="none")
    loss = loss_fn(out, targets)
    sp_targets = torch.chunk(targets, chunks=dist.get_world_size(), dim=1)[
        dist.get_rank()
    ]
    sp_loss = loss_fn(sp_out, sp_targets)
    loss.backward(gradient=torch.ones_like(loss))
    sp_loss.backward(gradient=torch.ones_like(sp_loss))

    for (n1, p1), (n2, p2) in zip(
        reversed(list(module.named_parameters())),
        reversed(list(sp_module.named_parameters())),
    ):
        assert n1 == n2, f"{n1} {n2}"
        # NOTE: Parameters are logically replicated across ranks. Either we
        # must all-reduce here to sum contributions from all subsequences, or
        # we need to apply `replicate()` and multiply by the world size to undo
        # the division.
        # dist.all_reduce(p2.grad)
        p2.grad *= dist.get_world_size()
        torch.testing.assert_close(p1.grad, p2.grad)


def test_sp_self_attn_module(config: GPTConfig):
    """Tests ``SPCausalSelfAttention`` against ``CausalSelfAttention``."""
    _check_cuda()
    torch.set_printoptions(linewidth=200)
    torch.manual_seed(seed)
    sp_self_attn = SPCausalSelfAttention(config).cuda()
    self_attn = CausalSelfAttention(config).cuda()
    _init_values(self_attn, sp_self_attn)
    replicate(sp_self_attn)
    _check_sp_forward_backward(self_attn, sp_self_attn, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_sp_self_attn_module!")


def profile_sp_self_attn_module():
    block_size = 1024
    n_head = 25
    n_embd = 1600
    batch_size = 8
    config = GPTConfig(
        block_size=block_size,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=0.1,
    )
    sp_self_attn = SPCausalSelfAttention(config).cuda()
    sp_self_attn = torch.compile(sp_self_attn)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    loss_fn = nn.MSELoss(reduction="none")
    optim = torch.optim.Adam(sp_self_attn.parameters(), lr=1e-2)

    def run_sp_step():
        optim.zero_grad()
        inp = (torch.randn((batch_size, block_size, n_embd), device="cuda"),)
        targets = torch.randn((batch_size, block_size, n_embd), device="cuda")
        sp_inp = (
            torch.chunk(inp[0], world_size, dim=1)[rank],
            *inp[1:],
        )
        sp_targets = torch.chunk(targets, chunks=dist.get_world_size(), dim=1)[rank]
        sp_out = sp_self_attn(*sp_inp)
        sp_loss = loss_fn(sp_out, sp_targets)
        sp_loss.backward(gradient=torch.ones_like(sp_loss))
        optim.step()

    wait, warmup, active = 5, 5, 2
    num_steps = wait + warmup + active
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=wait, warmup=warmup, active=active, repeat=1
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(PROFILE_SAVE_DIR)
        if not rank  # only save on rank 0
        else None,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,  # incurs an additional overhead, disable if not needed
        with_flops=True,
        with_modules=False,  # only for torchscript models atm
    ) as prof:
        for step_idx in range(1, num_steps + 1):
            run_sp_step()
            if rank == 0:
                prof.step()


test_sp_self_attn_module(config)
# profile_sp_self_attn_module()
