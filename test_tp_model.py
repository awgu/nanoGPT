"""
Unit tests for tp_model.py.

This test script expects to be run using torchrun. For example:
torchrun --standalone --nproc_per_node=2 test_tp_model.py

For each test, we place the models on CUDA because DTensor will automatically
do so for communication, as it only supports NCCL. To test tensor-parallel vs.
local parity, we must set the same seed on all ranks for initialization and
for creating the forward input to ensure that the local module and the input
are logically replicated across ranks.

TODO: Consider migrating to unittest instead of using this hacky setup.
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

from torch.distributed._tensor import DeviceMesh, DTensor, Replicate
from tp_model import (
    _init_qkv_params_from_tensors,
    parallelize_block,
    parallelize_causal_self_attention,
    parallelize_gpt,
    parallelize_mlp,
    TPBlock,
    TPCausalSelfAttention,
    TPMLP,
)


dist_setup()
world_size = dist.get_world_size()
rank = dist.get_rank()
device_mesh = DeviceMesh("cuda", torch.arange(world_size))  # 1D device mesh

seed = 1337
# TODO (awgu): Batch size greater than 1 errors on the self-attention's output
# projection:
# AssertionError: output spec does not match with output! Expected DTensorSpec, got None.
batch_size = 1
block_size = 4
n_layer = 2
n_head = 2
n_embd = 4
config = GPTConfig(
    block_size=block_size,
    n_layer=n_layer,
    n_head=n_head,
    n_embd=n_embd,
    dropout=0.0,  # disable for determinism and due to TP dropout issues
)


def collective_with_count(
    device_mesh: DeviceMesh,
    orig_collective: Callable,
    counter: Counter,
    *args,
    **kwargs,
):
    counter[orig_collective] += 1
    return orig_collective(device_mesh, *args, **kwargs)


@contextlib.contextmanager
def patch_device_mesh_all_reduce(new_all_reduce: Callable):
    _orig_all_reduce = torch.distributed._tensor.device_mesh.DeviceMesh.all_reduce
    torch.distributed._tensor.device_mesh.DeviceMesh.all_reduce = new_all_reduce
    try:
        yield
    finally:
        torch.distributed._tensor.device_mesh.DeviceMesh.all_reduce = _orig_all_reduce


@contextlib.contextmanager
def patch_device_mesh_all_gather(new_all_gather: Callable):
    _orig_all_gather = torch.distributed._tensor.device_mesh.DeviceMesh.all_gather
    torch.distributed._tensor.device_mesh.DeviceMesh.all_gather = new_all_gather
    try:
        yield
    finally:
        torch.distributed._tensor.device_mesh.DeviceMeshall_gather = _orig_all_gather


def _check_cuda() -> None:
    assert torch.cuda.is_available(), f"Requires CUDA"
    assert torch.cuda.device_count() > 1, f"Requires at least 2 GPUs"


def _init_values(module: nn.Module, tp_module: nn.Module, mesh: DeviceMesh) -> None:
    """
    Initializes the parameter values for ``tp_module`` to match those of
    ``module``. This is used for checking parity.

    NOTE: For ``CausalSelfAttention``, we must initialize the QKV weight and
    bias specially. See ``_init_qkv_params_from_tensors`` for the details.
    """
    for m1, m2 in zip(module.modules(), tp_module.modules()):
        for (n1, p1), (n2, p2) in zip(
            m1.named_parameters(recurse=False), m2.named_parameters(recurse=False)
        ):
            assert n1 == n2, f"{n1} {n2}"
            if isinstance(p2, DTensor):
                new_p2_replicated = DTensor.from_local(
                    p1.clone(), mesh, [Replicate()], run_check=False
                )
                new_p2 = nn.Parameter(
                    new_p2_replicated.redistribute(
                        device_mesh=mesh, placements=p2._spec.placements
                    )
                )
                setattr(m2, n2, new_p2)
            else:
                with torch.no_grad():
                    p2.copy_(p1)
    for m1, m2 in zip(module.modules(), tp_module.modules()):
        if isinstance(m1, CausalSelfAttention):
            assert isinstance(m2, (TPCausalSelfAttention, CausalSelfAttention))
            (
                m2.c_attn.weight,
                m2.c_attn.bias,
            ) = _init_qkv_params_from_tensors(
                src_weight=m1.c_attn.weight,
                src_bias=m1.c_attn.bias,
                dst_weight=m2.c_attn.weight,
                dst_bias=m2.c_attn.bias,
                mesh=device_mesh,
            )


def _check_unsharded_params(
    module: nn.Module, tp_module: nn.Module, mesh: DeviceMesh
) -> None:
    for (n1, p1), (n2, p2) in zip(
        module.named_parameters(), tp_module.named_parameters()
    ):
        assert n1 == n2, f"{n1} {n2}"
        if isinstance(p2, DTensor):
            p2 = p2.redistribute(device_mesh=mesh, placements=[Replicate()]).to_local()
        if not torch.all(torch.isclose(p1, p2)):
            print(f"Mismatch for {n1}!")
        torch.testing.assert_close(p1, p2)


def _get_block_inp() -> Tuple[torch.Tensor]:
    return (torch.randn((batch_size, block_size, n_embd), device="cuda"),)


def _get_gpt_inp() -> Tuple[torch.Tensor]:
    x = torch.randint(50257, (batch_size, block_size), device="cuda")
    y = torch.randint(50257, (batch_size, block_size), device="cuda")
    return (x, y)


def _check_tp_forward(
    device_mesh: DeviceMesh,
    module: nn.Module,
    tp_module: nn.Module,
    expected_num_all_reduces: int,
    expected_num_all_gathers: int,
    get_input_fn: Callable,
):
    """
    Checks the tensor parallel ``tp_module`` 's forward against that of
    ``module`` for parity. This also includes checking all-reduce and
    all-gather counts.
    """
    # Use the same seed on all ranks to replicate the input tensor
    torch.manual_seed(seed)
    inp = get_input_fn()
    out = module(*inp)
    counter = Counter()
    orig_all_reduce = torch.distributed._tensor.device_mesh.DeviceMesh.all_reduce
    orig_all_gather = torch.distributed._tensor.device_mesh.DeviceMesh.all_gather
    all_reduce_with_count = functools.partial(
        collective_with_count, device_mesh, orig_all_reduce, counter
    )
    all_gather_with_count = functools.partial(
        collective_with_count, device_mesh, orig_all_gather, counter
    )
    with patch_device_mesh_all_reduce(
        all_reduce_with_count
    ), patch_device_mesh_all_gather(all_gather_with_count):
        tp_out = tp_module(*inp)
    assert (
        counter[orig_all_reduce] == expected_num_all_reduces
    ), f"{counter[orig_all_reduce]}"
    assert (
        counter[orig_all_gather] == expected_num_all_gathers
    ), f"{counter[orig_all_gather]}"
    torch.testing.assert_close(tp_out, out)


def test_tp_self_attn_module(config: GPTConfig):
    """Tests ``TPCausalSelfAttention`` against ``CausalSelfAttention``."""
    _check_cuda()
    torch.manual_seed(seed)
    tp_self_attn = TPCausalSelfAttention(config, device_mesh).cuda()
    self_attn = CausalSelfAttention(config).cuda()
    _init_values(self_attn, tp_self_attn, device_mesh)
    _check_tp_forward(device_mesh, self_attn, tp_self_attn, 1, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_self_attn_module!")


def test_tp_self_attn_parallelize(config: GPTConfig):
    """
    Tests ``parallelize_causal_self_attention`` applied to
    ``CausalSelfAttention`` against ``CausalSelfAttention``.
    """
    _check_cuda()
    torch.manual_seed(seed)
    self_attn = CausalSelfAttention(config).cuda()
    tp_self_attn = parallelize_causal_self_attention(
        CausalSelfAttention(config).cuda(), device_mesh
    )
    _init_values(self_attn, tp_self_attn, device_mesh)
    _check_tp_forward(device_mesh, self_attn, tp_self_attn, 1, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_self_attn_parallelize!")


def test_tp_mlp_module(config: GPTConfig):
    """Tests ``TPMLP`` against ``MLP``."""
    _check_cuda()
    torch.manual_seed(seed)
    tp_mlp = TPMLP(config, device_mesh).cuda()
    mlp = MLP(config).cuda()
    _init_values(mlp, tp_mlp, device_mesh)
    _check_tp_forward(device_mesh, mlp, tp_mlp, 1, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_mlp_module!")


def test_tp_mlp_parallelize(config: GPTConfig):
    """Tests ``parallelize_mlp`` applied to ``MLP`` against ``MLP``."""
    _check_cuda()
    torch.manual_seed(seed)
    mlp = MLP(config).cuda()
    tp_mlp = parallelize_mlp(MLP(config).cuda(), device_mesh)
    _init_values(mlp, tp_mlp, device_mesh)
    _check_tp_forward(device_mesh, mlp, tp_mlp, 1, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_mlp_parallelize!")


def test_tp_block_module(config: GPTConfig):
    """Tests ``TPBlock`` against ``Block``."""
    _check_cuda()
    torch.manual_seed(seed)
    tp_block = TPBlock(config, device_mesh).cuda()
    block = Block(config).cuda()
    _init_values(block, tp_block, device_mesh)
    _check_tp_forward(device_mesh, block, tp_block, 2, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_block_module!")


def test_tp_block_parallelize(config: GPTConfig):
    """Tests ``parallelize_block`` applied to ``Block`` against ``Block``."""
    _check_cuda()
    torch.manual_seed(seed)
    block = Block(config).cuda()
    tp_block = parallelize_block(Block(config).cuda(), device_mesh)
    _init_values(block, tp_block, device_mesh)
    _check_tp_forward(device_mesh, block, tp_block, 2, 0, _get_block_inp)
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_block_parallelize!")


def test_tp_gpt_parallelize(config: GPTConfig):
    _check_cuda()
    torch.manual_seed(seed)
    gpt = GPT(config).cuda()
    tp_gpt = parallelize_gpt(GPT(config).cuda(), device_mesh)
    _init_values(gpt, tp_gpt, device_mesh)
    expected_num_all_reduce = config.n_layer * 2
    _check_tp_forward(
        device_mesh, gpt, tp_gpt, expected_num_all_reduce, 0, _get_gpt_inp
    )
    dist.barrier()
    if dist.get_rank() == 0:
        print(f"Passed test_tp_gpt_parallelize!")


test_tp_self_attn_module(config)
test_tp_mlp_module(config)
test_tp_block_module(config)
test_tp_self_attn_parallelize(config)
test_tp_mlp_parallelize(config)
test_tp_block_parallelize(config)
test_tp_gpt_parallelize(config)
