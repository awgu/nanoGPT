import functools
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn

from model import Block, CausalSelfAttention, MLP
from torch.distributed._composable import checkpoint, fully_shard
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
    checkpoint_wrapper,
    CheckpointImpl,
)
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import ModuleWrapPolicy
from torch.nn.parallel import DistributedDataParallel as DDP
from tp_model import parallelize_gpt


def dist_setup():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)


def get_rank() -> Optional[int]:
    try:
        rank = torch.distributed.get_rank()
    except RuntimeError:
        rank = None
    return rank


def apply_ddp(model: nn.Module) -> nn.Module:
    """
    Applies DDP to ``model``. We enable ``gradient_as_bucket_view`` as a
    performance optimization.
    """
    model = DDP(model, device_ids=[dist.get_rank()], gradient_as_bucket_view=True)
    return model


def apply_fsdp(model: nn.Module) -> nn.Module:
    """
    Applies FSDP to ``model``.

    For gpt2, applying FSDP to each ``Block`` with ``NO_SHARD`` suffices.
    For gpt2-xl,
    """
    # module_classes = {CausalSelfAttention, MLP}
    module_classes = {Block}
    auto_wrap_policy = ModuleWrapPolicy(module_classes)
    bf16_mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        # cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        # "sharding_strategy": ShardingStrategy.NO_SHARD,
        "sharding_strategy": ShardingStrategy.SHARD_GRAD_OP,
        # "sharding_strategy": ShardingStrategy.FULL_SHARD,
        "use_orig_params": True,  # needed for optimizer
        "limit_all_gathers": True,
    }

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        # mixed_precision=bf16_mp,
        **fsdp_kwargs,
    )
    return model


def apply_tp(model: nn.Module) -> nn.Module:
    world_size = torch.distributed.get_world_size()
    device_mesh = torch.distributed._tensor.DeviceMesh(
        "cuda", torch.arange(world_size)
    )  # 1D device mesh
    return parallelize_gpt(model, device_mesh)


def apply_ac(model: nn.Module) -> nn.Module:
    module_classes = (CausalSelfAttention, MLP)
    non_reentrant_wrapper = functools.partial(
        checkpoint_wrapper,
        offload_to_cpu=False,
        checkpoint_impl=CheckpointImpl.NO_REENTRANT,
    )
    check_fn = lambda submodule: isinstance(submodule, module_classes)
    apply_activation_checkpointing(
        model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
    )
    return model


def apply_composable(model: nn.Module) -> nn.Module:
    bf16_mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        # cast_forward_inputs=True,
    )
    fully_shard_kwargs = {
        "strategy": ShardingStrategy.FULL_SHARD,
        "mixed_precision": bf16_mp,
    }
    for i, block in enumerate(model.transformer["h"]):
        checkpoint(block)
        fully_shard(block, **fully_shard_kwargs)
        # block = checkpoint_wrapper(block, CheckpointImpl.REENTRANT)
        # block = FSDP(
        #     block, use_orig_params=True, limit_all_gathers=True, **fully_shard_kwargs
        # )
        # model.transformer["h"] = block
    fully_shard(model, **fully_shard_kwargs)
    # model = FSDP(
    #     model, use_orig_params=True, limit_all_gathers=True, **fully_shard_kwargs
    # )
    # return torch.compile(model)
    return model
