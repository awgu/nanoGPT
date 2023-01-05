import os
import time
from typing import Any, Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from dist_utils import get_rank

from oom_observer import init_oom_observer
from torch.distributed._tools import MemoryTracker


PROFILE_SAVE_DIR = "./bench_log"


def init_benchmark_get_batch_fn(
    batch_size: int,
    block_size: int,
    device: torch.device,
    use_real_data: bool,
    use_same_seed: bool,
) -> Callable:
    """
    Returns the ``get_batch`` function used for benchmarking. The ``split``
    argument for ``get_batch`` is ignored for benchmarking, as we always use
    the training data directly when using real data.

    TODO (awgu): Consider refactoring further.
    """
    if use_real_data:
        dataset = "openwebtext"
        data_dir = os.path.join("data", dataset)
        train_data = np.memmap(
            os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
        )

        def get_batch(split):
            data = train_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack(
                [
                    torch.from_numpy((data[i : i + block_size]).astype(np.int64))
                    for i in ix
                ]
            )
            y = torch.stack(
                [
                    torch.from_numpy(
                        (data[i + 1 : i + 1 + block_size]).astype(np.int64)
                    )
                    for i in ix
                ]
            )
            x, y = x.to(device), y.to(device)
            return x, y

    else:
        if use_same_seed:
            torch.manual_seed(42)
        # TODO (awgu): Get rid of hard-coded vocab size?
        x = torch.randint(50257, (batch_size, block_size), device=device)
        y = torch.randint(50257, (batch_size, block_size), device=device)
        get_batch = lambda split: (x, y)
    return get_batch


def _run_train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step_idx: int,
    num_steps: int,
    get_batch: Callable,
    autocast_dtype: torch.dtype,
):
    X, Y = get_batch("train")
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        _, loss = model(X, Y)
    # TODO (awgu): Why not before forward / after optimizer step?
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    # Only print the loss on the last iteration to avoid the CPU sync
    if step_idx == num_steps:
        lossf = loss.item()
        print(f"{step_idx}/{num_steps} loss: {lossf:.4f}")


def benchmark_with_profiler(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    get_batch: Callable,
    autocast_dtype: torch.dtype,
) -> None:
    """
    PyTorch profiler:
    - Tutorial: https://pytorch.org/tutorials/intermediate/tensorboard_profiler_tutorial.html
    - API: https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile
    """
    wait, warmup, active = 5, 5, 2
    num_steps = wait + warmup + active
    rank = get_rank()
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
            _run_train_step(
                model, optimizer, step_idx, num_steps, get_batch, autocast_dtype
            )
            if rank is None or rank == 0:
                prof.step()  # notify the profiler at end of each step


def benchmark(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    get_batch: Callable,
    autocast_dtype: torch.dtype,
    track_memory: bool,
    use_oom_observer: bool,
    oom_observer_kwargs: Dict[str, Any],
) -> None:
    rank = get_rank()
    if use_oom_observer:
        init_oom_observer(**oom_observer_kwargs)
    torch.cuda.synchronize()
    for stage, num_steps in enumerate([10, 20]):  # burnin, then benchmark
        t0 = time.time()
        # Only track memory for the 1st iteration after burnin on rank 0
        if track_memory and stage == 1 and not rank:
            tracker = MemoryTracker()
            tracker.start_monitor(model)
        for step_idx in range(1, num_steps + 1):
            _run_train_step(
                model, optimizer, step_idx, num_steps, get_batch, autocast_dtype
            )
            if track_memory and stage == 1 and not rank and step_idx == 0:
                tracker.stop()
                tracker.summary()
    torch.cuda.synchronize()
    t1 = time.time()
    if stage == 1:
        print(f"time per iteration: {(t1-t0)/num_steps*1000:.4f}ms")
