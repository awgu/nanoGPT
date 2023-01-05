import pickle
from typing import Optional

import torch
from dist_utils import get_rank

OOM_SNAPSHOT_FILENAME = "oom_snapshot"


def init_oom_observer(
    trace_alloc_record_context: bool = False,
    trace_alloc_max_entries: int = 100000,
    save_on_all_ranks: bool = False,
) -> None:
    """
    Args:
        trace_alloc_record_context (bool): If ``True``, then saves allocation
            trace information with the snapshot. (Default: ``False``)
        trace_alloc_max_entries (int): Specifies the max number of allocation
            entries to save in the snapshot (like the history length). This is
            ignored if ``trace_alloc_record_context == False``. (Default:
            100000)
        save_on_all_ranks (bool): If ``True``, then saves a snapshot on all
            ranks; if ``False``, then only saves a snapshot on rank 0. If
            setting to ``False``, be careful that OOM does not need to be
            uniform across ranks. This argument is vacuous if not running in a
            distributed setting. (Default: ``False``)
    """
    rank: Optional[int] = get_rank()
    snapshot_filename = (
        OOM_SNAPSHOT_FILENAME + f"_{rank}" if rank else OOM_SNAPSHOT_FILENAME
    ) + ".pickle"
    kwargs = (
        {
            "trace_alloc_record_context": trace_alloc_record_context,
            "trace_alloc_max_entries": trace_alloc_max_entries,
        }
        if trace_alloc_record_context
        else {}
    )
    torch.cuda.memory._record_memory_history(True, **kwargs)

    def oom_observer(device, alloc, device_alloc, device_free):
        if not save_on_all_ranks and rank is not None and rank > 0:
            return
        print(f"Saving memory snapshot during OOM")
        snapshot = torch.cuda.memory._snapshot()
        pickle.dump(snapshot, open(snapshot_filename, "wb"))

    torch._C._cuda_attach_out_of_memory_observer(oom_observer)
