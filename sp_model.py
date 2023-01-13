"""
This is a sequence parallel version of model.py.

The sequence parallelism follows the method outlined in "Sequence Parallelism:
Long Sequence Training from System Perspective".
https://arxiv.org/abs/2105.13120
"""
import math
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from common_model import BlockBase, GPTConfig, new_gelu

from model import Block, CausalSelfAttention, GPT, MLP


"""
B: batch size
H: number of attention heads
S: source sequence length
S': target sequence length; equal to S for self-attention
W: world size
D: hidden dimension
"""


def _ring_send_recv(
    send_tensor: torch.Tensor,
    pg: dist.ProcessGroup,
    rank: int,
    world_size: int,
) -> torch.Tensor:
    """
    Sends ``send_tensor`` to the next rank in the ring, and receives and
    returns a ``recv_tensor`` from the previous rank in the ring.
    """
    recv_tensor = torch.empty_like(send_tensor)
    send_dst_rank = (rank + 1) % world_size
    recv_src_rank = (rank - 1 + world_size) % world_size
    if rank % 2 == 0:
        dist.send(send_tensor, send_dst_rank, group=pg)
        dist.recv(recv_tensor, recv_src_rank, group=pg)
    else:
        dist.recv(recv_tensor, recv_src_rank, group=pg)
        dist.send(send_tensor, send_dst_rank, group=pg)
    return recv_tensor


class _RingQKT(torch.autograd.Function):
    """
    This defines the forward and backward for sequence parallel QK^T
    computation following a ring schedule.
    """

    @staticmethod
    def forward(
        ctx,
        Q_local: torch.Tensor,
        K_local: torch.Tensor,
        pg: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        Q_local: (B, H, S'/W, D)
        K_local: (B, H, S/W, D)
        Q_local @ K^T: (B, H, S'/W, S)

        Communicate K_local across ranks in a ring.
        """
        for i in (0, 1, 3):
            assert Q_local.size(i) == K_local.size(
                i
            ), f"Mismatched sizes! Q: {Q_local.size()} K: {K_local.size()}"
        batch_size, n_head, q_subseq_len, _ = Q_local.size()
        _, _, k_subseq_len, _ = K_local.size()
        K_local = K_local.contiguous()

        ctx.save_for_backward(Q_local, K_local)
        ctx.pg = pg
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)

        k_seq_len = k_subseq_len * world_size  # assume uniform
        Q_local_KT = torch.empty(
            (batch_size, n_head, q_subseq_len, k_seq_len),
            device=Q_local.device,
            dtype=Q_local.dtype,
        )
        k_local_start_idx = rank * k_subseq_len
        k_local_end_idx = k_local_start_idx + k_subseq_len
        Q_local_KT[
            :, :, :, k_local_start_idx:k_local_end_idx
        ] = Q_local @ K_local.transpose(-2, -1)
        for i in range(world_size - 1):
            K_local = _ring_send_recv(K_local, pg, rank, world_size)
            k_local_start_idx = ((rank - 1 - i) % world_size) * k_subseq_len
            k_local_end_idx = k_local_start_idx + k_subseq_len
            Q_local_KT[
                :, :, :, k_local_start_idx:k_local_end_idx
            ] = Q_local @ K_local.transpose(-2, -1)
        return Q_local_KT

    @staticmethod
    def backward(
        ctx,
        grad_Q_local_KT: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        grad_Q_local_KT: (B, H, S'/W, S)
        grad_Q_local: (B, H, S'/W, D)
        grad_K_local: (B, H, S/W, D)
        """
        Q_local, K_local = ctx.saved_tensors
        pg = ctx.pg
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)

        k_subseq_len = K_local.size(-2)
        k_local_start_idx = rank * k_subseq_len
        k_local_end_idx = k_local_start_idx + k_subseq_len

        # (B, H, S, D)
        grad_K = grad_Q_local_KT.transpose(-2, -1) @ Q_local
        # This rank's K_local contributes to every rank's Q_local_KT, so we
        # must all-reduce to sum the contributions.
        dist.all_reduce(grad_K, group=pg)
        # (B, H, S/W, D)
        grad_K_local = grad_K[:, :, k_local_start_idx:k_local_end_idx, :]

        # (B, H, S'/W, S/W) * (B, H, S/W, D) -> (B, H, S'/W, D)
        grad_Q_local = (
            grad_Q_local_KT[:, :, :, k_local_start_idx:k_local_end_idx] @ K_local
        )
        for i in range(world_size - 1):
            K_local = _ring_send_recv(K_local, pg, rank, world_size)
            k_local_start_idx = (rank - 1 - i) % world_size * k_subseq_len
            k_local_end_idx = k_local_start_idx + k_subseq_len
            grad_Q_local += (
                grad_Q_local_KT[:, :, :, k_local_start_idx:k_local_end_idx] @ K_local
            )
        return grad_Q_local, grad_K_local, None


def _ring_qkt(
    Q_local: torch.Tensor,
    K_local: torch.Tensor,
    pg: Optional[dist.ProcessGroup] = None,
):
    pg = pg or torch.distributed_c10d._get_default_group()
    return _RingQKT.apply(Q_local, K_local, pg)


class _RingAV(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        A_local: torch.Tensor,
        V_local: torch.Tensor,
        pg: dist.ProcessGroup,
    ) -> torch.Tensor:
        """
        A_local: (B, H, S'/W, S)
        V_local: (B, H, S/W, D)
        A_local V: (B, H, S'/W, D)

        Communicate V_local across ranks in a ring.
        """
        for i in (0, 1):
            assert A_local.size(i) == V_local.size(
                i
            ), f"Mismatched sizes! A: {A_local.size()} V: {V_local.size()}"
        _, _, _, v_seq_len = A_local.size()
        _, _, v_subseq_len, _ = V_local.size()
        V_local = V_local.contiguous()

        ctx.save_for_backward(A_local, V_local)
        ctx.pg = pg
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)

        assert v_seq_len == v_subseq_len * world_size, (
            f"Expects sequence length {v_seq_len} to be subsequence length "
            f"{v_subseq_len} times world size {world_size} for A: "
            f"{A_local.shape} and V: {V_local.shape}"
        )  # assume uniform

        v_local_start_idx = rank * v_subseq_len
        v_local_end_idx = v_local_start_idx + v_subseq_len
        # (B, H, S'/W, S/W) * (B, H, S/W, D) -> (B, H, S'/W, D)
        A_local_V = A_local[:, :, :, v_local_start_idx:v_local_end_idx] @ V_local
        for i in range(world_size - 1):
            V_local = _ring_send_recv(V_local, pg, rank, world_size)
            v_local_start_idx = ((rank - 1 - i) % world_size) * v_subseq_len
            v_local_end_idx = v_local_start_idx + v_subseq_len
            A_local_V += A_local[:, :, :, v_local_start_idx:v_local_end_idx] @ V_local
        return A_local_V

    @staticmethod
    def backward(
        ctx,
        grad_A_local_V: torch.Tensor,
    ) -> Tuple[Optional[torch.Tensor], ...]:
        """
        grad_A_local_V: (B, H, S'/W, D)
        grad_A_local: (B, H, S'/W, S)
        grad_V_local: (B, H, S/W, D)
        """
        A_local, V_local = ctx.saved_tensors
        pg = ctx.pg
        rank = dist.get_rank(pg)
        world_size = dist.get_world_size(pg)

        v_subseq_len = V_local.size(-2)
        v_local_start_idx = rank * v_subseq_len
        v_local_end_idx = v_local_start_idx + v_subseq_len

        # (B, H, S, D)
        grad_V = A_local.transpose(-2, -1) @ grad_A_local_V
        # This rank's Q_local contributes to every rank's A_local_V, so we must
        # all-reduce to sum the contributions.
        dist.all_reduce(grad_V, group=pg)
        # (B, H, S/W, D)
        grad_V_local = grad_V[:, :, v_local_start_idx:v_local_end_idx, :]

        # (B, H, S'/W, S)
        grad_A_local = torch.empty_like(A_local)
        # (B, H, S'/W, D) * (B, H, D, S/W) -> (B, H, S'/W, S/W)
        grad_A_local[
            :, :, :, v_local_start_idx:v_local_end_idx
        ] = grad_A_local_V @ V_local.transpose(-2, -1)
        for i in range(world_size - 1):
            V_local = _ring_send_recv(V_local, pg, rank, world_size)
            v_local_start_idx = ((rank - 1 - i) % world_size) * v_subseq_len
            v_local_end_idx = v_local_start_idx + v_subseq_len
            grad_A_local[
                :, :, :, v_local_start_idx:v_local_end_idx
            ] = grad_A_local_V @ V_local.transpose(-2, -1)
        return grad_A_local, grad_V_local, None


def _ring_av(
    A_local: torch.Tensor,
    V_local: torch.Tensor,
    pg: Optional[dist.ProcessGroup] = None,
):
    pg = pg or torch.distributed_c10d._get_default_group()
    return _RingAV.apply(A_local, V_local, pg)


class SPCausalSelfAttention(nn.Module):
    def __init__(
        self, config: GPTConfig, pg: Optional[dist.ProcessGroup] = None
    ) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.pg = pg or dist.distributed_c10d._get_default_group()
        self.rank = dist.get_rank(pg)
        self.world_size = dist.get_world_size(pg)

        self.c_attn = nn.Linear(self.n_embd, 3 * self.n_embd)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        causal_mask = torch.tril(torch.ones(config.block_size, config.block_size)).view(
            1, 1, config.block_size, config.block_size
        )
        self.register_buffer("bias", causal_mask)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        For self-attention, S' = S.
        """
        assert isinstance(x, torch.Tensor), f"Expects Tensor got {type(x)}"
        assert x.ndim == 3, f"Expects 3D input but got {x.shape}"
        (
            batch_size,
            subseq_len,
            n_embd,
        ) = x.size()
        seq_len = subseq_len * self.world_size
        assert (
            n_embd == self.n_embd
        ), f"Expects {self.n_embd} for rightmost dim but got {n_embd}"
        n_head = self.n_head

        qkv = self.c_attn(x)  # (batch_size, subseq_len, 3 * n_embd)
        # (batch_size, subseq_len, n_embd) for each of Q, K, V
        q, k, v = qkv.split(n_embd, dim=2)
        # (batch_size, n_head, subseq_len, n_embd / n_head)
        q = q.view(batch_size, subseq_len, n_head, n_embd // n_head).transpose(1, 2)
        k = k.view(batch_size, subseq_len, n_head, n_embd // n_head).transpose(1, 2)
        v = v.view(batch_size, subseq_len, n_head, n_embd // n_head).transpose(1, 2)
        # (batch_size, n_head, subseq_len, seq_len)
        attn = _ring_qkt(q, k, self.pg) * (1.0 / math.sqrt(k.size(-1)))
        start_idx = self.rank * subseq_len
        end_idx = start_idx + subseq_len
        attn = attn.masked_fill(
            self.bias[:, :, start_idx:end_idx, :seq_len] == 0, float("-inf")
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        # (batch_size, n_head, subseq_len, n_embd / n_head)
        y = _ring_av(attn, v, self.pg)
        y = y.transpose(1, 2).contiguous().view(batch_size, subseq_len, n_embd)
        # (batch_size, block_size, n_embd)
        y = self.c_proj(y)
        y = self.resid_dropout(y)
        return y
