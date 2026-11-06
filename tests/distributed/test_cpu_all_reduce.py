# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for cpu_all_reduce_ (gloo, CPU only)."""

import time
from datetime import timedelta

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.distributed.parallel_state import GroupCoordinator
from vllm.distributed.utils import (
    allocate_group_ports,
    cpu_all_reduce_,
    fetch_group_ports,
    stateless_init_torch_distributed_process_group,
)

pytestmark = pytest.mark.cpu_test


def _make_input(rank: int, world_size: int, step: int) -> torch.Tensor:
    # Same shape as the V2 DP metadata tensor; values differ per rank/step.
    t = torch.zeros(6, world_size, dtype=torch.int32)
    t[:, rank] = torch.arange(6, dtype=torch.int32) * (rank + 1) + step
    return t


def _stateless_gloo_group(store: dist.Store, rank: int, world_size: int):
    # Rank 0 binds the port itself and keeps the socket open until the
    # TCPStore takes it over. A port probed with get_open_port() is released
    # before the ranks bind it, and the gloo groups' ephemeral listeners can
    # be given that port in between (EADDRINUSE).
    if rank == 0:
        (port,), (listen_socket,) = allocate_group_ports(
            store, "stateless_pg", "127.0.0.1", 1
        )
    else:
        (port,) = fetch_group_ports(store, "stateless_pg", 1)
        listen_socket = None
    return stateless_init_torch_distributed_process_group(
        "127.0.0.1",
        port,
        rank,
        world_size,
        backend="gloo",
        listen_socket=listen_socket,
    )


def _worker(rank: int, world_size: int, store_path: str):
    # File rendezvous instead of a probed TCP port, for the same reason as
    # in _stateless_gloo_group. The store also carries the stateless port.
    store = dist.FileStore(store_path, world_size)
    dist.init_process_group(
        "gloo",
        store=store,
        rank=rank,
        world_size=world_size,
    )
    coord = GroupCoordinator(
        group_ranks=[list(range(world_size))],
        local_rank=0,
        torch_distributed_backend="gloo",
        use_device_communicator=False,
        group_name="cpu_all_reduce_test",
    )

    # Several steps to exercise tag reuse across consecutive calls.
    for step in range(5):
        expected = _make_input(rank, world_size, step)
        dist.all_reduce(expected, group=coord.cpu_group)
        t = _make_input(rank, world_size, step)
        out = coord.cpu_all_reduce_(t)
        assert out is t
        torch.testing.assert_close(t, expected, rtol=0, atol=0)

    # Float sums must be bitwise identical on every rank.
    t = torch.randn(6, world_size, generator=torch.Generator().manual_seed(rank))
    expected = t.clone()
    dist.all_reduce(expected, group=coord.cpu_group)
    coord.cpu_all_reduce_(t)
    torch.testing.assert_close(t, expected)
    gathered = [torch.empty_like(t) for _ in range(world_size)]
    dist.all_gather(gathered, t, group=coord.cpu_group)
    for other in gathered:
        assert torch.equal(other, t)

    # Unregistered (stateless) process group, as used by the engine-core DP
    # group and elastic EP.
    stateless_pg = _stateless_gloo_group(store, rank, world_size)
    expected = _make_input(rank, world_size, 0)
    dist.all_reduce(expected)
    t = _make_input(rank, world_size, 0)
    cpu_all_reduce_(t, stateless_pg)
    torch.testing.assert_close(t, expected, rtol=0, atol=0)

    dist.destroy_process_group()


# Non-power-of-two sizes exercise the cross-block exchange.
@pytest.mark.parametrize("world_size", [2, 3, 4, 6, 8, 9, 12])
def test_cpu_all_reduce(world_size: int, tmp_path):
    mp.spawn(
        _worker,
        args=(world_size, str(tmp_path / "store")),
        nprocs=world_size,
        join=True,
    )


def _timeout_worker(rank: int, world_size: int, store_path: str):
    store = dist.FileStore(store_path, world_size)
    pg = _stateless_gloo_group(store, rank, world_size)
    if rank != 0:
        # Never joins the all-reduce; rank 0's p2p wait must time out
        # well before this peer goes away.
        time.sleep(5)
        return
    t = _make_input(rank, world_size, 0)
    start = time.monotonic()
    with pytest.raises(RuntimeError, match="Timed out"):
        cpu_all_reduce_(t, pg, timeout=timedelta(seconds=1))
    assert time.monotonic() - start < 3


def test_cpu_all_reduce_timeout(tmp_path):
    mp.spawn(_timeout_worker, args=(2, str(tmp_path / "store")), nprocs=2, join=True)
