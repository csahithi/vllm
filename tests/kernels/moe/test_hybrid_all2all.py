# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid MoE all-to-all (VLLM_MOE_HYBRID_A2A_TOKENS): the per-call switch to
the large-batch prepare/finalize and its setup, with kernels, process groups
and the forward context faked out."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, sentinel

import pytest
import torch

import vllm.model_executor.layers.fused_moe.all2all_utils as all2all_utils
import vllm.model_executor.layers.fused_moe.modular_kernel as mk
import vllm.model_executor.layers.fused_moe.oracle.nvfp4 as nvfp4_oracle
from vllm.config import CUDAGraphMode
from vllm.distributed import GroupCoordinator
from vllm.distributed.device_communicators.all2all import AgRsAll2AllManager
from vllm.forward_context import DPMetadata
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.all2all_utils import (
    maybe_add_hybrid_all2all,
)
from vllm.model_executor.layers.fused_moe.config import (
    FUSED_MOE_UNQUANTIZED_CONFIG,
    FusedMoEConfig,
    FusedMoEParallelConfig,
    RoutingMethodType,
)
from vllm.model_executor.layers.fused_moe.prepare_finalize import naive_dp_ep
from vllm.model_executor.layers.fused_moe.prepare_finalize.naive_dp_ep import (
    MoEPrepareAndFinalizeNaiveDPEPModular,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)

THRESHOLD = 128


class _Experts(mk.FusedMoEExpertsModular):
    """Modular experts without kernels: enough to build a FusedMoEKernel."""

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return True

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(weight_key, activation_key) -> bool:
        return True

    @staticmethod
    def _supports_activation(activation) -> bool:
        return True

    @staticmethod
    def _supports_parallel_config(moe_parallel_config) -> bool:
        return True

    def workspace_shapes(self, *args, **kwargs):
        raise NotImplementedError

    def finalize_weight_and_reduce_impl(self):
        raise NotImplementedError

    def apply(self, *args, **kwargs):
        raise NotImplementedError


def _moe_config(
    all2all_backend: str = "flashinfer_nvlink_one_sided",
    dp_size: int = 2,
    tp_size: int = 1,
    sequence_parallel: bool = False,
    is_lora_enabled: bool = False,
) -> FusedMoEConfig:
    """A DP+EP layer: the experts are split over all DP x TP ranks and, with
    sequence-parallel MoE, each DP rank's tokens over its TP ranks."""
    ep_size = dp_size * tp_size
    return FusedMoEConfig(
        num_experts=8,
        experts_per_token=2,
        hidden_dim=128,
        intermediate_size=256,
        num_local_experts=8 // ep_size,
        num_logical_experts=8,
        activation=MoEActivation.SILU,
        device="cpu",
        routing_method=RoutingMethodType.TopK,
        moe_parallel_config=FusedMoEParallelConfig(
            tp_size=1,
            pcp_size=1,
            dp_size=dp_size,
            ep_size=ep_size,
            tp_rank=0,
            pcp_rank=0,
            dp_rank=0,
            ep_rank=0,
            sp_size=tp_size if sequence_parallel else 1,
            use_ep=True,
            all2all_backend=all2all_backend,
            enable_eplb=False,
        ),
        in_dtype=torch.bfloat16,
        is_lora_enabled=is_lora_enabled,
    )


def _prepare_finalize(
    supports_async: bool = False,
    activation_format: mk.FusedMoEActivationFormat = (
        mk.FusedMoEActivationFormat.Standard
    ),
) -> MagicMock:
    prepare_finalize = MagicMock(spec=mk.FusedMoEPrepareAndFinalizeModular)
    prepare_finalize.supports_async.return_value = supports_async
    prepare_finalize.activation_format = activation_format
    return prepare_finalize


def _kernel(moe_config: FusedMoEConfig | None = None) -> mk.FusedMoEKernel:
    """A modular kernel in place of one-sided prepare/finalize and experts."""
    experts = _Experts(moe_config or _moe_config(), FUSED_MOE_UNQUANTIZED_CONFIG)
    return mk.FusedMoEKernel(_prepare_finalize(), experts)


def _hybrid_kernel() -> mk.FusedMoEKernel:
    """A kernel that runs calls with over THRESHOLD tokens on some EP rank
    through a second prepare/finalize."""
    kernel = _kernel()
    kernel.set_large_batch_prepare_finalize(_prepare_finalize(), THRESHOLD)
    return kernel


def _monolithic_kernel() -> mk.FusedMoEKernel:
    prepare_finalize = MagicMock(spec=mk.FusedMoEPrepareAndFinalizeMonolithic)
    # Otherwise like _prepare_finalize(), so only being monolithic sets it apart.
    prepare_finalize.supports_async.return_value = False
    prepare_finalize.activation_format = mk.FusedMoEActivationFormat.Standard
    experts = MagicMock(spec=mk.FusedMoEExpertsMonolithic)
    experts.activation_format.return_value = mk.FusedMoEActivationFormat.Standard
    return mk.FusedMoEKernel(prepare_finalize, experts)


def _set_dp_metadata(monkeypatch: pytest.MonkeyPatch, dp_metadata: object) -> None:
    """Run the kernel's calls in a forward context with this DP metadata."""
    context = SimpleNamespace(dp_metadata=dp_metadata)
    monkeypatch.setattr(mk, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(mk, "get_forward_context", lambda: context)


@pytest.mark.parametrize(
    "local_sizes,use_large",
    [
        pytest.param([THRESHOLD, THRESHOLD], False, id="at-threshold"),
        pytest.param([THRESHOLD + 1, THRESHOLD], True, id="above-threshold"),
        # Every EP rank must take the same path, so the largest count decides,
        # not this rank's (dp_rank 0) own.
        pytest.param([0, THRESHOLD + 1], True, id="above-on-other-rank"),
        pytest.param(None, False, id="no-local-sizes"),
    ],
)
def test_select_modular_impl_by_largest_local_size(monkeypatch, local_sizes, use_large):
    kernel = _hybrid_kernel()
    _set_dp_metadata(monkeypatch, SimpleNamespace(local_sizes=local_sizes))
    expected = kernel._large_batch_impl if use_large else kernel.impl
    assert kernel._select_modular_impl() is expected


def test_select_modular_impl_without_dp_metadata(monkeypatch):
    kernel = _hybrid_kernel()
    _set_dp_metadata(monkeypatch, None)
    assert kernel._select_modular_impl() is kernel.impl


def test_select_modular_impl_without_forward_context(monkeypatch):
    kernel = _hybrid_kernel()
    monkeypatch.setattr(mk, "is_forward_context_available", lambda: False)
    assert kernel._select_modular_impl() is kernel.impl


def test_select_modular_impl_without_large_batch_impl(monkeypatch):
    kernel = _kernel()
    _set_dp_metadata(monkeypatch, SimpleNamespace(local_sizes=[THRESHOLD + 1]))
    assert kernel._select_modular_impl() is kernel.impl


def test_select_modular_impl_compares_per_ep_rank_sizes(monkeypatch):
    """The runner sets local_sizes for the MoE call. With sequence-parallel
    MoE each EP rank holds a slice of its DP rank's tokens, and the slices
    decide."""
    kernel = _hybrid_kernel()
    dp_metadata = DPMetadata(num_tokens_across_dp_cpu=torch.tensor([2 * THRESHOLD, 1]))
    _set_dp_metadata(monkeypatch, dp_metadata)
    # 2 * THRESHOLD tokens over 2 SP ranks: THRESHOLD on each EP rank.
    with dp_metadata.sp_local_sizes(sequence_parallel_size=2):
        assert kernel._select_modular_impl() is kernel.impl
    with dp_metadata.sp_local_sizes(sequence_parallel_size=1):
        assert kernel._select_modular_impl() is kernel._large_batch_impl
    # Outside the MoE call no per-EP-rank sizes are set.
    assert kernel._select_modular_impl() is kernel.impl


def test_select_modular_impl_drops_large_batch_impl_with_replaced_experts(
    monkeypatch,
):
    kernel = _hybrid_kernel()
    # warning_once is cached per message, so count the calls instead.
    warning_once = MagicMock()
    monkeypatch.setattr(mk.logger, "warning_once", warning_once)
    _set_dp_metadata(monkeypatch, SimpleNamespace(local_sizes=[THRESHOLD + 1]))
    # As FusedMoEModularMethod does when a weight reload builds new experts.
    new_impl = _kernel().impl
    kernel.impl = new_impl
    assert kernel._select_modular_impl() is new_impl
    assert kernel._large_batch_impl is None
    assert kernel._select_modular_impl() is new_impl
    warning_once.assert_called_once()


@pytest.mark.parametrize("use_large", [False, True])
def test_apply_runs_selected_impl(monkeypatch, use_large):
    kernel = _hybrid_kernel()
    primary_apply, large_apply = MagicMock(), MagicMock()
    monkeypatch.setattr(kernel.impl, "apply", primary_apply)
    monkeypatch.setattr(kernel._large_batch_impl, "apply", large_apply)
    num_tokens = THRESHOLD + 1 if use_large else THRESHOLD
    _set_dp_metadata(monkeypatch, SimpleNamespace(local_sizes=[num_tokens]))
    kwargs = dict(
        hidden_states=sentinel.hidden_states,
        w1=sentinel.w1,
        w2=sentinel.w2,
        topk_weights=sentinel.topk_weights,
        topk_ids=sentinel.topk_ids,
        activation=sentinel.activation,
        global_num_experts=sentinel.global_num_experts,
        expert_map=sentinel.expert_map,
        apply_router_weight_on_input=sentinel.apply_router_weight_on_input,
        shared_experts=sentinel.shared_experts,
        shared_experts_input=sentinel.shared_experts_input,
    )
    selected, other = (
        (large_apply, primary_apply) if use_large else (primary_apply, large_apply)
    )
    assert kernel.apply(**kwargs) is selected.return_value
    selected.assert_called_once_with(**kwargs)
    other.assert_not_called()


def test_set_large_batch_prepare_finalize():
    kernel = _kernel()
    primary = kernel.prepare_finalize
    prepare_finalize = _prepare_finalize()
    kernel.set_large_batch_prepare_finalize(prepare_finalize, THRESHOLD)

    large_impl = kernel._large_batch_impl
    assert large_impl is not None
    assert large_impl.prepare_finalize is prepare_finalize
    # One experts object, and so one copy of the weights, for both paths.
    assert large_impl.fused_experts is kernel.fused_experts
    prepare_finalize.post_init_setup.assert_called_once_with(kernel.fused_experts)
    assert kernel._large_batch_threshold == THRESHOLD
    # What is read outside the MoE op still comes from the primary.
    assert kernel.prepare_finalize is primary


@pytest.mark.parametrize(
    "threshold,mismatch",
    [
        pytest.param(0, {}, id="zero-threshold"),
        pytest.param(-1, {}, id="negative-threshold"),
        pytest.param(THRESHOLD, {"supports_async": True}, id="async"),
        pytest.param(
            THRESHOLD,
            {"activation_format": mk.FusedMoEActivationFormat.BatchedExperts},
            id="activation-format",
        ),
    ],
)
def test_set_large_batch_prepare_finalize_rejects(threshold, mismatch):
    kernel = _kernel()
    with pytest.raises(AssertionError):
        kernel.set_large_batch_prepare_finalize(
            _prepare_finalize(**mismatch), threshold
        )
    assert kernel._large_batch_impl is None


def test_set_large_batch_prepare_finalize_rejects_monolithic_kernel():
    kernel = _monolithic_kernel()
    with pytest.raises(AssertionError):
        kernel.set_large_batch_prepare_finalize(_prepare_finalize(), THRESHOLD)


@pytest.fixture
def hybrid_env(monkeypatch):
    """What maybe_add_hybrid_all2all needs to add the AgRs path: the variable
    set, an AgRs manager on the EP group that communicates over a group with
    PyNccl, TP=1 and no CUDA graphs. Tests take one piece away at a time."""
    monkeypatch.setenv("VLLM_MOE_HYBRID_A2A_TOKENS", str(THRESHOLD))
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "0")
    comm_group = SimpleNamespace(
        unique_name="dp:0",
        device_communicator=SimpleNamespace(
            pynccl_comm=SimpleNamespace(disabled=False)
        ),
    )
    manager = MagicMock(spec=AgRsAll2AllManager)
    manager.world_size = 2
    manager._get_comm_group.return_value = comm_group
    ep_group = SimpleNamespace(
        device_communicator=SimpleNamespace(hybrid_all2all_manager=manager)
    )
    compilation_config = SimpleNamespace(
        cudagraph_mode=CUDAGraphMode.NONE, max_cudagraph_capture_size=512
    )
    monkeypatch.setattr(all2all_utils, "get_ep_group", lambda: ep_group)
    monkeypatch.setattr(
        all2all_utils,
        "get_current_vllm_config",
        lambda: SimpleNamespace(compilation_config=compilation_config),
    )
    monkeypatch.setattr(
        all2all_utils, "get_tensor_model_parallel_world_size", lambda: 1
    )
    return SimpleNamespace(
        manager=manager,
        comm_group=comm_group,
        ep_group=ep_group,
        compilation_config=compilation_config,
    )


def _add_hybrid_all2all(
    moe_config: FusedMoEConfig | None = None,
) -> mk.FusedMoEKernel:
    kernel = _kernel(moe_config)
    maybe_add_hybrid_all2all(kernel, kernel.moe_config)
    return kernel


@pytest.mark.parametrize("tp_size,sequence_parallel", [(1, False), (2, True)])
def test_maybe_add_hybrid_all2all(monkeypatch, hybrid_env, tp_size, sequence_parallel):
    monkeypatch.setattr(
        all2all_utils, "get_tensor_model_parallel_world_size", lambda: tp_size
    )
    moe_config = _moe_config(tp_size=tp_size, sequence_parallel=sequence_parallel)
    manager = hybrid_env.manager
    manager.world_size = moe_config.ep_size
    kernel = _kernel(moe_config)
    primary = kernel.prepare_finalize
    maybe_add_hybrid_all2all(kernel, moe_config)
    # PyNccl was checked on the group the manager communicates over.
    manager._get_comm_group.assert_called_once_with(sequence_parallel)

    large_impl = kernel._large_batch_impl
    assert large_impl is not None
    assert large_impl.fused_experts is kernel.fused_experts
    prepare_finalize = large_impl.prepare_finalize
    assert isinstance(prepare_finalize, MoEPrepareAndFinalizeNaiveDPEPModular)
    assert prepare_finalize._comm() is manager
    assert prepare_finalize.num_dispatchers() == moe_config.ep_size
    assert prepare_finalize.is_sequence_parallel == sequence_parallel
    assert kernel._large_batch_threshold == THRESHOLD
    assert kernel.prepare_finalize is primary


@pytest.mark.parametrize(
    "cudagraph_mode,max_capture_size,threshold",
    [
        # Raised so that every captured MoE call stays on the one-sided path.
        (CUDAGraphMode.FULL_AND_PIECEWISE, 512, 512),
        (CUDAGraphMode.PIECEWISE, 512, 512),
        (CUDAGraphMode.FULL_DECODE_ONLY, 512, 512),
        # Already covers every captured size.
        (CUDAGraphMode.FULL_AND_PIECEWISE, THRESHOLD, THRESHOLD),
        (CUDAGraphMode.FULL_AND_PIECEWISE, THRESHOLD // 2, THRESHOLD),
        (CUDAGraphMode.FULL_AND_PIECEWISE, None, THRESHOLD),
        # Nothing is captured.
        (CUDAGraphMode.NONE, 512, THRESHOLD),
    ],
)
def test_maybe_add_hybrid_all2all_threshold_covers_cudagraphs(
    hybrid_env, cudagraph_mode, max_capture_size, threshold
):
    hybrid_env.compilation_config.cudagraph_mode = cudagraph_mode
    hybrid_env.compilation_config.max_cudagraph_capture_size = max_capture_size
    kernel = _add_hybrid_all2all()
    assert kernel._large_batch_impl is not None
    assert kernel._large_batch_threshold == threshold


@pytest.mark.parametrize("tokens", [None, "0", "-1"])
def test_maybe_add_hybrid_all2all_needs_positive_threshold(
    monkeypatch, hybrid_env, tokens
):
    if tokens is None:
        monkeypatch.delenv("VLLM_MOE_HYBRID_A2A_TOKENS")
    else:
        monkeypatch.setenv("VLLM_MOE_HYBRID_A2A_TOKENS", tokens)
    assert _add_hybrid_all2all()._large_batch_impl is None


@pytest.mark.parametrize(
    "moe_kwargs",
    [
        pytest.param({"all2all_backend": "allgather_reducescatter"}, id="ag-rs"),
        pytest.param({"all2all_backend": "deepep_low_latency"}, id="deepep-ll"),
        pytest.param({"is_lora_enabled": True}, id="lora"),
        # One-sided through sequence-parallel MoE, but without DP.
        pytest.param(
            {"dp_size": 1, "tp_size": 2, "sequence_parallel": True}, id="no-dp"
        ),
    ],
)
def test_maybe_add_hybrid_all2all_skips_layer(hybrid_env, moe_kwargs):
    assert _add_hybrid_all2all(_moe_config(**moe_kwargs))._large_batch_impl is None


def test_maybe_add_hybrid_all2all_skips_tp_without_sequence_parallel(
    monkeypatch, hybrid_env
):
    # output_is_reduced() still comes from the one-sided prepare/finalize,
    # which differs from AgRs in the TP all-reduce that it gates.
    monkeypatch.setattr(
        all2all_utils, "get_tensor_model_parallel_world_size", lambda: 2
    )
    assert _add_hybrid_all2all(_moe_config(tp_size=2))._large_batch_impl is None


def test_maybe_add_hybrid_all2all_skips_monolithic_kernel(hybrid_env):
    kernel = _monolithic_kernel()
    maybe_add_hybrid_all2all(kernel, _moe_config())
    assert kernel._large_batch_impl is None


def test_maybe_add_hybrid_all2all_skips_experts_without_ag_rs(monkeypatch, hybrid_env):
    supports_parallel_config = MagicMock(return_value=False)
    monkeypatch.setattr(_Experts, "_supports_parallel_config", supports_parallel_config)
    moe_config = _moe_config()
    assert _add_hybrid_all2all(moe_config)._large_batch_impl is None
    # Asked about this layer's parallel config with the AgRs backend.
    supports_parallel_config.assert_called_once_with(
        replace(
            moe_config.moe_parallel_config,
            all2all_backend="allgather_reducescatter",
        )
    )


@pytest.mark.parametrize(
    "device_communicator",
    [
        pytest.param(None, id="no-communicator"),
        pytest.param(SimpleNamespace(), id="not-cuda"),
        pytest.param(SimpleNamespace(hybrid_all2all_manager=None), id="no-manager"),
    ],
)
def test_maybe_add_hybrid_all2all_needs_manager(hybrid_env, device_communicator):
    hybrid_env.ep_group.device_communicator = device_communicator
    assert _add_hybrid_all2all()._large_batch_impl is None


@pytest.mark.parametrize(
    "device_communicator",
    [
        pytest.param(None, id="no-communicator"),
        pytest.param(SimpleNamespace(pynccl_comm=None), id="no-pynccl"),
        pytest.param(
            SimpleNamespace(pynccl_comm=SimpleNamespace(disabled=True)),
            id="pynccl-disabled",
        ),
    ],
)
def test_maybe_add_hybrid_all2all_needs_pynccl(hybrid_env, device_communicator):
    hybrid_env.comm_group.device_communicator = device_communicator
    assert _add_hybrid_all2all()._large_batch_impl is None


def test_maybe_add_hybrid_all2all_needs_pynccl_enabled(monkeypatch, hybrid_env):
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", "1")
    assert _add_hybrid_all2all()._large_batch_impl is None


def test_make_nvfp4_moe_kernel_adds_hybrid_all2all(monkeypatch, hybrid_env):
    # In place of the one-sided prepare/finalize, as in _kernel().
    monkeypatch.setattr(
        nvfp4_oracle,
        "maybe_make_prepare_finalize",
        lambda **kwargs: _prepare_finalize(),
    )
    kernel = nvfp4_oracle.make_nvfp4_moe_kernel(
        moe_quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        moe_config=_moe_config(),
        experts_cls=_Experts,
        backend=nvfp4_oracle.NvFp4MoeBackend.FLASHINFER_CUTLASS,
    )
    assert kernel._large_batch_impl is not None
    assert kernel._large_batch_threshold == THRESHOLD


@pytest.mark.parametrize("use_manager", [False, True])
def test_naive_dp_ep_communicates_through_given_manager(monkeypatch, use_manager):
    ep_group = MagicMock(spec=GroupCoordinator)
    manager = MagicMock(spec=AgRsAll2AllManager)
    monkeypatch.setattr(naive_dp_ep, "get_ep_group", lambda: ep_group)
    prepare_finalize = MoEPrepareAndFinalizeNaiveDPEPModular(
        is_sequence_parallel=True,
        all2all_manager=manager if use_manager else None,
    )
    comm, unused = (manager, ep_group) if use_manager else (ep_group, manager)
    assert prepare_finalize._comm() is comm

    hidden_states = torch.randn(4, 8)
    topk_weights = torch.rand(4, 2)
    topk_ids = torch.zeros(4, 2, dtype=torch.int32)
    comm.dispatch.return_value = (hidden_states, topk_weights, topk_ids)
    prepare_finalize.prepare(
        hidden_states,
        topk_weights,
        topk_ids,
        num_experts=8,
        expert_map=None,
        apply_router_weight_on_input=False,
        quant_config=FUSED_MOE_UNQUANTIZED_CONFIG,
        defer_input_quant=True,
    )
    weight_and_reduce = TopKWeightAndReduceNoOP()
    prepare_finalize.allocate_fused_expert_output(
        (4, 8), hidden_states.dtype, hidden_states.device, weight_and_reduce
    )
    output = torch.empty_like(hidden_states)
    prepare_finalize.finalize(
        output, hidden_states, topk_weights, topk_ids, False, weight_and_reduce
    )

    comm.dispatch.assert_called_once_with(
        hidden_states,
        topk_weights,
        topk_ids,
        is_sequence_parallel=True,
        extra_tensors=None,
    )
    comm.allocate_combine_input.assert_called_once_with(
        (4, 8), hidden_states.dtype, hidden_states.device, is_sequence_parallel=True
    )
    comm.combine_into_output.assert_called_once_with(
        hidden_states, output, is_sequence_parallel=True
    )
    assert unused.mock_calls == []
