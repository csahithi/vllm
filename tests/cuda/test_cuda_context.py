# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from vllm.platforms import current_platform


def check_cuda_context():
    """Check CUDA driver context status."""
    try:
        cuda = ctypes.CDLL("libcuda.so")
        device = ctypes.c_int()
        result = cuda.cuCtxGetDevice(ctypes.byref(device))
        return (True, device.value) if result == 0 else (False, None)
    except Exception:
        return False, None


def run_cuda_test_in_thread(device_input, expected_device_id):
    """Run CUDA context test in separate thread for isolation."""
    try:
        # New thread should have no CUDA context initially
        valid_before, device_before = check_cuda_context()
        if valid_before:
            return (
                False,
                (
                    "CUDA context should not exist in new thread, "
                    f"got device {device_before}"
                ),
            )

        # Test setting CUDA context
        current_platform.set_device(device_input)

        # Verify context is created correctly
        valid_after, device_id = check_cuda_context()
        if not valid_after:
            return False, "CUDA context should be valid after set_cuda_context"
        if device_id != expected_device_id:
            return False, f"Expected device {expected_device_id}, got {device_id}"

        return True, "Success"
    except Exception as e:
        return False, f"Exception in thread: {str(e)}"


class TestSetCudaContext:
    """Test suite for the set_cuda_context function."""

    @pytest.mark.skipif(not current_platform.is_cuda(), reason="CUDA not available")
    @pytest.mark.parametrize(
        argnames="device_input,expected_device_id",
        argvalues=[
            (0, 0),
            (torch.device("cuda:0"), 0),
            ("cuda:0", 0),
        ],
        ids=["int", "torch_device", "string"],
    )
    def test_set_cuda_context_parametrized(self, device_input, expected_device_id):
        """Test setting CUDA context in isolated threads."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(
                run_cuda_test_in_thread, device_input, expected_device_id
            )
            success, message = future.result(timeout=30)
        assert success, message

    @pytest.mark.skipif(
        not current_platform.is_cuda_alike(), reason="CUDA/ROCm not available"
    )
    def test_set_cuda_context_invalid_device_type(self):
        """Test error handling for invalid device type."""
        with pytest.raises(ValueError, match="Expected a cuda device"):
            current_platform.set_device(torch.device("cpu"))


def test_get_device_capability_uses_visible_device_ordinal(monkeypatch):
    import vllm.platforms.interface as platform_interface
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    seen_indices: list[int] = []

    def record_handle(index: int) -> str:
        seen_indices.append(index)
        return f"handle-{index}"

    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", [1])
    monkeypatch.setenv(NvmlCudaPlatform.device_control_env_var, "0,1")
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "device_control_id_to_physical_device_id",
        classmethod(lambda _cls, device_id: int(device_id)),
    )
    monkeypatch.setattr(pynvml, "nvmlInit", lambda: None)
    monkeypatch.setattr(pynvml, "nvmlShutdown", lambda: None)
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetHandleByIndex",
        record_handle,
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlDeviceGetCudaComputeCapability",
        lambda _handle: (9, 0),
    )
    NvmlCudaPlatform.get_device_capability.cache_clear()

    capability = NvmlCudaPlatform.get_device_capability(device_id=1)

    assert capability is not None
    assert capability.to_int() == 90
    assert seen_indices == [1]


def _stub_nvml(monkeypatch) -> dict[str, int]:
    """Stub NVML to report SM 9.0 and count init/shutdown pairs.

    Pins `NvmlCudaPlatform` rather than the `CudaPlatform` alias: that alias is
    `NonNvmlCudaPlatform` where NVML is unavailable, and that class reads
    `torch.cuda` instead, which needs a real device.
    """
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    calls = {"init": 0, "shutdown": 0}

    monkeypatch.setattr(
        pynvml, "nvmlInit", lambda: calls.__setitem__("init", calls["init"] + 1)
    )
    monkeypatch.setattr(
        pynvml,
        "nvmlShutdown",
        lambda: calls.__setitem__("shutdown", calls["shutdown"] + 1),
    )
    # Pin the visible-device mapping so the test does not depend on whatever
    # CUDA_VISIBLE_DEVICES happens to be set to in the environment.
    monkeypatch.setenv(NvmlCudaPlatform.device_control_env_var, "0")
    monkeypatch.setattr(
        NvmlCudaPlatform,
        "device_control_id_to_physical_device_id",
        classmethod(lambda _cls, device_id: int(device_id)),
    )
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetHandleByIndex", lambda index: f"handle-{index}"
    )
    monkeypatch.setattr(
        pynvml, "nvmlDeviceGetCudaComputeCapability", lambda _handle: (9, 0)
    )
    NvmlCudaPlatform.get_device_capability.cache_clear()
    return calls


def test_has_device_capability_does_not_reinit_nvml(monkeypatch):
    """Repeated capability checks must not re-enter an NVML context.

    `has_device_capability` only reads the cached `get_device_capability`, which
    carries its own NVML context. Wrapping it in `with_nvml_context` as well
    cost an nvmlInit()/nvmlShutdown() pair per call, and
    `triton_reshape_and_cache_flash` calls it per attention layer per step for
    fp8 and bfloat16 KV caches (issue #50381).
    """
    from vllm.platforms.cuda import NvmlCudaPlatform

    calls = _stub_nvml(monkeypatch)
    try:
        assert NvmlCudaPlatform.has_device_capability(80)
        for _ in range(20):
            NvmlCudaPlatform.has_device_capability(80)
            NvmlCudaPlatform.has_device_capability(89)
            NvmlCudaPlatform.has_device_capability((9, 0))

        assert calls["init"] == 1
        assert calls["shutdown"] == 1
    finally:
        NvmlCudaPlatform.get_device_capability.cache_clear()


def test_has_device_capability_comparisons(monkeypatch):
    """Dropping the redundant NVML context must not change the answers."""
    from vllm.platforms.cuda import NvmlCudaPlatform

    _stub_nvml(monkeypatch)
    try:
        assert NvmlCudaPlatform.has_device_capability(80)
        assert NvmlCudaPlatform.has_device_capability(90)
        assert NvmlCudaPlatform.has_device_capability((9, 0))
        assert not NvmlCudaPlatform.has_device_capability(100)
        assert not NvmlCudaPlatform.has_device_capability((10, 0))
    finally:
        NvmlCudaPlatform.get_device_capability.cache_clear()


# DGX Station layout: NVML (PCI order) lists the display GPU first, while CUDA's
# default fastest-first order makes the GB300 device 0.
_STATION_NVML = [("RTX PRO 4000", (12, 0), "rtx"), ("GB300", (10, 3), "gb300")]


@pytest.fixture
def station_nvml(monkeypatch):
    import vllm.platforms.cuda as cuda_platform
    from vllm.platforms.cuda import NvmlCudaPlatform, pynvml

    probes = {"count": 0}
    gpus = list(_STATION_NVML)

    def cuda_uuids() -> list[str]:
        probes["count"] += 1
        return ["gb300", "rtx"]

    for name, fn in {
        "nvmlInit": lambda: None,
        "nvmlShutdown": lambda: None,
        "nvmlDeviceGetCount": lambda: len(gpus),
        "nvmlDeviceGetHandleByIndex": lambda i: i,
        "nvmlDeviceGetIndex": lambda handle: handle,
        "nvmlDeviceGetHandleByUUID": lambda uuid: [g[2] for g in gpus].index(
            uuid.removeprefix("GPU-")
        ),
        "nvmlDeviceGetName": lambda handle: gpus[handle][0],
        "nvmlDeviceGetCudaComputeCapability": lambda handle: gpus[handle][1],
    }.items():
        monkeypatch.setattr(pynvml, name, fn)
    monkeypatch.setattr(cuda_platform, "cuda_is_initialized", lambda: False)
    monkeypatch.setattr(
        cuda_platform, "_cuda_device_uuids_in_enumeration_order", cuda_uuids
    )
    monkeypatch.delenv("CUDA_DEVICE_ORDER", raising=False)
    monkeypatch.delenv(NvmlCudaPlatform.device_control_env_var, raising=False)

    def clear():
        cuda_platform._cuda_index_to_nvml_index_map.cache_clear()
        NvmlCudaPlatform.get_device_capability.cache_clear()

    clear()
    yield gpus, probes
    clear()


@pytest.mark.parametrize(
    ("cuda_visible_devices", "expected_name", "expected_capability"),
    [
        (None, "GB300", (10, 3)),
        ("0", "GB300", (10, 3)),
        ("1", "RTX PRO 4000", (12, 0)),
    ],
)
def test_device_queries_follow_cuda_order_on_mixed_gpus(
    station_nvml,
    monkeypatch,
    cuda_visible_devices,
    expected_name,
    expected_capability,
):
    """Device queries must describe the GPU torch runs on, before CUDA init.

    Integer CUDA_VISIBLE_DEVICES entries and unset-env ordinals index CUDA's
    enumeration, not NVML's PCI order.
    """
    from vllm.platforms.cuda import NvmlCudaPlatform

    if cuda_visible_devices is not None:
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", cuda_visible_devices)

    assert NvmlCudaPlatform.get_device_name(0) == expected_name
    assert NvmlCudaPlatform.get_device_capability(0) == expected_capability


def test_physical_device_id_maps_to_cuda_ordinal_on_mixed_gpus(
    station_nvml, monkeypatch
):
    """`--device-ids 1` (nvidia-smi index of the GB300) must select cuda:0."""
    import vllm.platforms.interface as platform_interface
    from vllm.platforms.cuda import NvmlCudaPlatform

    monkeypatch.setattr(platform_interface, "_assigned_physical_gpu_ids", [1])

    assert NvmlCudaPlatform.logical_device_id_to_visible_device_id(0) == 0


@pytest.mark.parametrize("case", ["pci_bus_id_order", "same_model"])
def test_device_order_probe_skipped_when_orders_agree(station_nvml, monkeypatch, case):
    """CUDA and NVML order agree here, so no probe runs and ids map 1:1."""
    from vllm.platforms.cuda import NvmlCudaPlatform

    gpus, probes = station_nvml
    if case == "pci_bus_id_order":
        monkeypatch.setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    else:
        gpus[:] = [("B200", (10, 0), "b200-0"), ("B200", (10, 0), "b200-1")]

    assert NvmlCudaPlatform.visible_device_id_to_physical_device_id(0) == 0
    assert NvmlCudaPlatform.logical_device_id_to_visible_device_id(1) == 1
    assert probes["count"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
