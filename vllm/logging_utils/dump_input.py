# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import faulthandler
import hashlib
import json
import os
import shutil
import signal
import sys
import tempfile
import threading
import time
import traceback as traceback_utils
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.diagnostics import (
    DIAGNOSTIC_STRING_MAX_CHARS,
    make_request_id_summary,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S = 300.0
ENGINE_DIAGNOSTIC_BUNDLE_VERSION = 3
ENGINE_DIAGNOSTIC_DUMP_DIR = "engine_diagnostics"
ENGINE_DIAGNOSTIC_BUNDLE_PREFIX = "bundle_"
ENGINE_DIAGNOSTIC_MAX_BUNDLES = 20
ENGINE_DIAGNOSTIC_INCOMPLETE_MAX_AGE_S = 3600.0
ENGINE_DIAGNOSTIC_CONTEXT_MAX_BYTES = 1024 * 1024
ENGINE_DIAGNOSTIC_STACKS_MAX_BYTES = 1024 * 1024
ENGINE_DIAGNOSTIC_TEXT_MAX_CHARS = 32_768
ENGINE_DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_CHARS = 4096
ENGINE_DIAGNOSTIC_WRITE_TIMEOUT_S = 1.0
ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT = 20
ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS = 32_768
ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S = 1.0
STACK_TRACE_SIGNAL_ENV_VAR = "VLLM_DEBUG_STACK_TRACE_SIGNAL"
ENGINE_NO_PROGRESS_STAGE = "no_forward_progress"
_SAFE_STACK_TRACE_SIGNAL_NAMES = ("SIGUSR1", "SIGUSR2")
_engine_diagnostic_bundle_lock = threading.Lock()
_stack_trace_signal_handler_lock = threading.Lock()
_stack_trace_signal_handlers: dict[int, str] = {}
_ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS = (
    "dtype",
    "enforce_eager",
    "max_model_len",
    "quantization",
    "runner_type",
)
_ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS = (
    "data_parallel_size",
    "enable_expert_parallel",
    "pipeline_parallel_size",
    "tensor_parallel_size",
)
_ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS = (
    "async_scheduling",
    "enable_chunked_prefill",
    "long_prefill_token_threshold",
    "max_num_batched_tokens",
    "max_num_scheduled_tokens",
    "max_num_seqs",
    "policy",
    "prefill_schedule_interval",
    "runner_type",
)
_ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS = (
    "block_size",
    "cache_dtype",
    "enable_prefix_caching",
    "gpu_memory_utilization",
)
_ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS = ("offload_backend",)
_ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS = ("cpu_offload_gb",)
_ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS = (
    "draft_tensor_parallel_size",
    "max_model_len",
    "method",
    "num_speculative_tokens",
    "quantization",
)


@dataclass(frozen=True)
class EngineExecutionTimeoutSnapshot:
    scheduler_output_summary: dict[str, Any]
    scheduler_queue_summary: dict[str, Any]


def prepare_object_to_dump(obj) -> str:
    if isinstance(obj, str):
        return f"'{obj}'"  # Double quotes
    elif isinstance(obj, dict):
        dict_str = ", ".join(
            {f"{str(k)}: {prepare_object_to_dump(v)}" for k, v in obj.items()}
        )
        return f"{{{dict_str}}}"
    elif isinstance(obj, list):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, set):
        return f"[{', '.join([prepare_object_to_dump(v) for v in list(obj)])}]"
        # return [prepare_object_to_dump(v) for v in list(obj)]
    elif isinstance(obj, tuple):
        return f"[{', '.join([prepare_object_to_dump(v) for v in obj])}]"
    elif isinstance(obj, enum.Enum):
        return repr(obj)
    elif isinstance(obj, torch.Tensor):
        # We only print the 'draft' of the tensor to not expose sensitive data
        # and to get some metadata in case of CUDA runtime crashed
        return f"Tensor(shape={obj.shape}, device={obj.device},dtype={obj.dtype})"
    elif hasattr(obj, "anon_repr"):
        return obj.anon_repr()
    elif hasattr(obj, "__dict__"):
        items = obj.__dict__.items()
        dict_str = ", ".join(
            [f"{str(k)}={prepare_object_to_dump(v)}" for k, v in items]
        )
        return f"{type(obj).__name__}({dict_str})"
    else:
        # Hacky way to make sure we can serialize the object in JSON format
        try:
            return json.dumps(obj)
        except (TypeError, OverflowError):
            return repr(obj)


def dump_engine_exception(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    error: Exception | None = None,
    scheduler_snapshot: dict[str, Any] | None = None,
    scheduler_snapshot_fn: Callable[[], dict[str, Any] | None] | None = None,
):
    # NOTE: ensure we can log extra info without risking raises
    # unexpected errors during logging
    with contextlib.suppress(Exception):
        _dump_engine_execution_context(
            "exception",
            config,
            scheduler_output,
            scheduler_stats,
            error=error,
            scheduler_snapshot=scheduler_snapshot,
            scheduler_snapshot_fn=scheduler_snapshot_fn,
        )


def dump_engine_execution_timeout(
    config: VllmConfig,
    snapshot: EngineExecutionTimeoutSnapshot,
    timeout_s: float,
    stage: str,
    scheduler_snapshot: dict[str, Any] | None = None,
):
    _emit_engine_execution_timeout(stage, timeout_s)
    _dump_engine_execution_timeout_details(
        config,
        snapshot,
        timeout_s,
        stage,
        scheduler_snapshot,
    )


def _dump_engine_execution_timeout_details(
    config: VllmConfig,
    snapshot: EngineExecutionTimeoutSnapshot,
    timeout_s: float,
    stage: str,
    scheduler_snapshot: dict[str, Any] | None,
) -> None:

    diagnostic_bundle_dir: Path | None = None
    try:
        diagnostic_bundle_dir = _dump_engine_timeout_context(
            config,
            snapshot,
            stage=stage,
            timeout_s=timeout_s,
            scheduler_snapshot=scheduler_snapshot,
        )
    except Exception:
        with contextlib.suppress(Exception):
            logger.exception("Failed to dump V1 engine timeout context")

    if diagnostic_bundle_dir is not None:
        _write_engine_traceback_dump(diagnostic_bundle_dir)
        expected_files = ["context.json", "stacks.txt"]
        if scheduler_snapshot is not None:
            expected_files.append("scheduler_snapshot.json")
        _finalize_engine_diagnostic_bundle(
            diagnostic_bundle_dir,
            expected_files=tuple(expected_files),
        )


def _emit_engine_execution_timeout(stage: str, timeout_s: float) -> None:
    with contextlib.suppress(Exception):
        logger.error(
            "V1 LLM engine stage '%s' has not completed after %.2f seconds "
            "(pid=%d). Dumping sanitized scheduler state and Python stack "
            "traces. "
            "Further dumps for this stage are throttled for %.0f seconds. "
            "Set VLLM_ENGINE_SLOW_STAGE_DUMP_S=0 to disable this diagnostic.",
            stage,
            timeout_s,
            os.getpid(),
            ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S,
        )

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def dump_engine_no_progress(
    config: VllmConfig,
    timeout_s: float,
    progress_snapshot: dict[str, Any],
    scheduler_snapshot: dict[str, Any] | None = None,
) -> None:
    combined_snapshot = _combine_no_progress_snapshots(
        progress_snapshot, scheduler_snapshot
    )
    with contextlib.suppress(Exception):
        logger.error(
            "V1 LLM engine has not made forward progress for %.2f seconds "
            "(pid=%d, stage=%s). Dumping engine progress, scheduler state, "
            "and Python stack traces. Further no-progress dumps are "
            "throttled for %.0f seconds. Set "
            "VLLM_ENGINE_NO_PROGRESS_TIMEOUT_S=0 to disable this diagnostic.",
            timeout_s,
            os.getpid(),
            progress_snapshot.get("stage"),
            ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S,
        )
        logger.error(
            "Engine no-progress diagnostic snapshot: %s",
            _serialize_diagnostic(combined_snapshot),
        )

    diagnostic_bundle_dir: Path | None = None
    dump_root = _engine_diagnostic_dump_root(config)
    if dump_root is not None:
        try:
            try:
                config_summary = _make_engine_config_summary(config)
            except Exception:
                logger.exception("Failed to prepare engine diagnostic config summary")
                config_summary = {"summary_unavailable": True}
            diagnostic_bundle_dir = _write_engine_diagnostic_bundle(
                reason="no_progress",
                config=config,
                dump_root=dump_root,
                config_summary=config_summary,
                scheduler_output_summary=None,
                scheduler_queue_summary=None,
                scheduler_output_text=None,
                scheduler_stats_text=None,
                stage=ENGINE_NO_PROGRESS_STAGE,
                timeout_s=timeout_s,
                error=None,
                scheduler_snapshot=combined_snapshot,
            )
        except Exception:
            logger.exception("Failed to write V1 engine no-progress context")

    if diagnostic_bundle_dir is not None:
        _write_engine_traceback_dump(diagnostic_bundle_dir)
        _finalize_engine_diagnostic_bundle(
            diagnostic_bundle_dir,
            expected_files=("context.json", "scheduler_snapshot.json", "stacks.txt"),
        )

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


class EngineCoreProgressMonitor:
    """Tracks EngineCore activity and dumps diagnostics on no progress."""

    def __init__(
        self,
        *,
        config: VllmConfig,
        process_name: str,
        timeout_s: float | None,
        has_work_fn: Callable[[], bool],
        scheduler_snapshot_fn: Callable[[], dict[str, Any] | None],
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.process_name = process_name
        self.timeout_s = timeout_s
        self.has_work_fn = has_work_fn
        self.scheduler_snapshot_fn = scheduler_snapshot_fn
        self.time_fn = time_fn

        now_s = self.time_fn()
        self._lock = threading.Lock()
        self._stage = "initializing"
        self._details: dict[str, Any] = {}
        self._activity_index = 0
        self._progress_index = 0
        self._last_activity_s = now_s
        self._last_progress_s = now_s
        self._operation_active = False
        self._last_dump_progress_index = -1
        self._last_dump_s = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return (
            self.timeout_s is not None
            and self.timeout_s > 0
            and not self._stop_event.is_set()
        )

    def start(self) -> None:
        timeout_s = self.timeout_s
        if not self.enabled or self._thread is not None:
            return

        try:
            thread = self._create_thread()
            self._thread = thread
            thread.start()
        except Exception:
            self._thread = None
            self._stop_event.set()
            logger.warning(
                "Failed to start EngineCore no-progress monitor for %s. "
                "Continuing without no-progress monitoring.",
                self.process_name,
                exc_info=True,
            )
            return
        logger.info(
            "Started EngineCore no-progress monitor for %s with timeout %.2fs.",
            self.process_name,
            timeout_s,
        )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None and threading.current_thread() is not self._thread:
            self._thread.join(timeout=1.0)

    def record_activity(
        self,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._record_activity_unlocked(stage, details)
            self._operation_active = True

    def record_progress(
        self,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._record_progress_unlocked(stage, details)

    def record_idle(self) -> None:
        with self._lock:
            self._record_progress_unlocked("idle", {"has_work": False})

    def snapshot(self) -> dict[str, Any]:
        now_s = self.time_fn()
        with self._lock:
            return self._snapshot_unlocked(now_s)

    def maybe_dump_no_progress(self) -> bool:
        timeout_s = self.timeout_s
        if timeout_s is None or timeout_s <= 0:
            return False

        with self._lock:
            operation_active = self._operation_active

        if operation_active:
            has_work = True
        else:
            try:
                has_work = self.has_work_fn()
            except Exception:
                logger.exception("Failed to query EngineCore work state")
                return False

        now_s = self.time_fn()
        with self._lock:
            if not has_work and not self._operation_active:
                self._record_progress_unlocked("idle", {"has_work": False})
                return False
            elapsed_s = now_s - self._last_progress_s
            if elapsed_s < timeout_s:
                return False
            if (
                self._last_dump_progress_index == self._progress_index
                and now_s - self._last_dump_s < ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
            ):
                return False
            progress_snapshot = self._snapshot_unlocked(now_s)
            self._last_dump_progress_index = self._progress_index
            self._last_dump_s = now_s

        dump_engine_no_progress(
            self.config,
            timeout_s,
            progress_snapshot,
            self._make_scheduler_snapshot(),
        )
        return True

    def _run(self) -> None:
        if not self.enabled:
            return
        while not self._stop_event.wait(self._check_interval_s()):
            with contextlib.suppress(Exception):
                self.maybe_dump_no_progress()

    def _check_interval_s(self) -> float:
        timeout_s = self.timeout_s
        if timeout_s is None:
            return 5.0
        return min(max(timeout_s / 4, 0.1), 5.0)

    def _create_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._run,
            name=f"{self.process_name}NoProgressMonitor",
            daemon=True,
        )

    def _record_activity_unlocked(
        self,
        stage: str,
        details: dict[str, Any] | None,
    ) -> None:
        self._activity_index += 1
        self._stage = stage
        self._details = details or {}
        self._last_activity_s = self.time_fn()

    def _record_progress_unlocked(
        self,
        stage: str,
        details: dict[str, Any] | None,
    ) -> None:
        self._record_activity_unlocked(stage, details)
        self._operation_active = False
        self._progress_index += 1
        self._last_progress_s = self._last_activity_s

    def _snapshot_unlocked(self, now_s: float) -> dict[str, Any]:
        return {
            "activity_index": self._activity_index,
            "elapsed_since_activity_s": max(0.0, now_s - self._last_activity_s),
            "elapsed_since_progress_s": max(0.0, now_s - self._last_progress_s),
            "last_activity_s": self._last_activity_s,
            "last_progress_s": self._last_progress_s,
            "operation_active": self._operation_active,
            "pid": os.getpid(),
            "process_name": self.process_name,
            "progress_index": self._progress_index,
            "stage": self._stage,
            "stage_details": self._details,
            "timeout_s": self.timeout_s,
        }

    def _make_scheduler_snapshot(self) -> dict[str, Any] | None:
        try:
            return self.scheduler_snapshot_fn()
        except Exception:
            logger.exception("Failed to collect V1 scheduler diagnostic snapshot")
            return None


def install_stack_trace_signal_handler(process_name: str) -> bool:
    signal_value = envs.VLLM_DEBUG_STACK_TRACE_SIGNAL
    if not signal_value:
        return False

    signum = _parse_stack_trace_signal(signal_value)
    if signum is None:
        logger.warning(
            "Ignoring invalid %s=%r. Use a signal name such as SIGUSR1 or "
            "a positive signal number.",
            STACK_TRACE_SIGNAL_ENV_VAR,
            signal_value,
        )
        return False

    signal_name = _format_signal_name(signum)
    if not _is_safe_stack_trace_signal(signum):
        logger.warning(
            "Ignoring %s=%r because %s is not a supported diagnostic signal. "
            "Use SIGUSR1 or SIGUSR2.",
            STACK_TRACE_SIGNAL_ENV_VAR,
            signal_value,
            signal_name,
        )
        return False

    with _stack_trace_signal_handler_lock:
        installed_process = _stack_trace_signal_handlers.get(signum)
        if installed_process is not None:
            logger.debug(
                "Stack trace signal handler for %s is already installed in "
                "%s (pid=%d).",
                signal_name,
                installed_process,
                os.getpid(),
            )
            return True

        try:
            existing_handler = signal.getsignal(signum)
            if existing_handler != signal.SIG_DFL:
                logger.warning(
                    "Ignoring %s=%r because %s already has a signal handler.",
                    STACK_TRACE_SIGNAL_ENV_VAR,
                    signal_value,
                    signal_name,
                )
                return False
            faulthandler.register(
                signum,
                file=sys.stderr,
                all_threads=True,
                chain=False,
            )
        except Exception as exc:
            logger.warning(
                "Failed to install stack trace signal handler for %s=%r in "
                "%s (pid=%d): %s",
                STACK_TRACE_SIGNAL_ENV_VAR,
                signal_value,
                process_name,
                os.getpid(),
                exc,
            )
            return False

        _stack_trace_signal_handlers[signum] = process_name

    logger.info(
        "Installed stack trace signal handler for %s in %s (pid=%d).",
        signal_name,
        process_name,
        os.getpid(),
    )
    return True


def _parse_stack_trace_signal(signal_value: str) -> int | None:
    normalized = signal_value.strip().upper()
    if not normalized:
        return None

    if normalized.isdecimal():
        signum = int(normalized)
        return signum if signum > 0 else None

    signal_name = normalized if normalized.startswith("SIG") else f"SIG{normalized}"
    signum = getattr(signal, signal_name, None)
    if isinstance(signum, int) and signum > 0:
        return int(signum)
    return None


def _format_signal_name(signum: int) -> str:
    try:
        return signal.Signals(signum).name
    except ValueError:
        return str(signum)


def _is_safe_stack_trace_signal(signum: int) -> bool:
    for signal_name in _SAFE_STACK_TRACE_SIGNAL_NAMES:
        safe_signum = getattr(signal, signal_name, None)
        if isinstance(safe_signum, int) and int(safe_signum) == signum:
            return True
    return False


def _combine_no_progress_snapshots(
    progress_snapshot: dict[str, Any],
    scheduler_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    combined_snapshot: dict[str, Any] = {"engine_progress": progress_snapshot}
    if scheduler_snapshot is not None:
        combined_snapshot["scheduler"] = scheduler_snapshot
    return combined_snapshot


def _dump_engine_timeout_context(
    config: VllmConfig,
    snapshot: EngineExecutionTimeoutSnapshot,
    *,
    stage: str | None = None,
    timeout_s: float | None = None,
    scheduler_snapshot: dict[str, Any] | None = None,
) -> Path | None:
    config_summary = _make_engine_config_summary(config)
    summary = {
        "config": config_summary,
        **snapshot.scheduler_output_summary,
    }
    scheduler_output_dump = _serialize_diagnostic(summary)
    logger.error("Scheduler output summary: %s", scheduler_output_dump)
    scheduler_queue_dump = None
    if snapshot.scheduler_queue_summary:
        scheduler_queue_dump = _serialize_diagnostic(snapshot.scheduler_queue_summary)
        logger.error(
            "Scheduler queue summary: %s",
            scheduler_queue_dump,
        )
    if scheduler_snapshot is not None:
        try:
            logger.error(
                "Scheduler diagnostic snapshot: %s",
                _serialize_diagnostic(scheduler_snapshot),
            )
        except Exception:
            logger.exception("Failed to log V1 scheduler diagnostic snapshot")
    return _write_engine_diagnostic_bundle(
        reason="timeout",
        config=config,
        config_summary=config_summary,
        scheduler_output_summary=snapshot.scheduler_output_summary,
        scheduler_queue_summary=snapshot.scheduler_queue_summary,
        scheduler_output_text=None,
        scheduler_stats_text=None,
        stage=stage,
        timeout_s=timeout_s,
        error=None,
        scheduler_snapshot=scheduler_snapshot,
    )


def make_engine_execution_timeout_snapshot(
    scheduler_output: SchedulerOutput,
    scheduler_state: dict[str, Any] | None,
) -> EngineExecutionTimeoutSnapshot:
    """Copy bounded timeout diagnostics off scheduler-owned objects."""
    scheduler_state = scheduler_state or {}
    cached_sampling_params = scheduler_state.get("cached_request_sampling_params", {})
    request_samples = _make_request_samples(
        scheduler_output,
        cached_sampling_params=cached_sampling_params,
    )
    num_scheduled_requests = (
        len(scheduler_output.scheduled_new_reqs)
        + scheduler_output.scheduled_cached_reqs.num_reqs
    )
    scheduler_output_summary: dict[str, Any] = {
        "has_kv_connector_metadata": (
            scheduler_output.kv_connector_metadata is not None
        ),
        "has_pending_structured_output_tokens": (
            scheduler_output.pending_structured_output_tokens
        ),
        "num_finished_reqs": len(scheduler_output.finished_req_ids),
        "num_preempted_reqs": len(scheduler_output.preempted_req_ids or ()),
        "num_scheduled_cached_reqs": scheduler_output.scheduled_cached_reqs.num_reqs,
        "num_scheduled_encoder_inputs": len(scheduler_output.scheduled_encoder_inputs),
        "num_scheduled_new_reqs": len(scheduler_output.scheduled_new_reqs),
        "num_scheduled_reqs": len(scheduler_output.num_scheduled_tokens),
        "num_scheduled_spec_decode_reqs": len(
            scheduler_output.scheduled_spec_decode_tokens
        ),
        "request_sample_limit": ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT,
        "request_samples": request_samples,
        "request_samples_truncated": num_scheduled_requests > len(request_samples),
        "total_num_scheduled_tokens": scheduler_output.total_num_scheduled_tokens,
    }
    queue_summary = {
        _bounded_diagnostic_string(str(key)): _diagnostic_scalar(value)
        for key, value in scheduler_state.items()
        if key != "cached_request_sampling_params"
    }
    return EngineExecutionTimeoutSnapshot(scheduler_output_summary, queue_summary)


def _make_engine_config_summary(config: VllmConfig) -> dict[str, Any]:
    model_config = config.model_config
    hf_config = getattr(model_config, "hf_config", None)
    speculative_config = config.speculative_config
    summary = {
        "model": {
            **_select_diagnostic_fields(
                model_config,
                _ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS,
            ),
            "architectures": list(getattr(hf_config, "architectures", None) or ()),
            "model_type": getattr(hf_config, "model_type", None),
        },
        "parallel": _select_diagnostic_fields(
            config.parallel_config,
            _ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS,
        ),
        "scheduler": _select_diagnostic_fields(
            config.scheduler_config,
            _ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS,
        ),
        "cache": _select_diagnostic_fields(
            config.cache_config,
            _ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS,
        ),
        "offload": {
            **_select_diagnostic_fields(
                config.offload_config,
                _ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS,
            ),
            **_select_diagnostic_fields(
                config.offload_config.uva,
                _ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS,
            ),
        },
        "speculative": {
            "enabled": speculative_config is not None,
            **_select_diagnostic_fields(
                speculative_config,
                _ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS,
            ),
        },
    }
    return summary


def _select_diagnostic_fields(
    obj: object | None,
    field_names: tuple[str, ...],
) -> dict[str, Any]:
    if obj is None:
        return {}
    return {
        field_name: _diagnostic_scalar(value)
        for field_name in field_names
        if (value := getattr(obj, field_name, None)) is not None
    }


def _diagnostic_scalar(value: Any) -> Any:
    if isinstance(value, enum.Enum):
        return value.name
    if isinstance(value, str):
        return _bounded_diagnostic_string(value)
    if value is None or isinstance(value, (bool, float, int)):
        return value
    return _bounded_diagnostic_string(str(value))


def _bounded_diagnostic_string(value: str) -> str:
    if len(value) <= DIAGNOSTIC_STRING_MAX_CHARS:
        return value
    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    return f"{value[:160]}...{value[-64:]} [length={len(value)}, sha256={digest}]"


def _serialize_diagnostic(value: dict[str, Any]) -> str:
    serialized = json.dumps(value, sort_keys=True, default=str)
    if len(serialized) <= ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS:
        return serialized

    metadata = {
        "diagnostic_output_truncated": True,
        "original_length": len(serialized),
        "sha256": hashlib.sha256(serialized.encode()).hexdigest(),
    }
    low = 0
    high = min(len(serialized), ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS)
    bounded = json.dumps({**metadata, "diagnostic_prefix": ""}, sort_keys=True)
    while low <= high:
        prefix_length = (low + high) // 2
        candidate = json.dumps(
            {**metadata, "diagnostic_prefix": serialized[:prefix_length]},
            sort_keys=True,
        )
        if len(candidate) <= ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS:
            bounded = candidate
            low = prefix_length + 1
        else:
            high = prefix_length - 1
    return bounded


def _make_request_samples(
    scheduler_output: SchedulerOutput,
    cached_sampling_params: dict[str, dict[str, Any] | None] | None = None,
) -> list[dict[str, Any]]:
    cached_requests = scheduler_output.scheduled_cached_reqs
    new_request_indices, cached_request_indices = (
        get_engine_timeout_request_sample_indices(
            len(scheduler_output.scheduled_new_reqs), cached_requests.num_reqs
        )
    )
    samples = [
        _make_new_request_sample(
            scheduler_output.scheduled_new_reqs[index], scheduler_output
        )
        for index in new_request_indices
    ]
    samples.extend(
        _make_cached_request_sample(
            index,
            scheduler_output,
            cached_sampling_params=cached_sampling_params,
        )
        for index in cached_request_indices
    )
    return samples


def get_engine_timeout_request_sample_indices(
    num_new_requests: int, num_cached_requests: int
) -> tuple[list[int], list[int]]:
    total_requests = num_new_requests + num_cached_requests
    sample_limit = min(ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT, total_requests)
    if num_new_requests == 0:
        return [], _evenly_spaced_indices(num_cached_requests, sample_limit)
    if num_cached_requests == 0:
        return _evenly_spaced_indices(num_new_requests, sample_limit), []

    new_request_limit = max(
        1,
        min(
            num_new_requests,
            sample_limit - 1,
            sample_limit * num_new_requests // total_requests,
        ),
    )
    cached_request_limit = sample_limit - new_request_limit
    return (
        _evenly_spaced_indices(num_new_requests, new_request_limit),
        _evenly_spaced_indices(num_cached_requests, cached_request_limit),
    )


def _evenly_spaced_indices(item_count: int, sample_count: int) -> list[int]:
    if sample_count >= item_count:
        return list(range(item_count))
    if sample_count == 1:
        return [item_count // 2]
    return [
        index * (item_count - 1) // (sample_count - 1) for index in range(sample_count)
    ]


def _make_new_request_sample(
    request: Any,
    scheduler_output: SchedulerOutput,
) -> dict[str, Any]:
    request_id = request.req_id
    prompt_embeds_shape = (
        tuple(request.prompt_embeds.shape)
        if request.prompt_embeds is not None
        else None
    )
    return {
        "has_lora": request.lora_request is not None,
        "has_pooling_params": request.pooling_params is not None,
        "num_computed_tokens": request.num_computed_tokens,
        "num_kv_blocks": _num_blocks(request.block_ids),
        "num_kv_cache_groups": len(request.block_ids),
        "num_mm_features": len(request.mm_features),
        "num_prefill_tokens": _optional_len(request.prefill_token_ids),
        "num_prompt_tokens": _optional_len(request.prompt_token_ids),
        "prompt_embeds_shape": prompt_embeds_shape,
        **make_request_id_summary(request_id),
        "request_kind": "new",
        "sampling_params": make_sampling_params_summary(request.sampling_params),
        **_make_scheduled_request_summary(request_id, scheduler_output),
    }


def _make_cached_request_sample(
    index: int,
    scheduler_output: SchedulerOutput,
    cached_sampling_params: dict[str, dict[str, Any] | None] | None = None,
) -> dict[str, Any]:
    cached_requests = scheduler_output.scheduled_cached_reqs
    request_id = cached_requests.req_ids[index]
    new_block_ids = _item_at(cached_requests.new_block_ids, index)
    new_token_ids = _item_at(cached_requests.new_token_ids, index)
    return {
        "is_resumed": request_id in cached_requests.resumed_req_ids,
        "num_all_tokens": _optional_len(cached_requests.all_token_ids.get(request_id)),
        "num_computed_tokens": _item_at(cached_requests.num_computed_tokens, index),
        "num_new_blocks": _num_blocks(new_block_ids),
        "num_new_tokens": _optional_len(new_token_ids),
        "num_output_tokens": _item_at(cached_requests.num_output_tokens, index),
        **make_request_id_summary(request_id),
        "request_kind": "cached",
        "sampling_params": (cached_sampling_params or {}).get(request_id),
        **_make_scheduled_request_summary(request_id, scheduler_output),
    }


def _make_scheduled_request_summary(
    request_id: str,
    scheduler_output: SchedulerOutput,
) -> dict[str, int]:
    return {
        "num_encoder_inputs": len(
            scheduler_output.scheduled_encoder_inputs.get(request_id, ())
        ),
        "num_scheduled_tokens": scheduler_output.num_scheduled_tokens.get(
            request_id, 0
        ),
        "num_spec_tokens": len(
            scheduler_output.scheduled_spec_decode_tokens.get(request_id, ())
        ),
    }


def make_sampling_params_summary(sampling_params: Any | None) -> dict[str, Any] | None:
    if sampling_params is None:
        return None
    summary = _select_diagnostic_fields(
        sampling_params,
        (
            "detokenize",
            "flat_logprobs",
            "frequency_penalty",
            "ignore_eos",
            "include_stop_str_in_output",
            "logprobs",
            "max_tokens",
            "min_p",
            "min_tokens",
            "n",
            "output_kind",
            "presence_penalty",
            "prompt_logprobs",
            "repetition_penalty",
            "seed",
            "skip_special_tokens",
            "spaces_between_special_tokens",
            "temperature",
            "thinking_token_budget",
            "top_k",
            "top_p",
        ),
    )
    stop = getattr(sampling_params, "stop", None)
    summary.update(
        {
            "has_extra_args": getattr(sampling_params, "extra_args", None) is not None,
            "has_repetition_detection": (
                getattr(sampling_params, "repetition_detection", None) is not None
            ),
            "has_structured_outputs": (
                getattr(sampling_params, "structured_outputs", None) is not None
            ),
            "num_allowed_token_ids": _optional_len(
                getattr(sampling_params, "allowed_token_ids", None)
            ),
            "num_bad_words": _optional_len(getattr(sampling_params, "bad_words", None)),
            "num_logit_bias_entries": _optional_len(
                getattr(sampling_params, "logit_bias", None)
            ),
            "num_logprob_token_ids": _optional_len(
                getattr(sampling_params, "logprob_token_ids", None)
            ),
            "num_stop_strings": 1 if isinstance(stop, str) else _optional_len(stop),
            "num_stop_token_ids": _optional_len(
                getattr(sampling_params, "stop_token_ids", None)
            ),
        }
    )
    return summary


def _item_at(values: list[Any], index: int) -> Any | None:
    return values[index] if index < len(values) else None


def _optional_len(value: Any | None) -> int | None:
    return len(value) if value is not None else None


def _num_blocks(block_ids: Any | None) -> int | None:
    if block_ids is None:
        return None
    return sum(len(group) for group in block_ids)


def _dump_engine_execution_context(
    reason: str,
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    stage: str | None = None,
    timeout_s: float | None = None,
    error: Exception | None = None,
    scheduler_snapshot: dict[str, Any] | None = None,
    scheduler_snapshot_fn: Callable[[], dict[str, Any] | None] | None = None,
) -> Path | None:
    logger.error(
        "Dumping input data for V1 LLM engine (v%s, reason=%s) with config: %s, ",
        VLLM_VERSION,
        reason,
        config,
    )
    scheduler_output_dump: str | None = None
    scheduler_stats_dump: str | None = None
    try:
        scheduler_output_dump = prepare_object_to_dump(scheduler_output)
        logger.error(
            "Dumping scheduler output for model execution: %s",
            scheduler_output_dump,
        )
        if scheduler_stats:
            scheduler_stats_dump = str(scheduler_stats)
            logger.error("Dumping scheduler stats: %s", scheduler_stats_dump)
        if scheduler_snapshot is not None:
            logger.error(
                "Dumping scheduler diagnostic snapshot: %s",
                _serialize_diagnostic(scheduler_snapshot),
            )
    except Exception:
        logger.exception("Error preparing object to dump")

    dump_root = _engine_diagnostic_dump_root(config)
    if dump_root is None:
        return None

    try:
        config_summary = _make_engine_config_summary(config)
    except Exception:
        logger.exception("Failed to prepare engine diagnostic config summary")
        config_summary = {"summary_unavailable": True}

    return _write_engine_diagnostic_bundle_with_timeout(
        reason=reason,
        config=config,
        dump_root=dump_root,
        config_summary=config_summary,
        scheduler_output_summary=None,
        scheduler_queue_summary=None,
        scheduler_output_text=scheduler_output_dump,
        scheduler_stats_text=scheduler_stats_dump,
        stage=stage,
        timeout_s=timeout_s,
        error=error,
        scheduler_snapshot=scheduler_snapshot,
        scheduler_snapshot_fn=scheduler_snapshot_fn,
    )


def _write_engine_diagnostic_bundle_with_timeout(
    *,
    reason: str,
    config: VllmConfig,
    dump_root: Path,
    config_summary: dict[str, Any],
    scheduler_output_summary: dict[str, Any] | None,
    scheduler_queue_summary: dict[str, Any] | None,
    scheduler_output_text: str | None,
    scheduler_stats_text: str | None,
    stage: str | None,
    timeout_s: float | None,
    error: Exception | None,
    scheduler_snapshot: dict[str, Any] | None = None,
    scheduler_snapshot_fn: Callable[[], dict[str, Any] | None] | None = None,
) -> Path | None:
    result: list[Path] = []

    def write_bundle() -> None:
        captured_scheduler_snapshot = scheduler_snapshot
        if captured_scheduler_snapshot is None and scheduler_snapshot_fn is not None:
            try:
                captured_scheduler_snapshot = scheduler_snapshot_fn()
            except Exception:
                logger.exception("Failed to collect V1 scheduler diagnostic snapshot")

        if (
            captured_scheduler_snapshot is not None
            and scheduler_snapshot_fn is not None
        ):
            try:
                logger.error(
                    "Dumping scheduler diagnostic snapshot: %s",
                    _serialize_diagnostic(captured_scheduler_snapshot),
                )
            except Exception:
                logger.exception("Failed to log V1 scheduler diagnostic snapshot")

        bundle_dir = _write_engine_diagnostic_bundle(
            reason=reason,
            config=config,
            dump_root=dump_root,
            config_summary=config_summary,
            scheduler_output_summary=scheduler_output_summary,
            scheduler_queue_summary=scheduler_queue_summary,
            scheduler_output_text=scheduler_output_text,
            scheduler_stats_text=scheduler_stats_text,
            stage=stage,
            timeout_s=timeout_s,
            error=error,
            scheduler_snapshot=captured_scheduler_snapshot,
        )
        expected_files = ["context.json"]
        if captured_scheduler_snapshot is not None:
            expected_files.append("scheduler_snapshot.json")
        if bundle_dir is not None and _finalize_engine_diagnostic_bundle(
            bundle_dir, expected_files=tuple(expected_files)
        ):
            result.append(bundle_dir)

    writer = threading.Thread(
        target=write_bundle,
        name="EngineDiagnosticBundleWriter",
        daemon=True,
    )
    try:
        writer.start()
    except RuntimeError:
        logger.exception("Failed to start V1 LLM engine diagnostic bundle writer")
        return None

    writer.join(ENGINE_DIAGNOSTIC_WRITE_TIMEOUT_S)
    if writer.is_alive():
        logger.warning(
            "V1 LLM engine diagnostic bundle write did not complete within %.1f "
            "seconds; continuing exception propagation",
            ENGINE_DIAGNOSTIC_WRITE_TIMEOUT_S,
        )
        return None
    return result[0] if result else None


def _write_engine_diagnostic_bundle(
    *,
    reason: str,
    config: VllmConfig,
    config_summary: dict[str, Any],
    scheduler_output_summary: dict[str, Any] | None,
    scheduler_queue_summary: dict[str, Any] | None,
    scheduler_output_text: str | None,
    scheduler_stats_text: str | None,
    stage: str | None,
    timeout_s: float | None,
    error: Exception | None,
    scheduler_snapshot: dict[str, Any] | None = None,
    dump_root: Path | None = None,
) -> Path | None:
    bundle_dir: Path | None = None
    try:
        dump_root = dump_root or _engine_diagnostic_dump_root(config)
        if dump_root is None:
            return None

        bundle_dir, created_at = _create_engine_diagnostic_bundle_dir(
            dump_root, reason, stage
        )
        scheduler_snapshot_file: str | None = None
        if scheduler_snapshot is not None:
            scheduler_snapshot_path = bundle_dir / "scheduler_snapshot.json"
            try:
                _write_engine_diagnostic_json(
                    scheduler_snapshot_path,
                    scheduler_snapshot,
                    max_bytes=ENGINE_DIAGNOSTIC_CONTEXT_MAX_BYTES,
                )
                scheduler_snapshot_file = "scheduler_snapshot.json"
            except Exception:
                with contextlib.suppress(OSError):
                    scheduler_snapshot_path.unlink()
                logger.exception("Failed to write V1 scheduler diagnostic snapshot")
        context = {
            "bundle_version": ENGINE_DIAGNOSTIC_BUNDLE_VERSION,
            "config_summary": config_summary,
            "created_at": created_at.isoformat(),
            "exception": _format_engine_diagnostic_exception(error),
            "pid": os.getpid(),
            "reason": reason,
            "scheduler_snapshot_file": scheduler_snapshot_file,
            "scheduler_output_summary": scheduler_output_summary,
            "scheduler_output_text": _bounded_engine_diagnostic_text(
                scheduler_output_text
            ),
            "scheduler_queue_summary": scheduler_queue_summary,
            "scheduler_stats_text": _bounded_engine_diagnostic_text(
                scheduler_stats_text
            ),
            "stage": stage,
            "timeout_s": timeout_s,
            "vllm_version": VLLM_VERSION,
        }
        _write_engine_diagnostic_json(
            bundle_dir / "context.json",
            context,
            max_bytes=ENGINE_DIAGNOSTIC_CONTEXT_MAX_BYTES,
        )
        return bundle_dir
    except Exception:
        logger.exception("Failed to write V1 LLM engine diagnostic bundle")
        if bundle_dir is not None:
            _remove_engine_diagnostic_bundle(bundle_dir)
        return None


def _engine_diagnostic_dump_root(config: VllmConfig) -> Path | None:
    configured_path = envs.VLLM_ENGINE_DIAGNOSTIC_DUMP_PATH
    if not configured_path:
        return None

    parallel_config = config.parallel_config
    rank = getattr(parallel_config, "rank", 0)
    dp_index = getattr(parallel_config, "data_parallel_index", 0)
    rank_dir = f"rank_{rank}_dp_{dp_index}"
    return (
        Path(configured_path).expanduser().absolute()
        / rank_dir
        / ENGINE_DIAGNOSTIC_DUMP_DIR
    )


def _create_engine_diagnostic_bundle_dir(
    dump_root: Path,
    reason: str,
    stage: str | None,
) -> tuple[Path, datetime]:
    created_at = datetime.now(timezone.utc)
    timestamp = created_at.strftime("%Y%m%dT%H%M%S.%fZ")
    reason_part = _safe_diagnostic_filename_component(reason)
    stage_part = _safe_diagnostic_filename_component(stage)
    base_name = (
        f"{ENGINE_DIAGNOSTIC_BUNDLE_PREFIX}{timestamp}_pid{os.getpid()}_"
        f"{reason_part}_{stage_part}"
    )

    with _engine_diagnostic_bundle_lock:
        dump_root.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(dump_root.parent, 0o700)
        dump_root.mkdir(mode=0o700, exist_ok=True)
        os.chmod(dump_root, 0o700)
        _prune_engine_diagnostic_bundles_locked(dump_root)
        for suffix in range(1000):
            bundle_dir = dump_root / (
                base_name if suffix == 0 else f"{base_name}_{suffix}"
            )
            try:
                bundle_dir.mkdir(mode=0o700, exist_ok=False)
                os.chmod(bundle_dir, 0o700)
                return bundle_dir, created_at
            except FileExistsError:
                continue

    raise FileExistsError(f"Could not create unique diagnostic bundle in {dump_root}")


def _safe_diagnostic_filename_component(value: str | None) -> str:
    if value is None:
        return "unknown"
    safe_value = "".join(
        ch if ch.isascii() and (ch.isalnum() or ch in ("-", "_")) else "_"
        for ch in value
    )
    return safe_value[:64] or "unknown"


def _format_engine_diagnostic_exception(
    error: Exception | None,
) -> dict[str, str] | None:
    if error is None:
        return None
    message = _bounded_engine_diagnostic_text(
        str(error), max_chars=ENGINE_DIAGNOSTIC_EXCEPTION_MESSAGE_MAX_CHARS
    )
    traceback = _bounded_engine_diagnostic_text(
        "".join(
            traceback_utils.format_exception(type(error), error, error.__traceback__)
        )
    )
    return {
        "message": message or "",
        "traceback": traceback or "",
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
    }


def _bounded_engine_diagnostic_text(
    value: str | None,
    *,
    max_chars: int = ENGINE_DIAGNOSTIC_TEXT_MAX_CHARS,
) -> str | None:
    if value is None or len(value) <= max_chars:
        return value

    digest = hashlib.sha256(value.encode("utf-8", errors="surrogatepass")).hexdigest()
    marker = f"\n...[truncated length={len(value)}, sha256={digest}]"
    return value[: max(0, max_chars - len(marker))] + marker


def _write_private_text_atomic(
    path: Path,
    value: str,
    *,
    max_bytes: int,
) -> None:
    encoded = value.encode("utf-8", errors="surrogatepass")
    if len(encoded) > max_bytes:
        raise ValueError(f"Diagnostic output exceeds {max_bytes} bytes")

    fd: int | None
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        os.chmod(temporary_path, 0o600)
        with os.fdopen(fd, "wb") as output:
            fd = None
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        os.chmod(path, 0o600)
    except Exception:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        with contextlib.suppress(OSError):
            temporary_path.unlink()
        raise


def _write_engine_diagnostic_json(
    path: Path,
    value: dict[str, Any],
    *,
    max_bytes: int,
) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, default=str)
    _write_private_text_atomic(path, serialized, max_bytes=max_bytes)


def _write_engine_traceback_dump(bundle_dir: Path) -> bool:
    stack_path = bundle_dir / "stacks.txt"
    fd: int | None = None
    temporary_path: Path | None = None
    try:
        fd, temporary_name = tempfile.mkstemp(
            prefix=".stacks.txt.", suffix=".tmp", dir=bundle_dir
        )
        temporary_path = Path(temporary_name)
        os.chmod(temporary_path, 0o600)
        with os.fdopen(fd, "w+b") as dump_file:
            fd = None
            faulthandler.dump_traceback(file=dump_file, all_threads=True)
            dump_file.flush()
            size = os.fstat(dump_file.fileno()).st_size
            if size > ENGINE_DIAGNOSTIC_STACKS_MAX_BYTES:
                marker = b"\n...[stack dump truncated]\n"
                dump_file.seek(ENGINE_DIAGNOSTIC_STACKS_MAX_BYTES - len(marker))
                dump_file.write(marker)
                dump_file.truncate(ENGINE_DIAGNOSTIC_STACKS_MAX_BYTES)
                dump_file.flush()
            os.fsync(dump_file.fileno())
        os.replace(temporary_path, stack_path)
        os.chmod(stack_path, 0o600)
        return True
    except Exception:
        if fd is not None:
            with contextlib.suppress(OSError):
                os.close(fd)
        if temporary_path is not None:
            with contextlib.suppress(OSError):
                temporary_path.unlink()
        logger.exception("Failed to write V1 LLM engine stack trace dump")
        return False


def _finalize_engine_diagnostic_bundle(
    bundle_dir: Path,
    *,
    expected_files: tuple[str, ...] = ("context.json",),
) -> bool:
    try:
        files = sorted(
            path.name
            for path in bundle_dir.iterdir()
            if path.is_file() and path.name != "manifest.json"
        )
        artifacts = {
            name: "written" if name in files else "failed" for name in expected_files
        }
        complete = all(status == "written" for status in artifacts.values())
        _write_engine_diagnostic_json(
            bundle_dir / "manifest.json",
            {
                "artifacts": artifacts,
                "bundle_version": ENGINE_DIAGNOSTIC_BUNDLE_VERSION,
                "complete": complete,
                "files": files,
            },
            max_bytes=ENGINE_DIAGNOSTIC_CONTEXT_MAX_BYTES,
        )
    except Exception:
        logger.exception("Failed to finalize V1 LLM engine diagnostic bundle")
        _remove_engine_diagnostic_bundle(bundle_dir)
        return False

    if complete:
        logger.error("Wrote V1 LLM engine diagnostic bundle to %s", bundle_dir)
    else:
        logger.warning(
            "Wrote incomplete V1 LLM engine diagnostic bundle to %s: %s",
            bundle_dir,
            artifacts,
        )
    try:
        _prune_engine_diagnostic_bundles(bundle_dir.parent)
    except OSError:
        logger.warning(
            "Failed to prune old V1 LLM engine diagnostic bundles in %s",
            bundle_dir.parent,
            exc_info=True,
        )
    return True


def _remove_engine_diagnostic_bundle(bundle_dir: Path) -> None:
    try:
        shutil.rmtree(bundle_dir)
    except OSError:
        logger.warning(
            "Failed to remove incomplete V1 LLM engine diagnostic bundle %s",
            bundle_dir,
            exc_info=True,
        )


def _prune_engine_diagnostic_bundles(dump_root: Path) -> None:
    with _engine_diagnostic_bundle_lock:
        _prune_engine_diagnostic_bundles_locked(dump_root)


def _prune_engine_diagnostic_bundles_locked(dump_root: Path) -> None:
    now = time.time()
    bundle_dirs = sorted(
        (
            path
            for path in dump_root.iterdir()
            if path.is_dir()
            and not path.is_symlink()
            and path.name.startswith(ENGINE_DIAGNOSTIC_BUNDLE_PREFIX)
        ),
        key=lambda path: path.name,
        reverse=True,
    )
    finalized_dirs = [
        path for path in bundle_dirs if (path / "manifest.json").is_file()
    ]
    finalized_dir_set = set(finalized_dirs)
    stale_incomplete_dirs = []
    for path in bundle_dirs:
        if path in finalized_dir_set:
            continue
        try:
            if now - path.stat().st_mtime > ENGINE_DIAGNOSTIC_INCOMPLETE_MAX_AGE_S:
                stale_incomplete_dirs.append(path)
        except OSError:
            continue

    for path in finalized_dirs[ENGINE_DIAGNOSTIC_MAX_BUNDLES:] + stale_incomplete_dirs:
        try:
            shutil.rmtree(path)
        except OSError:
            logger.warning(
                "Failed to remove old V1 LLM engine diagnostic bundle %s",
                path,
                exc_info=True,
            )


@dataclass(frozen=True)
class _EngineExecutionTimeoutState:
    deadline_s: float
    generation: int
    snapshot: EngineExecutionTimeoutSnapshot
    stage: str


class EngineExecutionTimeoutWatchdog:
    """Dumps engine state when an armed execution stage exceeds its deadline."""

    def __init__(
        self,
        *,
        config: VllmConfig,
        timeout_s: float | None,
        scheduler_snapshot_fn: Callable[[], dict[str, Any] | None] | None = None,
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.timeout_s = timeout_s
        self.scheduler_snapshot_fn = scheduler_snapshot_fn
        self.time_fn = time_fn

        self._generation = 0
        self._last_dump_s_by_stage: dict[str, float] = {}
        self._lock = threading.Lock()
        self._state: _EngineExecutionTimeoutState | None = None
        self._stopped = False
        self._diagnostic_thread: threading.Thread | None = None
        self._thread: threading.Thread | None = None
        self._wake_event = threading.Event()

    @property
    def enabled(self) -> bool:
        return self.timeout_s is not None and self.timeout_s > 0 and not self._stopped

    def start(self) -> None:
        if not self.enabled:
            return

        start_error = None
        with self._lock:
            if self._stopped or self._thread is not None:
                return
            self._thread = self._create_thread()
            try:
                self._thread.start()
            except RuntimeError as err:
                self._thread = None
                self._stopped = True
                start_error = err

        if start_error is not None:
            logger.warning(
                "Unable to start the engine execution timeout watchdog; "
                "continuing with this diagnostic disabled: %s",
                start_error,
            )

    def _create_thread(self) -> threading.Thread:
        return threading.Thread(
            target=self._run,
            name="EngineExecutionTimeoutWatchdog",
            daemon=True,
        )

    def stop(self) -> None:
        with self._lock:
            if self._stopped:
                return
            self._stopped = True
            self._state = None
            thread = self._thread
        self._wake_event.set()
        if thread is not None and threading.current_thread() is not thread:
            thread.join(timeout=ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S)
            if thread.is_alive():
                logger.warning(
                    "Engine execution timeout watchdog did not stop within "
                    "%.1f seconds",
                    ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S,
                )

    def arm(
        self,
        snapshot: EngineExecutionTimeoutSnapshot,
        stage: str,
    ) -> int | None:
        timeout_s = self.timeout_s
        if timeout_s is None or timeout_s <= 0 or self._stopped:
            return None

        with self._lock:
            if self._stopped:
                return None
            self._generation += 1
            generation = self._generation
            previous_state = self._state
            deadline_s = self.time_fn() + timeout_s
            self._state = _EngineExecutionTimeoutState(
                deadline_s=deadline_s,
                generation=generation,
                snapshot=snapshot,
                stage=stage,
            )
            should_wake = (
                previous_state is None or deadline_s < previous_state.deadline_s
            )
        if should_wake:
            self._wake_event.set()
        return generation

    def disarm(self, generation: int | None) -> None:
        if generation is None:
            return
        with self._lock:
            if self._stopped:
                return
            state = self._state
            if state is None or state.generation != generation:
                return
            self._state = None

    def _run(self) -> None:
        while True:
            self._wake_event.clear()
            with self._lock:
                if self._stopped:
                    return
                state = self._state

            if state is None:
                self._wake_event.wait()
                continue

            remaining_s = max(0.0, state.deadline_s - self.time_fn())
            if self._wake_event.wait(remaining_s):
                continue

            with self._lock:
                current_state = self._state
                if (
                    current_state is None
                    or current_state.generation != state.generation
                    or current_state.deadline_s > self.time_fn()
                ):
                    continue
                self._state = None

            if not self._mark_dump_if_allowed(state.stage):
                continue
            self._dispatch_timeout_dump(state)

    def _dispatch_timeout_dump(self, state: _EngineExecutionTimeoutState) -> bool:
        timeout_s = self.timeout_s or 0
        creation_error: Exception | None = None
        with self._lock:
            diagnostic_thread = self._diagnostic_thread
            if diagnostic_thread is not None and diagnostic_thread.is_alive():
                write_in_flight = True
                next_thread = None
            else:
                write_in_flight = False
                try:
                    next_thread = threading.Thread(
                        target=self._dump_timeout,
                        args=(state, timeout_s),
                        name="EngineExecutionTimeoutDiagnostic",
                        daemon=True,
                    )
                    self._diagnostic_thread = next_thread
                except Exception as err:
                    creation_error = err
                    next_thread = None

        if write_in_flight:
            _emit_engine_execution_timeout(state.stage, timeout_s)
            logger.warning(
                "Skipping file-backed timeout diagnostics for stage '%s' because "
                "a previous diagnostic write is still in progress",
                state.stage,
            )
            return False

        if next_thread is None:
            _emit_engine_execution_timeout(state.stage, timeout_s)
            logger.warning(
                "Unable to create the engine timeout diagnostic writer; "
                "continuing without file-backed output: %s",
                creation_error,
            )
            return False
        try:
            next_thread.start()
        except RuntimeError as err:
            with self._lock:
                if self._diagnostic_thread is next_thread:
                    self._diagnostic_thread = None
            _emit_engine_execution_timeout(state.stage, timeout_s)
            logger.warning(
                "Unable to start the engine timeout diagnostic writer; "
                "continuing without file-backed output: %s",
                err,
            )
            return False
        return True

    def _dump_timeout(
        self,
        state: _EngineExecutionTimeoutState,
        timeout_s: float,
    ) -> None:
        if self.scheduler_snapshot_fn is None:
            dump_engine_execution_timeout(
                self.config,
                state.snapshot,
                timeout_s,
                state.stage,
            )
            return

        _emit_engine_execution_timeout(state.stage, timeout_s)
        scheduler_snapshot = self._make_scheduler_snapshot()
        _dump_engine_execution_timeout_details(
            self.config,
            state.snapshot,
            timeout_s,
            state.stage,
            scheduler_snapshot,
        )

    def _make_scheduler_snapshot(self) -> dict[str, Any] | None:
        if self.scheduler_snapshot_fn is None:
            return None
        try:
            return self.scheduler_snapshot_fn()
        except Exception:
            logger.exception("Failed to collect V1 scheduler diagnostic snapshot")
            return None

    def _mark_dump_if_allowed(self, stage: str) -> bool:
        now_s = self.time_fn()
        with self._lock:
            last_dump_s = self._last_dump_s_by_stage.get(stage)
            if (
                last_dump_s is not None
                and now_s - last_dump_s < ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
            ):
                return False
            self._last_dump_s_by_stage[stage] = now_s
            return True
