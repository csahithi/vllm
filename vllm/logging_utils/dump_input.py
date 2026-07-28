# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import enum
import faulthandler
import json
import os
import signal
import sys
import threading
import time
import traceback as traceback_utils
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any

import torch
from typing_extensions import Self

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S = 300.0
ENGINE_DIAGNOSTIC_BUNDLE_VERSION = 1
ENGINE_DIAGNOSTIC_DUMP_DIR = "engine_diagnostics"
STACK_TRACE_SIGNAL_ENV_VAR = "VLLM_DEBUG_STACK_TRACE_SIGNAL"
ENGINE_NO_PROGRESS_STAGE = "no_forward_progress"
_PROTECTED_STACK_TRACE_SIGNAL_NAMES = ("SIGINT", "SIGTERM", "SIGKILL", "SIGSTOP")
_engine_execution_timeout_dump_lock = threading.Lock()
_engine_execution_timeout_dump_last_s: dict[str, float] = {}
_stack_trace_signal_handler_lock = threading.Lock()
_stack_trace_signal_handlers: dict[int, str] = {}


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
        )


def dump_engine_execution_timeout(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    timeout_s: float,
    stage: str,
    scheduler_snapshot: dict[str, Any] | None = None,
):
    if not _mark_engine_execution_timeout_dump(stage):
        return

    diagnostic_bundle_dir: Path | None = None
    with contextlib.suppress(Exception):
        logger.error(
            "V1 LLM engine stage '%s' has not completed after %.2f seconds "
            "(pid=%d). Dumping scheduler state and Python stack traces. "
            "Further dumps for this stage are throttled for %.0f seconds. "
            "Set VLLM_ENGINE_ITERATION_TIMEOUT_S=0 to disable this diagnostic.",
            stage,
            timeout_s,
            os.getpid(),
            ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S,
        )
        diagnostic_bundle_dir = _dump_engine_execution_context(
            "timeout",
            config,
            scheduler_output,
            scheduler_stats,
            stage=stage,
            timeout_s=timeout_s,
            scheduler_snapshot=scheduler_snapshot,
        )

    if diagnostic_bundle_dir is not None:
        _write_engine_traceback_dump(diagnostic_bundle_dir)

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


def dump_engine_no_progress(
    config: VllmConfig,
    scheduler_stats: SchedulerStats | None,
    timeout_s: float,
    progress_snapshot: dict[str, Any],
    scheduler_snapshot: dict[str, Any] | None = None,
) -> None:
    if not _mark_engine_execution_timeout_dump(ENGINE_NO_PROGRESS_STAGE):
        return

    diagnostic_bundle_dir: Path | None = None
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
        diagnostic_bundle_dir = _dump_engine_execution_context(
            "no_progress",
            config,
            None,
            scheduler_stats,
            stage=ENGINE_NO_PROGRESS_STAGE,
            timeout_s=timeout_s,
            scheduler_snapshot=_combine_no_progress_snapshots(
                progress_snapshot, scheduler_snapshot
            ),
        )

    if diagnostic_bundle_dir is not None:
        _write_engine_traceback_dump(diagnostic_bundle_dir)

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
        scheduler_stats_fn: Callable[[], SchedulerStats | None],
        scheduler_snapshot_fn: Callable[[], dict[str, Any] | None],
        time_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        self.config = config
        self.process_name = process_name
        self.timeout_s = timeout_s
        self.has_work_fn = has_work_fn
        self.scheduler_stats_fn = scheduler_stats_fn
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
        self._last_dump_progress_index = -1
        self._last_dump_s = 0.0
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def enabled(self) -> bool:
        return self.timeout_s is not None and self.timeout_s > 0

    def start(self) -> None:
        timeout_s = self.timeout_s
        if timeout_s is None or timeout_s <= 0 or self._thread is not None:
            return

        self._thread = threading.Thread(
            target=self._run,
            name=f"{self.process_name}NoProgressMonitor",
            daemon=True,
        )
        self._thread.start()
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

    def record_progress(
        self,
        stage: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        with self._lock:
            self._record_activity_unlocked(stage, details)
            self._progress_index += 1
            self._last_progress_s = self._last_activity_s

    def snapshot(self) -> dict[str, Any]:
        now_s = self.time_fn()
        with self._lock:
            return self._snapshot_unlocked(now_s)

    def maybe_dump_no_progress(self) -> bool:
        timeout_s = self.timeout_s
        if timeout_s is None or timeout_s <= 0:
            return False

        try:
            has_work = self.has_work_fn()
        except Exception:
            logger.exception("Failed to query EngineCore work state")
            return False

        if not has_work:
            self.record_progress("idle", {"has_work": False})
            return False

        now_s = self.time_fn()
        with self._lock:
            elapsed_s = now_s - self._last_progress_s
            if elapsed_s < timeout_s:
                return False
            if (
                self._last_dump_progress_index == self._progress_index
                and now_s - self._last_dump_s
                < ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
            ):
                return False
            progress_snapshot = self._snapshot_unlocked(now_s)
            self._last_dump_progress_index = self._progress_index
            self._last_dump_s = now_s

        scheduler_stats = self._make_scheduler_stats()
        scheduler_snapshot = self._make_scheduler_snapshot()
        dump_engine_no_progress(
            self.config,
            scheduler_stats,
            timeout_s,
            progress_snapshot,
            scheduler_snapshot,
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

    def _record_activity_unlocked(
        self,
        stage: str,
        details: dict[str, Any] | None,
    ) -> None:
        self._activity_index += 1
        self._stage = stage
        self._details = details or {}
        self._last_activity_s = self.time_fn()

    def _snapshot_unlocked(self, now_s: float) -> dict[str, Any]:
        return {
            "activity_index": self._activity_index,
            "elapsed_since_activity_s": max(0.0, now_s - self._last_activity_s),
            "elapsed_since_progress_s": max(0.0, now_s - self._last_progress_s),
            "last_activity_s": self._last_activity_s,
            "last_progress_s": self._last_progress_s,
            "pid": os.getpid(),
            "process_name": self.process_name,
            "progress_index": self._progress_index,
            "stage": self._stage,
            "stage_details": self._details,
            "timeout_s": self.timeout_s,
        }

    def _make_scheduler_stats(self) -> SchedulerStats | None:
        try:
            return self.scheduler_stats_fn()
        except Exception:
            logger.exception("Failed to collect V1 scheduler stats")
            return None

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
    if _is_protected_stack_trace_signal(signum):
        logger.warning(
            "Ignoring %s=%r because %s is reserved for process control.",
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


def _is_protected_stack_trace_signal(signum: int) -> bool:
    for signal_name in _PROTECTED_STACK_TRACE_SIGNAL_NAMES:
        protected_signum = getattr(signal, signal_name, None)
        if isinstance(protected_signum, int) and int(protected_signum) == signum:
            return True
    return False


def _mark_engine_execution_timeout_dump(stage: str) -> bool:
    now_s = time.monotonic()
    with _engine_execution_timeout_dump_lock:
        last_dump_s = _engine_execution_timeout_dump_last_s.get(stage)
        if (
            last_dump_s is not None
            and now_s - last_dump_s < ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
        ):
            return False
        _engine_execution_timeout_dump_last_s[stage] = now_s
        return True


def _combine_no_progress_snapshots(
    progress_snapshot: dict[str, Any],
    scheduler_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    combined_snapshot: dict[str, Any] = {
        "engine_progress": progress_snapshot,
    }
    if scheduler_snapshot is not None:
        combined_snapshot["scheduler"] = scheduler_snapshot
    return combined_snapshot


def _dump_engine_execution_context(
    reason: str,
    config: VllmConfig,
    scheduler_output: SchedulerOutput | None,
    scheduler_stats: SchedulerStats | None,
    stage: str | None = None,
    timeout_s: float | None = None,
    error: Exception | None = None,
    scheduler_snapshot: dict[str, Any] | None = None,
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
        if scheduler_output is not None:
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
                json.dumps(
                    scheduler_snapshot,
                    default=prepare_object_to_dump,
                    sort_keys=True,
                ),
            )
    except Exception:
        logger.exception("Error preparing object to dump")

    return _write_engine_diagnostic_bundle(
        reason=reason,
        config=config,
        scheduler_output_dump=scheduler_output_dump,
        scheduler_stats_dump=scheduler_stats_dump,
        stage=stage,
        timeout_s=timeout_s,
        error=error,
        scheduler_snapshot=scheduler_snapshot,
    )


def _write_engine_diagnostic_bundle(
    *,
    reason: str,
    config: VllmConfig,
    scheduler_output_dump: str | None,
    scheduler_stats_dump: str | None,
    stage: str | None,
    timeout_s: float | None,
    error: Exception | None,
    scheduler_snapshot: dict[str, Any] | None,
) -> Path | None:
    try:
        dump_root = _engine_diagnostic_dump_root(config)
        if dump_root is None:
            return None

        bundle_dir, created_at = _create_engine_diagnostic_bundle_dir(
            dump_root, reason, stage
        )
        scheduler_snapshot_file: str | None = None
        if scheduler_snapshot is not None:
            scheduler_snapshot_file = "scheduler_snapshot.json"
            _write_engine_diagnostic_json(
                bundle_dir / scheduler_snapshot_file,
                scheduler_snapshot,
            )

        context = {
            "bundle_version": ENGINE_DIAGNOSTIC_BUNDLE_VERSION,
            "config": str(config),
            "created_at": created_at.isoformat(),
            "exception": _format_engine_diagnostic_exception(error),
            "pid": os.getpid(),
            "reason": reason,
            "scheduler_snapshot_file": scheduler_snapshot_file,
            "scheduler_output": scheduler_output_dump,
            "scheduler_stats": scheduler_stats_dump,
            "stage": stage,
            "timeout_s": timeout_s,
            "vllm_version": VLLM_VERSION,
        }
        _write_engine_diagnostic_json(bundle_dir / "context.json", context)
        logger.error("Wrote V1 LLM engine diagnostic bundle to %s", bundle_dir)
        return bundle_dir
    except Exception:
        logger.exception("Failed to write V1 LLM engine diagnostic bundle")
        return None


def _engine_diagnostic_dump_root(config: VllmConfig) -> Path | None:
    if not hasattr(config, "compile_debug_dump_path"):
        return None

    debug_dump_path = config.compile_debug_dump_path()
    if debug_dump_path is None:
        return None
    return debug_dump_path / ENGINE_DIAGNOSTIC_DUMP_DIR


def _write_engine_diagnostic_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(
            payload,
            default=prepare_object_to_dump,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
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
    base_name = f"{timestamp}_pid{os.getpid()}_{reason_part}_{stage_part}"

    for suffix in range(1000):
        bundle_dir = dump_root / (
            base_name if suffix == 0 else f"{base_name}_{suffix}"
        )
        try:
            bundle_dir.mkdir(parents=True, exist_ok=False)
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
    return {
        "message": str(error),
        "traceback": "".join(
            traceback_utils.format_exception(type(error), error, error.__traceback__)
        ),
        "type": f"{type(error).__module__}.{type(error).__qualname__}",
    }


def _write_engine_traceback_dump(bundle_dir: Path) -> None:
    try:
        with (bundle_dir / "stacks.txt").open("w", encoding="utf-8") as dump_file:
            faulthandler.dump_traceback(file=dump_file, all_threads=True)
    except Exception:
        logger.exception("Failed to write V1 LLM engine stack trace dump")


class EngineExecutionTimeoutDumper:
    """Dumps engine state if a model execution stage exceeds a timeout."""

    def __init__(
        self,
        config: VllmConfig,
        scheduler_output: SchedulerOutput,
        scheduler_stats: SchedulerStats | None,
        timeout_s: float | None,
        stage: str,
        scheduler_snapshot_fn: Callable[[], dict[str, Any] | None] | None = None,
    ) -> None:
        self.config = config
        self.scheduler_output = scheduler_output
        self.scheduler_stats = scheduler_stats
        self.timeout_s = timeout_s
        self.stage = stage
        self.scheduler_snapshot_fn = scheduler_snapshot_fn
        self._timer: threading.Timer | None = None

    def __enter__(self) -> Self:
        if self.timeout_s is None or self.timeout_s <= 0:
            return self

        self._timer = threading.Timer(self.timeout_s, self._dump_timeout)
        self._timer.daemon = True
        self._timer.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if self._timer is not None:
            self._timer.cancel()

    def _dump_timeout(self) -> None:
        dump_engine_execution_timeout(
            self.config,
            self.scheduler_output,
            self.scheduler_stats,
            self.timeout_s or 0,
            self.stage,
            self._make_scheduler_snapshot(),
        )

    def _make_scheduler_snapshot(self) -> dict[str, Any] | None:
        if self.scheduler_snapshot_fn is None:
            return None
        try:
            return self.scheduler_snapshot_fn()
        except Exception:
            logger.exception("Failed to collect V1 scheduler diagnostic snapshot")
            return None
