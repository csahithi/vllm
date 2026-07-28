# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
from datetime import datetime, timezone
import enum
import faulthandler
import json
import os
from pathlib import Path
import sys
import threading
import time
import traceback as traceback_utils
from types import TracebackType

import torch
from typing_extensions import Self

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.metrics.stats import SchedulerStats
from vllm.version import __version__ as VLLM_VERSION

logger = init_logger(__name__)

ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S = 300.0
ENGINE_DIAGNOSTIC_BUNDLE_VERSION = 1
ENGINE_DIAGNOSTIC_DUMP_DIR = "engine_diagnostics"
_engine_execution_timeout_dump_lock = threading.Lock()
_engine_execution_timeout_dump_last_s: dict[str, float] = {}


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
        )


def dump_engine_execution_timeout(
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    timeout_s: float,
    stage: str,
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
        )

    if diagnostic_bundle_dir is not None:
        _write_engine_traceback_dump(diagnostic_bundle_dir)

    with contextlib.suppress(Exception):
        faulthandler.dump_traceback(file=sys.stderr, all_threads=True)


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


def _dump_engine_execution_context(
    reason: str,
    config: VllmConfig,
    scheduler_output: SchedulerOutput,
    scheduler_stats: SchedulerStats | None,
    stage: str | None = None,
    timeout_s: float | None = None,
    error: Exception | None = None,
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
) -> Path | None:
    try:
        dump_root = _engine_diagnostic_dump_root(config)
        if dump_root is None:
            return None

        bundle_dir, created_at = _create_engine_diagnostic_bundle_dir(
            dump_root, reason, stage
        )
        context = {
            "bundle_version": ENGINE_DIAGNOSTIC_BUNDLE_VERSION,
            "config": str(config),
            "created_at": created_at.isoformat(),
            "exception": _format_engine_diagnostic_exception(error),
            "pid": os.getpid(),
            "reason": reason,
            "scheduler_output": scheduler_output_dump,
            "scheduler_stats": scheduler_stats_dump,
            "stage": stage,
            "timeout_s": timeout_s,
            "vllm_version": VLLM_VERSION,
        }
        (bundle_dir / "context.json").write_text(
            json.dumps(context, indent=2, sort_keys=True),
            encoding="utf-8",
        )
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
    ) -> None:
        self.config = config
        self.scheduler_output = scheduler_output
        self.scheduler_stats = scheduler_stats
        self.timeout_s = timeout_s
        self.stage = stage
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
        )
