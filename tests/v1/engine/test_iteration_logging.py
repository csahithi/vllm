# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import os
import signal
import stat
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.engine.core as engine_core_module
import vllm.v1.engine.core_client as core_client_module
import vllm.v1.engine.utils as engine_utils
import vllm.v1.executor.multiproc_executor as multiproc_executor_module
import vllm.v1.utils as v1_utils
from vllm.config import ModelConfig, SpeculativeConfig, VllmConfig
from vllm.logging_utils import dump_input
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.interface import SchedulerInterface
from vllm.v1.core.sched.output import (
    CachedRequestData,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.worker.worker_base import WorkerWrapperBase


class FakeEngineCore:
    def _make_iteration_details_stats(
        self, iteration_details: SchedulerIterationDetails
    ) -> SchedulerStats:
        return SchedulerStats(iteration_details=iteration_details)


def make_iteration_details() -> SchedulerIterationDetails:
    return SchedulerIterationDetails(
        iteration_index=1,
        num_ctx_requests=2,
        num_ctx_tokens=3,
        num_generation_requests=4,
        num_generation_tokens=5,
        elapsed_ms=6.7,
    )


def make_fake_engine(log_stats: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        log_stats=log_stats,
        vllm_config=SimpleNamespace(
            observability_config=SimpleNamespace(
                enable_logging_iteration_details=True,
            )
        ),
    )


class FakeFuture:
    def __init__(self, result, events=None, label=None):
        self._result = result
        self._events = events
        self._label = label

    def result(self):
        if self._events is not None and self._label is not None:
            self._events.append(("call", self._label))
        return self._result


class FakeStageScheduler:
    def __init__(self, scheduler_output, has_requests=True):
        self.scheduler_output = scheduler_output
        self._has_requests = has_requests
        self.updated_with = None

    def has_requests(self):
        return self._has_requests

    def schedule(self, throttle_prefills):
        return self.scheduler_output

    def get_grammar_bitmask(self, scheduler_output):
        return "grammar"

    def update_from_output(self, scheduler_output, model_output):
        self.updated_with = (scheduler_output, model_output)
        return {}


class FakeStageModelExecutor:
    def __init__(self, execute_result=None, sample_result=None):
        self.events = []
        self.execute_future = FakeFuture(
            execute_result, self.events, "execute_model.result"
        )
        self.sample_future = FakeFuture(
            sample_result, self.events, "sample_tokens.result"
        )
        self.sample_result = sample_result
        self.sample_calls = []

    def execute_model(self, scheduler_output, non_block=False):
        self.events.append(("call", "execute_model"))
        return self.execute_future

    def sample_tokens(self, grammar_output, non_block=False):
        self.events.append(("call", "sample_tokens"))
        self.sample_calls.append((grammar_output, non_block))
        return self.sample_future if non_block else self.sample_result


class FakeStageEngine:
    def __init__(self, scheduler, model_executor=None):
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.stages = []
        self.stage_states = []
        self.prepared_timeout_outputs = []
        self.events = model_executor.events if model_executor is not None else []
        self.batch_queue = None
        self.batch_queue_size = 2
        self.is_ec_consumer = True
        self.is_pooling_model = False
        self.check_for_draft_tokens = False

    @contextmanager
    def capture_iteration_details(self, scheduler_output):
        yield None

    @contextmanager
    def log_error_detail(self, scheduler_output):
        yield

    @contextmanager
    def dump_on_slow_execution(self, stage, snapshot):
        self.stages.append(stage)
        self.stage_states.append(snapshot)
        self.events.append(("enter", stage))
        try:
            yield
        finally:
            self.events.append(("exit", stage))

    def _prepare_timeout_diagnostic_state(self, scheduler_output):
        self.prepared_timeout_outputs.append(scheduler_output)
        return dump_input.EngineExecutionTimeoutSnapshot(
            {"scheduler_output": scheduler_output}, {}
        )

    def _should_throttle_prefills(self):
        return False

    def _process_aborts_queue(self):
        return

    def _attach_iteration_details(self, outputs, iteration_details):
        return


class FakeDiagnosticWorker:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


class FakeProgressMonitor:
    def __init__(self) -> None:
        self.activities: list[tuple[str, dict[str, Any] | None]] = []
        self.progress: list[tuple[str, dict[str, Any] | None]] = []

    def record_activity(
        self, stage: str, details: dict[str, Any] | None = None
    ) -> None:
        self.activities.append((stage, details))

    def record_progress(
        self, stage: str, details: dict[str, Any] | None = None
    ) -> None:
        self.progress.append((stage, details))

    def record_idle(self) -> None:
        self.record_progress("idle", {"has_work": False})


class FakeProc:
    def __init__(
        self,
        name: str,
        *,
        pid: int,
        exitcode: int | None,
        sentinel: int,
        pending_exitcode_reads: int = 0,
    ) -> None:
        self.name = name
        self.pid = pid
        self._exitcode = exitcode
        self._pending_exitcode_reads = pending_exitcode_reads
        self.sentinel = sentinel

    @property
    def exitcode(self) -> int | None:
        if self._pending_exitcode_reads:
            self._pending_exitcode_reads -= 1
            return None
        return self._exitcode


@pytest.fixture(autouse=True)
def reset_stack_trace_signal_handler_state() -> Iterator[None]:
    with dump_input._stack_trace_signal_handler_lock:
        dump_input._stack_trace_signal_handlers.clear()
    yield
    with dump_input._stack_trace_signal_handler_lock:
        dump_input._stack_trace_signal_handlers.clear()


def make_timeout_scheduler_output() -> SimpleNamespace:
    sampling_params = SimpleNamespace(
        extra_args={"private": "private-extra-arg"},
        max_tokens=8,
        stop="private-stop-string",
        stop_token_ids=[1, 2],
        structured_outputs=SimpleNamespace(json="private-schema"),
        temperature=0.25,
        top_k=10,
        top_p=0.9,
    )
    return SimpleNamespace(
        finished_req_ids=set(),
        kv_connector_metadata=None,
        num_scheduled_tokens={"request-123": 4, "request-456": 2},
        pending_structured_output_tokens=False,
        preempted_req_ids=None,
        scheduled_cached_reqs=SimpleNamespace(
            all_token_ids={"request-456": [10, 11, 12, 13]},
            new_block_ids=[([1, 2],)],
            new_token_ids=[[10, 11]],
            num_computed_tokens=[3],
            num_output_tokens=[1],
            num_reqs=1,
            req_ids=["request-456"],
            resumed_req_ids=set(),
        ),
        scheduled_encoder_inputs={},
        scheduled_new_reqs=[
            SimpleNamespace(
                block_ids=([1, 2, 3],),
                lora_request=None,
                mm_features=[],
                num_computed_tokens=0,
                pooling_params=None,
                prefill_token_ids=None,
                prompt_embeds=None,
                prompt_token_ids=[101, 102, 103],
                req_id="request-123",
                sampling_params=sampling_params,
            )
        ],
        scheduled_spec_decode_tokens={},
        total_num_scheduled_tokens=6,
    )


def make_timeout_config() -> SimpleNamespace:
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=16,
            cache_dtype="auto",
            gpu_memory_utilization=0.9,
        ),
        model_config=SimpleNamespace(
            dtype="float16",
            enforce_eager=False,
            hf_config=SimpleNamespace(
                architectures=["TestArchitecture"],
                model_type="test-model-type",
            ),
            max_model_len=4096,
            model="private/model/path",
            runner_type="generate",
        ),
        offload_config=SimpleNamespace(
            offload_backend="auto",
            uva=SimpleNamespace(cpu_offload_gb=0),
        ),
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            data_parallel_size=1,
            pipeline_parallel_size=1,
            rank=0,
            tensor_parallel_size=2,
        ),
        scheduler_config=SimpleNamespace(
            async_scheduling=False,
            enable_chunked_prefill=True,
            max_num_batched_tokens=2048,
            max_num_seqs=128,
            policy="fcfs",
        ),
        speculative_config=None,
    )


def make_timeout_snapshot(
    scheduler_output: SimpleNamespace | None = None,
    scheduler_state: dict[str, Any] | None = None,
) -> dump_input.EngineExecutionTimeoutSnapshot:
    return dump_input.make_engine_execution_timeout_snapshot(
        scheduler_output or make_timeout_scheduler_output(),
        scheduler_state,
    )


def enable_engine_diagnostic_bundles(monkeypatch, tmp_path: Path) -> SimpleNamespace:
    monkeypatch.setattr(
        dump_input.envs,
        "VLLM_ENGINE_DIAGNOSTIC_DUMP_PATH",
        str(tmp_path),
    )
    return make_timeout_config()


def make_diagnostic_request(
    request_id: str,
    *,
    arrival_time: float,
    status,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        arrival_time=arrival_time,
        status=status,
        ec_transfer_params=None,
        kv_transfer_params=None,
        lora_request=None,
        pooling_params=None,
        sampling_params=SimpleNamespace(max_tokens=4),
        is_prefill_chunk=False,
        max_tokens=4,
        num_computed_tokens=3,
        num_encoder_inputs=1,
        num_in_flight_tokens=2,
        num_output_tokens=1,
        num_preemptions=1,
        num_prompt_tokens=8,
        spec_token_ids=[1, 2],
        num_tokens=9,
        priority=0,
        use_structured_output=False,
    )


def test_scheduler_diagnostic_snapshot_summarizes_requests_and_resources(
    monkeypatch,
):
    scheduler = cast(Any, object.__new__(scheduler_module.Scheduler))
    running_req = make_diagnostic_request(
        "running-req",
        arrival_time=990.0,
        status=scheduler_module.RequestStatus.RUNNING,
    )
    waiting_req = make_diagnostic_request(
        "waiting-req",
        arrival_time=980.0,
        status=scheduler_module.RequestStatus.WAITING,
    )
    scheduler.current_step = 12
    scheduler.deferred_frees = deque([(10, [object(), object()])])
    scheduler.finished_req_ids = {"finished-req"}
    scheduler.max_model_len = 32768
    scheduler.max_num_running_reqs = 16
    scheduler.max_num_scheduled_tokens = 1024
    scheduler.requests = {
        running_req.request_id: running_req,
        waiting_req.request_id: waiting_req,
    }
    scheduler.num_waiting_for_streaming_input = 1
    scheduler._pause_state = scheduler_module.PauseState.UNPAUSED
    scheduler.policy = scheduler_module.SchedulingPolicy.FCFS
    scheduler.prefill_capacity_bound = True
    scheduler.processed_step_seq = 9
    scheduler.running = [running_req]
    scheduler.waiting = [waiting_req]
    scheduler.skipped_waiting = []
    scheduler.kv_cache_manager = SimpleNamespace(
        usage=0.25,
        num_kv_cache_groups=2,
        watermark_blocks=3,
        block_pool=SimpleNamespace(
            num_gpu_blocks=11,
            get_num_free_blocks=lambda: 6,
        ),
    )
    scheduler.encoder_cache_manager = SimpleNamespace(
        cache_size=10,
        num_free_slots=6,
        num_freeable_slots=7,
        cached={"hash": {"running-req"}},
        freeable={"old-hash": 1},
        freed=["freed-hash"],
    )
    scheduler.connector = SimpleNamespace(has_pending_push_work=lambda: True)
    scheduler.ec_connector = object()
    scheduler.finished_recving_kv_req_ids = {"loaded-req"}
    scheduler.failed_recving_kv_req_ids = {"failed-req"}
    scheduler.defer_block_free = True
    monkeypatch.setattr(
        scheduler_module,
        "time",
        SimpleNamespace(time=lambda: 1000.0),
    )

    snapshot = scheduler_module.Scheduler.make_diagnostic_snapshot(scheduler)

    assert snapshot["scheduler"]["current_step"] == 12
    assert snapshot["scheduler"]["deferred_free_blocks"] == 2
    assert snapshot["requests"]["running"]["requests"][0]["request_id"] == (
        "running-req"
    )
    assert snapshot["requests"]["running"]["requests"][0]["age_s"] == 10.0
    assert snapshot["requests"]["waiting"]["oldest_sampled_request_id"] == (
        "waiting-req"
    )
    assert snapshot["kv_cache"] == {
        "num_allocatable_blocks": 10,
        "num_free_blocks": 6,
        "num_gpu_blocks": 11,
        "num_kv_cache_groups": 2,
        "num_used_blocks": 4,
        "usage": 0.25,
        "watermark_blocks": 3,
    }
    assert snapshot["encoder_cache"]["num_cached_entries"] == 1
    assert snapshot["connectors"]["kv_connector_pending_push_work"]


def test_scheduler_diagnostic_snapshot_bounds_real_request_id():
    request_id = f"request-{'x' * 500}-suffix"
    request = Request(
        request_id=request_id,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(max_tokens=4),
        pooling_params=None,
        arrival_time=990.0,
    )
    request.status = RequestStatus.RUNNING

    snapshot = Scheduler._make_request_diagnostic_snapshot(
        cast(Any, None), request, now_s=1000.0
    )

    assert snapshot["age_s"] == 10.0
    assert snapshot["num_prompt_tokens"] == 3
    assert snapshot["status"] == "RUNNING"
    assert snapshot["request_id"] == f"{request_id[:160]}...{request_id[-64:]}"
    assert snapshot["request_id_length"] == len(request_id)
    assert snapshot["request_id_truncated"] is True
    assert len(snapshot["request_id_sha256"]) == 64


def test_make_scheduler_diagnostic_snapshot_returns_none_on_failure():
    def raise_snapshot_error():
        raise RuntimeError("snapshot failed")

    engine = SimpleNamespace(
        scheduler=SimpleNamespace(make_diagnostic_snapshot=raise_snapshot_error)
    )

    assert EngineCore.make_scheduler_diagnostic_snapshot(engine) is None


def test_capture_iteration_details_disabled_without_log_stats():
    engine = make_fake_engine(log_stats=False)

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is None

    assert not hasattr(engine, "_iteration_index")


def test_capture_iteration_details_fills_elapsed_time():
    engine = make_fake_engine()

    with EngineCore.capture_iteration_details(engine, None) as iteration_details:
        assert iteration_details is not None
        assert iteration_details.elapsed_ms == 0.0
        assert iteration_details.is_dummy
        time.sleep(0.001)

    assert iteration_details is not None
    assert iteration_details.elapsed_ms > 0.0
    assert engine._iteration_index == 1


def test_attach_iteration_details_uses_existing_output():
    iteration_details = make_iteration_details()
    outputs = {
        2: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
        1: EngineCoreOutputs(scheduler_stats=SchedulerStats()),
    }

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert 0 not in outputs
    assert outputs[2].scheduler_stats is not None
    assert outputs[2].scheduler_stats.iteration_details == iteration_details
    assert outputs[1].scheduler_stats is not None
    assert outputs[1].scheduler_stats.iteration_details is None


def test_attach_iteration_details_falls_back_to_client_zero_without_outputs():
    iteration_details = make_iteration_details()
    outputs: dict[int, EngineCoreOutputs] = {}

    EngineCore._attach_iteration_details(FakeEngineCore(), outputs, iteration_details)

    assert set(outputs) == {0}
    assert outputs[0].scheduler_stats is not None
    assert outputs[0].scheduler_stats.iteration_details == iteration_details


def test_engine_execution_timeout_watchdog_disabled_is_lazy():
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0,
    )

    watchdog.start()
    generation = watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    watchdog.disarm(generation)
    watchdog.stop()

    assert generation is None
    assert watchdog._thread is None
    assert not watchdog.enabled


def test_engine_execution_timeout_watchdog_fails_open_on_thread_start_error(
    monkeypatch,
):
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
    )

    def fail_start():
        raise RuntimeError("thread limit reached")

    failing_thread = SimpleNamespace(start=fail_start)
    monkeypatch.setattr(watchdog, "_create_thread", lambda: failing_thread)

    watchdog.start()
    generation = watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert generation is None
    assert watchdog._thread is None
    assert not watchdog.enabled


def test_engine_execution_timeout_watchdog_ignores_stale_disarm(monkeypatch):
    now_s = [0.0]
    dumps = []
    dump_completed = threading.Event()

    def record_dump(config, snapshot, timeout_s, stage):
        dumps.append((snapshot, stage))
        dump_completed.set()

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", record_dump)
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
        time_fn=lambda: now_s[0],
    )
    watchdog.start()

    try:
        stale_output = make_timeout_scheduler_output()
        stale_output.scheduled_new_reqs[0].req_id = "stale-generation-request"
        current_output = make_timeout_scheduler_output()
        current_output.scheduled_new_reqs[0].req_id = "current-generation-request"
        stale_generation = watchdog.arm(
            make_timeout_snapshot(stale_output, {"snapshot": 1}),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        watchdog.arm(
            make_timeout_snapshot(current_output, {"snapshot": 2}),
            engine_core_module.SAMPLE_TOKENS_STAGE,
        )
        watchdog.disarm(stale_generation)
        now_s[0] = 11.0
        watchdog._wake_event.set()

        assert dump_completed.wait(timeout=1.0)
        assert len(dumps) == 1
        snapshot, stage = dumps[0]
        assert snapshot.scheduler_output_summary["request_samples"][0][
            "request_id"
        ] == ("current-generation-request")
        assert snapshot.scheduler_queue_summary == {"snapshot": 2}
        assert stage == engine_core_module.SAMPLE_TOKENS_STAGE
        diagnostic_thread = watchdog._diagnostic_thread
        assert diagnostic_thread is not None
        diagnostic_thread.join(timeout=1.0)
    finally:
        watchdog.stop()


def test_engine_execution_timeout_watchdog_rearm_wakes_only_when_needed():
    now_s = [0.0]
    wake_count = 0

    def record_wake():
        nonlocal wake_count
        wake_count += 1

    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=10.0,
        time_fn=lambda: now_s[0],
    )
    watchdog._wake_event = SimpleNamespace(set=record_wake)

    watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    assert wake_count == 1

    now_s[0] = 1.0
    watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.SAMPLE_TOKENS_STAGE,
    )
    assert wake_count == 1

    watchdog.timeout_s = 1.0
    now_s[0] = 2.0
    generation = watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
    )
    assert wake_count == 2

    watchdog.disarm(generation)
    watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    assert wake_count == 3


def test_engine_execution_timeout_watchdog_keeps_arm_and_stop_nonblocking(
    monkeypatch,
):
    dump_started = threading.Event()
    release_dump = threading.Event()
    skipped_dump_emitted = threading.Event()
    dump_calls = 0

    def blocking_dump(*args):
        nonlocal dump_calls
        dump_calls += 1
        dump_started.set()
        assert release_dump.wait(timeout=2.0)

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", blocking_dump)
    monkeypatch.setattr(
        dump_input,
        "_emit_engine_execution_timeout",
        lambda *args: skipped_dump_emitted.set(),
    )
    monkeypatch.setattr(
        dump_input,
        "ENGINE_EXECUTION_TIMEOUT_WATCHDOG_STOP_TIMEOUT_S",
        0.01,
    )
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )
    watchdog.start()
    arm_finished = threading.Event()
    stop_finished = threading.Event()
    generations = []

    try:
        watchdog.arm(
            make_timeout_snapshot(scheduler_state={"snapshot": 1}),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert dump_started.wait(timeout=1.0)

        def arm_again():
            generations.append(
                watchdog.arm(
                    make_timeout_snapshot(scheduler_state={"snapshot": 2}),
                    engine_core_module.SAMPLE_TOKENS_STAGE,
                )
            )
            arm_finished.set()

        arm_thread = threading.Thread(target=arm_again)
        arm_thread.start()
        assert arm_finished.wait(timeout=1.0)
        arm_thread.join(timeout=1.0)
        assert generations[0] is not None
        assert skipped_dump_emitted.wait(timeout=1.0)
        assert dump_calls == 1

        def stop_watchdog():
            watchdog.stop()
            stop_finished.set()

        stop_thread = threading.Thread(target=stop_watchdog)
        stop_thread.start()
        assert stop_finished.wait(timeout=1.0)
        stop_thread.join(timeout=1.0)
        assert watchdog._thread is not None
        assert not watchdog._thread.is_alive()
        assert watchdog._diagnostic_thread is not None
        assert watchdog._diagnostic_thread.is_alive()
    finally:
        release_dump.set()
        watchdog.stop()
        if watchdog._thread is not None:
            watchdog._thread.join(timeout=1.0)
        if watchdog._diagnostic_thread is not None:
            watchdog._diagnostic_thread.join(timeout=1.0)

    assert watchdog._thread is not None
    assert not watchdog._thread.is_alive()
    assert watchdog._diagnostic_thread is not None
    assert not watchdog._diagnostic_thread.is_alive()


def test_engine_execution_timeout_watchdog_disarm_suppresses_dump(monkeypatch):
    dumped = threading.Event()
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumped.set(),
    )
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )

    generation = watchdog.arm(
        make_timeout_snapshot(),
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    watchdog.disarm(generation)
    watchdog.start()
    try:
        assert not dumped.wait(timeout=0.05)
    finally:
        watchdog.stop()


def test_engine_execution_timeout_real_timeout_emits_useful_summary(monkeypatch):
    logs = []
    traceback_dumped = threading.Event()
    scheduler_snapshot = {"schema_version": 1, "kv_cache": {"usage": 0.5}}

    def record_log(message, *args):
        logs.append(message % args)

    monkeypatch.setattr(dump_input.logger, "error", record_log)
    monkeypatch.setattr(
        dump_input.faulthandler,
        "dump_traceback",
        lambda *args, **kwargs: traceback_dumped.set(),
    )

    def make_scheduler_snapshot():
        assert traceback_dumped.is_set()
        return scheduler_snapshot

    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
        scheduler_snapshot_fn=make_scheduler_snapshot,
    )
    watchdog.start()

    try:
        watchdog.arm(
            make_timeout_snapshot(
                scheduler_state={"num_running_reqs": 1, "num_waiting_reqs": 0}
            ),
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        )
        assert traceback_dumped.wait(timeout=1.0)
        diagnostic_thread = watchdog._diagnostic_thread
        assert diagnostic_thread is not None
        diagnostic_thread.join(timeout=1.0)
        assert not diagnostic_thread.is_alive()
    finally:
        watchdog.stop()

    combined_logs = "\n".join(logs)
    assert engine_core_module.EXECUTE_MODEL_WAIT_STAGE in combined_logs
    assert "request-123" in combined_logs
    assert "request-456" in combined_logs
    assert "temperature" in combined_logs
    assert "num_running_reqs" in combined_logs
    snapshot_log = next(
        message
        for message in logs
        if message.startswith("Scheduler diagnostic snapshot: ")
    )
    assert json.loads(snapshot_log.removeprefix("Scheduler diagnostic snapshot: ")) == (
        scheduler_snapshot
    )
    assert "private-stop-string" not in combined_logs
    assert "private/model/path" not in combined_logs


def test_engine_execution_timeout_context_failure_is_reported(monkeypatch):
    errors = []
    traceback_dumped = threading.Event()

    def fail_context(*args):
        raise RuntimeError("context failure")

    monkeypatch.setattr(dump_input, "_dump_engine_timeout_context", fail_context)
    monkeypatch.setattr(
        dump_input.logger,
        "exception",
        lambda message, *args: errors.append(message % args),
    )
    monkeypatch.setattr(
        dump_input.faulthandler,
        "dump_traceback",
        lambda *args, **kwargs: traceback_dumped.set(),
    )

    dump_input.dump_engine_execution_timeout(
        make_timeout_config(),
        make_timeout_snapshot(),
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert errors == ["Failed to dump V1 engine timeout context"]
    assert traceback_dumped.is_set()


def test_log_error_detail_forwards_and_reraises_original_error(monkeypatch):
    config = object()
    scheduler_output = object()
    scheduler_stats = object()
    scheduler_snapshot = {"requests": {"running": {"count": 1}}}
    engine = SimpleNamespace(
        vllm_config=config,
        scheduler=SimpleNamespace(make_stats=lambda: scheduler_stats),
        make_scheduler_diagnostic_snapshot=lambda: scheduler_snapshot,
    )
    calls = []

    def record_dump(
        actual_config,
        actual_output,
        actual_stats,
        *,
        error,
        scheduler_snapshot_fn,
    ):
        calls.append(
            (
                actual_config,
                actual_output,
                actual_stats,
                error,
                scheduler_snapshot_fn,
            )
        )

    monkeypatch.setattr(engine_core_module, "dump_engine_exception", record_dump)
    error = RuntimeError("execute failed")

    with (
        pytest.raises(RuntimeError) as caught,
        EngineCore.log_error_detail(engine, scheduler_output),
    ):
        raise error

    assert caught.value is error
    assert len(calls) == 1
    assert calls[0][:4] == (config, scheduler_output, scheduler_stats, error)
    assert calls[0][4]() is scheduler_snapshot


def test_dump_engine_exception_does_not_wait_for_scheduler_snapshot(
    tmp_path, monkeypatch
):
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    bundle_written = threading.Event()

    def make_scheduler_snapshot():
        snapshot_started.set()
        assert release_snapshot.wait(timeout=2.0)
        return {"schema_version": 1}

    def record_bundle(**kwargs):
        assert kwargs["scheduler_snapshot"] == {"schema_version": 1}
        bundle_written.set()
        return None

    config = enable_engine_diagnostic_bundles(monkeypatch, tmp_path)
    monkeypatch.setattr(dump_input, "ENGINE_DIAGNOSTIC_WRITE_TIMEOUT_S", 0.0)
    monkeypatch.setattr(dump_input, "_write_engine_diagnostic_bundle", record_bundle)

    try:
        dump_input.dump_engine_exception(
            config,
            SimpleNamespace(request_id="req-1"),
            None,
            scheduler_snapshot_fn=make_scheduler_snapshot,
        )

        assert snapshot_started.wait(timeout=1.0)
        assert not bundle_written.is_set()
    finally:
        release_snapshot.set()
    assert bundle_written.wait(timeout=1.0)


def test_engine_execution_context_writes_diagnostic_bundle(tmp_path, monkeypatch):
    scheduler_output = SimpleNamespace(request_id="req-1", token_ids=[1, 2, 3])
    scheduler_stats = SchedulerStats(num_running_reqs=1)
    error = RuntimeError("model execution failed")

    bundle_dir = dump_input._dump_engine_execution_context(
        reason="exception",
        config=enable_engine_diagnostic_bundles(monkeypatch, tmp_path),
        scheduler_output=scheduler_output,
        scheduler_stats=scheduler_stats,
        error=error,
        scheduler_snapshot={"kv_cache": {"usage": 0.5}},
    )

    assert bundle_dir is not None
    assert bundle_dir.parent == (
        tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    )

    context = json.loads((bundle_dir / "context.json").read_text(encoding="utf-8"))
    assert context["bundle_version"] == dump_input.ENGINE_DIAGNOSTIC_BUNDLE_VERSION
    assert context["reason"] == "exception"
    assert context["stage"] is None
    assert context["timeout_s"] is None
    assert context["scheduler_output_summary"] is None
    assert context["scheduler_queue_summary"] is None
    assert context["scheduler_snapshot_file"] == "scheduler_snapshot.json"
    assert "req-1" in context["scheduler_output_text"]
    assert "num_running_reqs=1" in context["scheduler_stats_text"]
    assert context["exception"]["type"] == "builtins.RuntimeError"
    assert context["exception"]["message"] == "model execution failed"
    scheduler_snapshot = json.loads(
        (bundle_dir / "scheduler_snapshot.json").read_text(encoding="utf-8")
    )
    assert scheduler_snapshot == {"kv_cache": {"usage": 0.5}}
    assert json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8")) == {
        "artifacts": {
            "context.json": "written",
            "scheduler_snapshot.json": "written",
        },
        "bundle_version": dump_input.ENGINE_DIAGNOSTIC_BUNDLE_VERSION,
        "complete": True,
        "files": ["context.json", "scheduler_snapshot.json"],
    }
    if os.name != "nt":
        assert stat.S_IMODE(bundle_dir.parent.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(bundle_dir.parent.stat().st_mode) == 0o700
        assert stat.S_IMODE(bundle_dir.stat().st_mode) == 0o700
        assert stat.S_IMODE((bundle_dir / "context.json").stat().st_mode) == 0o600
        assert stat.S_IMODE((bundle_dir / "manifest.json").stat().st_mode) == 0o600
        assert (
            stat.S_IMODE((bundle_dir / "scheduler_snapshot.json").stat().st_mode)
            == 0o600
        )


def test_scheduler_snapshot_write_failure_preserves_diagnostic_bundle(
    tmp_path, monkeypatch
):
    real_write_json = dump_input._write_engine_diagnostic_json

    def fail_scheduler_snapshot(path, value, *, max_bytes):
        if path.name == "scheduler_snapshot.json":
            raise ValueError("snapshot too large")
        real_write_json(path, value, max_bytes=max_bytes)

    monkeypatch.setattr(
        dump_input,
        "_write_engine_diagnostic_json",
        fail_scheduler_snapshot,
    )
    bundle_dir = dump_input._dump_engine_execution_context(
        reason="exception",
        config=enable_engine_diagnostic_bundles(monkeypatch, tmp_path),
        scheduler_output=SimpleNamespace(request_id="req-1"),
        scheduler_stats=None,
        scheduler_snapshot={"schema_version": 1},
    )

    assert bundle_dir is not None
    context = json.loads((bundle_dir / "context.json").read_text(encoding="utf-8"))
    assert context["scheduler_snapshot_file"] is None
    assert not (bundle_dir / "scheduler_snapshot.json").exists()
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["artifacts"] == {
        "context.json": "written",
        "scheduler_snapshot.json": "failed",
    }


def test_engine_execution_context_ignores_compilation_debug_dump_path(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(dump_input.envs, "VLLM_ENGINE_DIAGNOSTIC_DUMP_PATH", None)
    bundle_dir = dump_input._dump_engine_execution_context(
        reason="exception",
        config=SimpleNamespace(compile_debug_dump_path=lambda: tmp_path),
        scheduler_output=SimpleNamespace(request_id="req-1"),
        scheduler_stats=None,
    )

    assert bundle_dir is None
    assert not (tmp_path / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR).exists()


def test_engine_execution_timeout_writes_stack_bundle(tmp_path, monkeypatch):
    def record_traceback(file, all_threads):
        assert all_threads
        try:
            file.write("stack dump\n")
        except TypeError:
            file.write(b"stack dump\n")

    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs[0].req_id = "req-2"
    scheduler_snapshot = {"schema_version": 1, "kv_cache": {"usage": 0.5}}

    dump_input.dump_engine_execution_timeout(
        config=enable_engine_diagnostic_bundles(monkeypatch, tmp_path),
        snapshot=make_timeout_snapshot(
            scheduler_output,
            scheduler_state={"num_waiting_reqs": 2},
        ),
        timeout_s=2.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_snapshot=scheduler_snapshot,
    )

    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    bundles = list(dump_root.iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "stacks.txt").read_text(encoding="utf-8") == "stack dump\n"

    context = json.loads((bundles[0] / "context.json").read_text(encoding="utf-8"))
    assert context["reason"] == "timeout"
    assert context["stage"] == engine_core_module.EXECUTE_MODEL_WAIT_STAGE
    assert context["timeout_s"] == 2.0
    assert context["scheduler_output_text"] is None
    assert context["scheduler_stats_text"] is None
    assert (
        context["scheduler_output_summary"]["request_samples"][0]["request_id"]
        == "req-2"
    )
    assert context["scheduler_queue_summary"]["num_waiting_reqs"] == 2
    assert context["scheduler_snapshot_file"] == "scheduler_snapshot.json"
    assert context["exception"] is None
    assert (
        json.loads((bundles[0] / "scheduler_snapshot.json").read_text(encoding="utf-8"))
        == scheduler_snapshot
    )
    manifest = json.loads((bundles[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["artifacts"] == {
        "context.json": "written",
        "scheduler_snapshot.json": "written",
        "stacks.txt": "written",
    }
    assert manifest["files"] == [
        "context.json",
        "scheduler_snapshot.json",
        "stacks.txt",
    ]


def test_engine_execution_timeout_stack_failure_is_fail_open(tmp_path, monkeypatch):
    stderr_dumped = threading.Event()
    real_mkstemp = dump_input.tempfile.mkstemp

    def fail_stack_file(*args, **kwargs):
        if kwargs.get("prefix") == ".stacks.txt.":
            raise OSError("disk full")
        return real_mkstemp(*args, **kwargs)

    def record_traceback(file, all_threads):
        assert all_threads
        if file is dump_input.sys.stderr:
            stderr_dumped.set()

    monkeypatch.setattr(dump_input.tempfile, "mkstemp", fail_stack_file)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)

    dump_input.dump_engine_execution_timeout(
        config=enable_engine_diagnostic_bundles(monkeypatch, tmp_path),
        snapshot=make_timeout_snapshot(),
        timeout_s=2.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    bundles = list(dump_root.iterdir())
    assert stderr_dumped.is_set()
    assert len(bundles) == 1
    manifest = json.loads((bundles[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is False
    assert manifest["artifacts"] == {
        "context.json": "written",
        "stacks.txt": "failed",
    }
    assert manifest["files"] == ["context.json"]


def test_engine_diagnostic_atomic_replace_failure_does_not_close_owned_fd(
    tmp_path, monkeypatch
):
    close_calls = []
    real_close = os.close

    def record_close(fd):
        close_calls.append(fd)
        real_close(fd)

    def fail_replace(*args):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(dump_input.os, "close", record_close)
    monkeypatch.setattr(dump_input.os, "replace", fail_replace)

    with pytest.raises(OSError, match="read-only filesystem"):
        dump_input._write_private_text_atomic(
            tmp_path / "context.json",
            "{}",
            max_bytes=dump_input.ENGINE_DIAGNOSTIC_CONTEXT_MAX_BYTES,
        )

    assert close_calls == []
    assert not list(tmp_path.iterdir())


def test_engine_diagnostic_bundle_retention_is_bounded(tmp_path, monkeypatch):
    config = enable_engine_diagnostic_bundles(monkeypatch, tmp_path)
    monkeypatch.setattr(dump_input, "ENGINE_DIAGNOSTIC_MAX_BUNDLES", 2)

    for request_id in ("req-1", "req-2", "req-3"):
        assert (
            dump_input._dump_engine_execution_context(
                reason="exception",
                config=config,
                scheduler_output=SimpleNamespace(request_id=request_id),
                scheduler_stats=None,
            )
            is not None
        )

    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    bundles = list(dump_root.iterdir())
    assert len(bundles) == 2
    assert all((bundle / "manifest.json").is_file() for bundle in bundles)
    retained_outputs = {
        json.loads((bundle / "context.json").read_text(encoding="utf-8"))[
            "scheduler_output_text"
        ]
        for bundle in bundles
    }
    assert all("req-1" not in output for output in retained_outputs)
    assert any("req-2" in output for output in retained_outputs)
    assert any("req-3" in output for output in retained_outputs)


def test_engine_diagnostic_prune_removes_only_stale_incomplete_bundles(tmp_path):
    stale = tmp_path / f"{dump_input.ENGINE_DIAGNOSTIC_BUNDLE_PREFIX}stale"
    fresh = tmp_path / f"{dump_input.ENGINE_DIAGNOSTIC_BUNDLE_PREFIX}fresh"
    stale.mkdir()
    fresh.mkdir()
    expired = time.time() - dump_input.ENGINE_DIAGNOSTIC_INCOMPLETE_MAX_AGE_S - 1
    os.utime(stale, (expired, expired))

    dump_input._prune_engine_diagnostic_bundles(tmp_path)

    assert not stale.exists()
    assert fresh.is_dir()


def test_new_bundle_prunes_stale_incomplete_bundle_before_write(tmp_path, monkeypatch):
    config = enable_engine_diagnostic_bundles(monkeypatch, tmp_path)
    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    stale = dump_root / f"{dump_input.ENGINE_DIAGNOSTIC_BUNDLE_PREFIX}stale"
    stale.mkdir(parents=True)
    expired = time.time() - dump_input.ENGINE_DIAGNOSTIC_INCOMPLETE_MAX_AGE_S - 1
    os.utime(stale, (expired, expired))

    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dump_input, "_write_engine_diagnostic_json", fail_write)

    assert (
        dump_input._dump_engine_execution_context(
            reason="exception",
            config=config,
            scheduler_output=SimpleNamespace(request_id="req-1"),
            scheduler_stats=None,
        )
        is None
    )
    assert not list(dump_root.iterdir())


def test_engine_diagnostic_bundle_write_failure_leaves_no_partial_bundle(
    tmp_path, monkeypatch
):
    def fail_write(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(dump_input, "_write_engine_diagnostic_json", fail_write)

    assert (
        dump_input._write_engine_diagnostic_bundle(
            reason="exception",
            config=make_timeout_config(),
            dump_root=tmp_path,
            config_summary={},
            scheduler_output_summary=None,
            scheduler_queue_summary=None,
            scheduler_output_text=None,
            scheduler_stats_text=None,
            stage=None,
            timeout_s=None,
            error=None,
        )
        is None
    )
    assert not list(tmp_path.iterdir())


def test_engine_diagnostic_manifest_failure_removes_partial_bundle(
    tmp_path, monkeypatch
):
    config = enable_engine_diagnostic_bundles(monkeypatch, tmp_path)
    real_write = dump_input._write_engine_diagnostic_json

    def fail_manifest(path, value, *, max_bytes):
        if path.name == "manifest.json":
            raise OSError("disk full")
        return real_write(path, value, max_bytes=max_bytes)

    monkeypatch.setattr(dump_input, "_write_engine_diagnostic_json", fail_manifest)

    result = dump_input._dump_engine_execution_context(
        reason="exception",
        config=config,
        scheduler_output=SimpleNamespace(request_id="req-1"),
        scheduler_stats=None,
    )

    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    assert result is None
    assert not list(dump_root.iterdir())


def test_engine_exception_bundle_write_timeout_is_fail_open(tmp_path, monkeypatch):
    write_started = threading.Event()
    release_write = threading.Event()
    write_finished = threading.Event()

    def block_write(**kwargs):
        write_started.set()
        release_write.wait(timeout=1.0)
        write_finished.set()
        return None

    monkeypatch.setattr(dump_input, "_write_engine_diagnostic_bundle", block_write)
    monkeypatch.setattr(dump_input, "ENGINE_DIAGNOSTIC_WRITE_TIMEOUT_S", 0.01)

    try:
        result = dump_input._write_engine_diagnostic_bundle_with_timeout(
            reason="exception",
            config=make_timeout_config(),
            dump_root=tmp_path,
            config_summary={},
            scheduler_output_summary=None,
            scheduler_queue_summary=None,
            scheduler_output_text=None,
            scheduler_stats_text=None,
            stage=None,
            timeout_s=None,
            error=None,
        )
        assert not write_finished.is_set()
    finally:
        release_write.set()

    assert write_started.is_set()
    assert result is None
    assert write_finished.wait(timeout=1.0)


def test_engine_execution_timeout_watchdog_reuses_control_thread(monkeypatch):
    dumped_stages = []
    dump_completed = threading.Event()

    def record_dump(config, snapshot, timeout_s, stage):
        dumped_stages.append(stage)
        dump_completed.set()

    monkeypatch.setattr(dump_input, "dump_engine_execution_timeout", record_dump)
    watchdog = dump_input.EngineExecutionTimeoutWatchdog(
        config=make_timeout_config(),
        timeout_s=0.01,
    )
    watchdog.start()
    thread = watchdog._thread

    try:
        for stage in (
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            engine_core_module.SAMPLE_TOKENS_STAGE,
        ):
            dump_completed.clear()
            watchdog.arm(make_timeout_snapshot(), stage)
            assert dump_completed.wait(timeout=1.0)
            diagnostic_thread = watchdog._diagnostic_thread
            assert diagnostic_thread is not None
            diagnostic_thread.join(timeout=1.0)
            assert not diagnostic_thread.is_alive()
        assert watchdog._thread is thread
        assert dumped_stages == [
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            engine_core_module.SAMPLE_TOKENS_STAGE,
        ]
    finally:
        watchdog.stop()


def test_engine_execution_timeout_throttle_is_per_watchdog_and_stage():
    now_s = [100.0]

    def make_watchdog():
        return dump_input.EngineExecutionTimeoutWatchdog(
            config=make_timeout_config(),
            timeout_s=1.0,
            time_fn=lambda: now_s[0],
        )

    first_watchdog = make_watchdog()
    second_watchdog = make_watchdog()
    stage = engine_core_module.EXECUTE_MODEL_WAIT_STAGE

    assert first_watchdog._mark_dump_if_allowed(stage)
    assert not first_watchdog._mark_dump_if_allowed(stage)
    assert first_watchdog._mark_dump_if_allowed(
        engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    )
    assert second_watchdog._mark_dump_if_allowed(stage)

    now_s[0] += dump_input.ENGINE_EXECUTION_TIMEOUT_DUMP_THROTTLE_S
    assert first_watchdog._mark_dump_if_allowed(stage)


def test_engine_execution_timeout_context_balances_detail_and_privacy(monkeypatch):
    logs = []

    def record_log(message, *args):
        logs.append(message % args)

    monkeypatch.setattr(dump_input.logger, "error", record_log)
    dump_input._dump_engine_timeout_context(
        make_timeout_config(),
        make_timeout_snapshot(
            scheduler_state={"num_running_reqs": 1, "num_waiting_reqs": 0}
        ),
    )

    combined_logs = "\n".join(logs)
    assert "request-123" in combined_logs
    assert "request-456" in combined_logs
    assert "temperature" in combined_logs
    assert "0.25" in combined_logs
    assert "TestArchitecture" in combined_logs
    assert "private/model/path" not in combined_logs
    assert "private-extra-arg" not in combined_logs
    assert "private-schema" not in combined_logs
    assert "private-stop-string" not in combined_logs
    assert "[101, 102, 103]" not in combined_logs
    assert "num_scheduled_new_reqs" in combined_logs
    assert "num_running_reqs" in combined_logs


def test_engine_timeout_config_allowlists_match_real_configs(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["LlamaForCausalLM"],
                "hidden_size": 32,
                "intermediate_size": 64,
                "max_position_embeddings": 32,
                "model_type": "llama",
                "num_attention_heads": 4,
                "num_hidden_layers": 1,
                "vocab_size": 32,
            }
        ),
        encoding="utf-8",
    )
    config = VllmConfig(model_config=ModelConfig(model=str(tmp_path)))
    allowlists = (
        (config.model_config, dump_input._ENGINE_TIMEOUT_MODEL_CONFIG_FIELDS),
        (config.parallel_config, dump_input._ENGINE_TIMEOUT_PARALLEL_CONFIG_FIELDS),
        (config.scheduler_config, dump_input._ENGINE_TIMEOUT_SCHEDULER_CONFIG_FIELDS),
        (config.cache_config, dump_input._ENGINE_TIMEOUT_CACHE_CONFIG_FIELDS),
        (config.offload_config, dump_input._ENGINE_TIMEOUT_OFFLOAD_CONFIG_FIELDS),
        (
            config.offload_config.uva,
            dump_input._ENGINE_TIMEOUT_UVA_OFFLOAD_CONFIG_FIELDS,
        ),
    )

    for config_object, field_names in allowlists:
        assert not [name for name in field_names if not hasattr(config_object, name)]
    speculative_fields = {field.name for field in fields(SpeculativeConfig)}
    assert set(dump_input._ENGINE_TIMEOUT_SPECULATIVE_CONFIG_FIELDS).issubset(
        speculative_fields
    )

    summary = dump_input._make_engine_config_summary(config)
    assert "swap_space_bytes" not in summary["cache"]
    assert summary["offload"]["cpu_offload_gb"] == 0


def test_engine_execution_timeout_snapshot_uses_scheduler_dataclasses():
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[
            NewRequestData(
                req_id="new-request",
                prompt_token_ids=[1, 2, 3],
                mm_features=[],
                sampling_params=SamplingParams(max_tokens=8, temperature=0.25),
                pooling_params=None,
                block_ids=([1, 2],),
                num_computed_tokens=0,
                lora_request=None,
            )
        ],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=["cached-request"],
            resumed_req_ids={"cached-request"},
            new_token_ids=[[4]],
            all_token_ids={"cached-request": [1, 2, 3, 4]},
            new_block_ids=[([3],)],
            num_computed_tokens=[3],
            num_output_tokens=[1],
        ),
        num_scheduled_tokens={"new-request": 3, "cached-request": 1},
        total_num_scheduled_tokens=4,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[0],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
    )

    snapshot = dump_input.make_engine_execution_timeout_snapshot(
        scheduler_output,
        {"cached_request_sampling_params": {"cached-request": {"temperature": 0.5}}},
    )

    samples = snapshot.scheduler_output_summary["request_samples"]
    assert [sample["request_id"] for sample in samples] == [
        "new-request",
        "cached-request",
    ]
    assert samples[0]["num_prefill_tokens"] is None
    assert samples[0]["sampling_params"]["temperature"] == 0.25
    assert samples[1]["is_resumed"]
    assert samples[1]["num_all_tokens"] == 4
    assert samples[1]["sampling_params"] == {"temperature": 0.5}


def test_engine_execution_timeout_samples_include_cached_requests_when_truncated():
    scheduler_output = make_timeout_scheduler_output()
    template_request = scheduler_output.scheduled_new_reqs[0]
    scheduler_output.scheduled_new_reqs = []
    for index in range(dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT):
        request_fields = vars(template_request).copy()
        request_fields["req_id"] = f"new-request-{index}"
        scheduler_output.scheduled_new_reqs.append(SimpleNamespace(**request_fields))

    samples = dump_input._make_request_samples(scheduler_output)

    assert len(samples) == dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT
    assert samples[-1]["request_kind"] == "cached"
    assert samples[-1]["request_id"] == "request-456"


def test_engine_execution_timeout_bounds_oversized_request_ids(monkeypatch):
    logs = []
    scheduler_output = make_timeout_scheduler_output()
    oversized_request_id = "tenant-secret-" + "x" * 10_000
    scheduler_output.scheduled_new_reqs[0].req_id = oversized_request_id

    monkeypatch.setattr(
        dump_input.logger,
        "error",
        lambda message, *args: logs.append(message % args),
    )
    dump_input._dump_engine_timeout_context(
        make_timeout_config(), make_timeout_snapshot(scheduler_output)
    )

    combined_logs = "\n".join(logs)
    assert oversized_request_id not in combined_logs
    assert '"request_id_truncated": true' in combined_logs
    assert '"request_id_length": 10014' in combined_logs
    assert '"request_id_sha256"' in combined_logs
    assert len(logs[0]) <= dump_input.ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS + 100


def test_engine_execution_timeout_serialized_summary_has_hard_limit():
    serialized = dump_input._serialize_diagnostic({"value": "x" * 100_000})
    payload = json.loads(serialized)

    assert len(serialized) <= dump_input.ENGINE_EXECUTION_TIMEOUT_SUMMARY_MAX_CHARS
    assert payload["diagnostic_output_truncated"] is True
    assert payload["original_length"] > len(serialized)
    assert len(payload["sha256"]) == 64
    assert payload["diagnostic_prefix"]


def test_engine_execution_timeout_logs_request_sample_truncation(monkeypatch):
    logs = []
    scheduler_output = make_timeout_scheduler_output()
    template_request = scheduler_output.scheduled_new_reqs[0]
    scheduler_output.scheduled_new_reqs = []
    for index in range(dump_input.ENGINE_EXECUTION_TIMEOUT_REQUEST_SAMPLE_LIMIT + 1):
        request_fields = vars(template_request).copy()
        request_fields["req_id"] = f"new-request-{index}"
        scheduler_output.scheduled_new_reqs.append(SimpleNamespace(**request_fields))

    monkeypatch.setattr(
        dump_input.logger,
        "error",
        lambda message, *args: logs.append(message % args),
    )
    dump_input._dump_engine_timeout_context(
        make_timeout_config(), make_timeout_snapshot(scheduler_output)
    )

    output_summary = logs[0]
    assert '"request_samples_truncated": true' in output_summary
    assert "new-request-0" in output_summary
    assert "new-request-20" in output_summary
    assert "new-request-19" not in output_summary


def test_engine_execution_timeout_spreads_samples_across_cached_requests():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    cached_requests = scheduler_output.scheduled_cached_reqs
    cached_requests.req_ids = [f"cached-request-{index}" for index in range(100)]
    cached_requests.num_reqs = len(cached_requests.req_ids)
    cached_requests.all_token_ids = {
        request_id: [1] for request_id in cached_requests.req_ids
    }
    cached_requests.new_block_ids = [([],)] * cached_requests.num_reqs
    cached_requests.new_token_ids = [[]] * cached_requests.num_reqs
    cached_requests.num_computed_tokens = [1] * cached_requests.num_reqs
    cached_requests.num_output_tokens = [1] * cached_requests.num_reqs
    cached_requests.resumed_req_ids = set()

    samples = dump_input._make_request_samples(scheduler_output)

    assert [sample["request_id"] for sample in samples] == [
        "cached-request-0",
        "cached-request-5",
        "cached-request-10",
        "cached-request-15",
        "cached-request-20",
        "cached-request-26",
        "cached-request-31",
        "cached-request-36",
        "cached-request-41",
        "cached-request-46",
        "cached-request-52",
        "cached-request-57",
        "cached-request-62",
        "cached-request-67",
        "cached-request-72",
        "cached-request-78",
        "cached-request-83",
        "cached-request-88",
        "cached-request-93",
        "cached-request-99",
    ]


def test_engine_execution_timeout_cached_sample_includes_sampling_params():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    samples = dump_input._make_request_samples(
        scheduler_output,
        cached_sampling_params={"request-456": {"temperature": 0.25, "top_p": 0.9}},
    )

    assert samples[0]["request_kind"] == "cached"
    assert samples[0]["sampling_params"] == {
        "temperature": 0.25,
        "top_p": 0.9,
    }


def test_engine_execution_timeout_cached_sample_marks_unreported_tokens_unknown():
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs = []
    scheduler_output.scheduled_cached_reqs.all_token_ids = {}

    samples = dump_input._make_request_samples(scheduler_output)

    assert samples[0]["num_all_tokens"] is None


def test_dump_on_slow_execution_arms_and_disarms_watchdog():
    calls: list[Any] = []
    snapshot = make_timeout_snapshot(scheduler_state={"num_running_reqs": 2})

    class FakeWatchdog:
        enabled = True

        def arm(self, armed_snapshot, stage):
            calls.append(("arm", armed_snapshot, stage))
            return 123

        def disarm(self, generation):
            calls.append(("disarm", generation))

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
    )
    with EngineCore.dump_on_slow_execution(
        engine,
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        snapshot,
    ):
        calls.append("body")

    assert calls == [
        (
            "arm",
            snapshot,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        ),
        "body",
        ("disarm", 123),
    ]


def test_dump_on_slow_execution_disarms_watchdog_after_exception():
    calls: list[Any] = []
    snapshot = make_timeout_snapshot(scheduler_state={"num_running_reqs": 2})

    class FakeWatchdog:
        enabled = True

        def arm(self, armed_snapshot, stage):
            calls.append(("arm", armed_snapshot, stage))
            return 123

        def disarm(self, generation):
            calls.append(("disarm", generation))

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
    )
    try:
        with EngineCore.dump_on_slow_execution(
            engine,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            snapshot,
        ):
            raise RuntimeError("stage failed")
    except RuntimeError as err:
        assert str(err) == "stage failed"
    else:
        raise AssertionError("expected stage failure")

    assert calls == [
        (
            "arm",
            snapshot,
            engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        ),
        ("disarm", 123),
    ]


def test_dump_on_slow_execution_disabled_skips_snapshot():
    calls = []

    class FakeWatchdog:
        enabled = False

        def arm(self, snapshot, stage):
            calls.append(("arm", snapshot, stage))
            return None

        def disarm(self, generation):
            calls.append(("disarm", generation))

    def fail_snapshot(_output):
        raise AssertionError("disabled watchdog must not collect a snapshot")

    engine = SimpleNamespace(
        execution_timeout_watchdog=FakeWatchdog(),
        _make_scheduler_timeout_state=fail_snapshot,
    )

    scheduler_state = EngineCore._prepare_timeout_diagnostic_state(
        engine, SimpleNamespace()
    )
    with EngineCore.dump_on_slow_execution(
        engine,
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_state,
    ):
        pass

    assert scheduler_state.scheduler_output_summary == {}
    assert calls == []


def test_prepare_timeout_diagnostic_state_fails_open(monkeypatch):
    def fail_snapshot(*args):
        raise RuntimeError("snapshot failed")

    monkeypatch.setattr(
        engine_core_module,
        "make_engine_execution_timeout_snapshot",
        fail_snapshot,
    )
    engine = SimpleNamespace(
        execution_timeout_watchdog=SimpleNamespace(enabled=True),
        _make_scheduler_timeout_state=lambda scheduler_output: {},
    )

    snapshot = EngineCore._prepare_timeout_diagnostic_state(
        engine, make_timeout_scheduler_output()
    )

    assert snapshot.scheduler_output_summary == {
        "scheduler_output_summary_unavailable": True
    }
    assert snapshot.scheduler_queue_summary == {}


def test_engine_shutdown_stops_watchdog_before_teardown(monkeypatch):
    teardown_order = []
    monkeypatch.setattr(
        engine_core_module.gc,
        "unfreeze",
        lambda: teardown_order.append("gc"),
    )
    monkeypatch.setattr(
        engine_core_module,
        "cleanup_dist_env_and_memory",
        lambda: teardown_order.append("distributed"),
    )
    engine = SimpleNamespace(
        execution_timeout_watchdog=SimpleNamespace(
            stop=lambda: teardown_order.append("watchdog")
        ),
        structured_output_manager=SimpleNamespace(
            clear_backend=lambda: teardown_order.append("structured_output")
        ),
        model_executor=SimpleNamespace(
            shutdown=lambda: teardown_order.append("executor")
        ),
        scheduler=SimpleNamespace(shutdown=lambda: teardown_order.append("scheduler")),
    )

    EngineCore.shutdown(engine)

    assert teardown_order == [
        "watchdog",
        "structured_output",
        "executor",
        "scheduler",
        "gc",
        "distributed",
    ]


def test_make_scheduler_timeout_state_is_rich_and_non_mutating():
    sampling_params = (
        make_timeout_scheduler_output().scheduled_new_reqs[0].sampling_params
    )
    scheduler = SimpleNamespace(
        make_timeout_diagnostic_state=lambda: {
            "kv_cache_usage": 0.75,
            "num_running_reqs": 2,
            "num_skipped_waiting_reqs": 1,
            "num_waiting_reqs": 3,
        },
    )
    scheduler_output = make_timeout_scheduler_output()
    scheduler_output.scheduled_new_reqs[0].req_id = "request-456"
    scheduler_output.scheduled_new_reqs[0].sampling_params = sampling_params
    engine = SimpleNamespace(
        scheduler=scheduler,
        _timeout_sampling_params_by_request={},
    )

    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state["kv_cache_usage"] == 0.75
    assert state["num_running_reqs"] == 2
    assert state["num_skipped_waiting_reqs"] == 1
    assert state["num_waiting_reqs"] == 3
    assert state["cached_request_sampling_params"]["request-456"]["temperature"] == 0.25

    scheduler_output.scheduled_new_reqs = []
    scheduler_output.finished_req_ids = {"request-456"}
    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state["cached_request_sampling_params"] == {}
    assert engine._timeout_sampling_params_by_request == {}


def test_make_scheduler_timeout_state_refreshes_rescheduled_request(monkeypatch):
    scheduler_output = make_timeout_scheduler_output()
    scheduler = SimpleNamespace(make_timeout_diagnostic_state=lambda: {})
    engine = SimpleNamespace(
        scheduler=scheduler,
        _timeout_sampling_params_by_request={},
    )
    calls = []

    def record_summary(sampling_params):
        calls.append(sampling_params.temperature)
        return {"temperature": sampling_params.temperature}

    monkeypatch.setattr(
        engine_core_module, "make_sampling_params_summary", record_summary
    )

    EngineCore._make_scheduler_timeout_state(engine, scheduler_output)
    scheduler_output.scheduled_new_reqs[0].sampling_params.temperature = 0.75
    EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert calls == [0.25, 0.75]
    assert (
        engine._timeout_sampling_params_by_request["request-123"]["temperature"] == 0.75
    )


def test_make_scheduler_timeout_state_failure_still_refreshes_sampling_cache():
    def fail_snapshot():
        raise RuntimeError("snapshot failed")

    scheduler_output = make_timeout_scheduler_output()
    request = scheduler_output.scheduled_new_reqs[0]
    request.req_id = "reused-request-id"
    scheduler_output.finished_req_ids = {request.req_id}
    engine = SimpleNamespace(
        scheduler=SimpleNamespace(make_timeout_diagnostic_state=fail_snapshot),
        _timeout_sampling_params_by_request={
            request.req_id: {"temperature": 0.99},
            "request-456": {"temperature": 0.5},
        },
    )

    state = EngineCore._make_scheduler_timeout_state(engine, scheduler_output)

    assert state == {
        "cached_request_sampling_params": {"request-456": {"temperature": 0.5}}
    }
    assert (
        engine._timeout_sampling_params_by_request[request.req_id]["temperature"]
        == 0.25
    )


def test_scheduler_timeout_diagnostic_state_includes_resource_pressure():
    scheduler = SimpleNamespace(
        get_kv_cache_usage=lambda: 0.75,
        running=[1, 2],
        skipped_waiting=[1],
        waiting=[1, 2, 3],
    )

    assert Scheduler.make_timeout_diagnostic_state(scheduler) == {
        "kv_cache_usage": 0.75,
        "num_running_reqs": 2,
        "num_skipped_waiting_reqs": 1,
        "num_waiting_reqs": 3,
    }


def test_scheduler_timeout_diagnostic_state_keeps_custom_schedulers_compatible():
    scheduler = SimpleNamespace(get_request_counts=lambda: (2, 3))

    assert "make_timeout_diagnostic_state" not in SchedulerInterface.__abstractmethods__
    assert SchedulerInterface.make_timeout_diagnostic_state(scheduler) == {
        "num_running_reqs": 2,
        "num_waiting_reqs": 3,
    }


def test_step_records_execute_and_sync_sample_timeout_stages():
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_output = SimpleNamespace()
    model_executor = FakeStageModelExecutor(
        execute_result=None, sample_result=model_output
    )
    engine = FakeStageEngine(scheduler, model_executor)

    outputs, model_executed = EngineCore.step(engine)

    assert outputs == {}
    assert model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert model_executor.sample_calls == [("grammar", False)]
    assert engine.stages == [
        engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]
    assert engine.events == [
        ("call", "execute_model"),
        ("enter", engine_core_module.EXECUTE_MODEL_WAIT_STAGE),
        ("call", "execute_model.result"),
        ("exit", engine_core_module.EXECUTE_MODEL_WAIT_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert all(state is engine.stage_states[0] for state in engine.stage_states)


def test_step_with_batch_queue_enqueues_sample_tokens_wait_stage():
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=False,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_executor = FakeStageModelExecutor(
        execute_result=SimpleNamespace(), sample_result=SimpleNamespace()
    )
    engine = FakeStageEngine(scheduler, model_executor)
    engine.batch_queue = deque(maxlen=engine.batch_queue_size)

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs is None
    assert model_executed
    assert len(engine.batch_queue) == 1
    (
        future,
        queued_scheduler_output,
        exec_future,
        future_stage,
        timeout_state,
    ) = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert timeout_state is engine.stage_states[0]
    assert model_executor.sample_calls == [("grammar", True)]
    assert engine.stages == [engine_core_module.SAMPLE_TOKENS_STAGE]
    assert engine.events == [
        ("call", "execute_model"),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert all(state is timeout_state for state in engine.stage_states)


def test_step_with_batch_queue_reuses_snapshot_for_deferred_sampling():
    old_scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler_output = SimpleNamespace(
        total_num_scheduled_tokens=1,
        pending_structured_output_tokens=True,
    )
    scheduler = FakeStageScheduler(scheduler_output)
    model_executor = FakeStageModelExecutor(
        execute_result=SimpleNamespace(), sample_result=SimpleNamespace()
    )
    engine = FakeStageEngine(scheduler, model_executor)
    old_timeout_state = dump_input.EngineExecutionTimeoutSnapshot(
        {"snapshot": "old"}, {}
    )
    old_model_output = SimpleNamespace()
    engine.batch_queue = deque(
        [
            (
                FakeFuture(
                    old_model_output,
                    model_executor.events,
                    "old_sample_tokens.result",
                ),
                old_scheduler_output,
                FakeFuture(
                    SimpleNamespace(),
                    model_executor.events,
                    "old_execute_model.result",
                ),
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
                old_timeout_state,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert model_executed
    assert scheduler.updated_with == (old_scheduler_output, old_model_output)
    assert engine.prepared_timeout_outputs == [scheduler_output]
    assert engine.stages == [
        engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
        engine_core_module.SAMPLE_TOKENS_STAGE,
    ]
    scheduled_timeout_state = engine.stage_states[1]
    assert engine.stage_states == [
        old_timeout_state,
        scheduled_timeout_state,
    ]
    assert engine.events == [
        ("call", "execute_model"),
        ("enter", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("call", "old_sample_tokens.result"),
        ("exit", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("enter", engine_core_module.SAMPLE_TOKENS_STAGE),
        ("call", "sample_tokens"),
        ("exit", engine_core_module.SAMPLE_TOKENS_STAGE),
    ]
    assert len(engine.batch_queue) == 1
    (
        future,
        queued_scheduler_output,
        exec_future,
        future_stage,
        timeout_state,
    ) = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert timeout_state is scheduled_timeout_state


def test_step_with_batch_queue_uses_queued_future_stage():
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler = FakeStageScheduler(scheduler_output, has_requests=False)
    model_output = SimpleNamespace()
    engine = FakeStageEngine(scheduler)
    exec_future = FakeFuture(SimpleNamespace(), engine.events, "execute_model.result")
    timeout_state = dump_input.EngineExecutionTimeoutSnapshot({"snapshot": 1}, {})
    engine.batch_queue = deque(
        [
            (
                FakeFuture(model_output, engine.events, "sample_tokens.result"),
                scheduler_output,
                exec_future,
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
                timeout_state,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert not model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert engine.stages == [engine_core_module.SAMPLE_TOKENS_WAIT_STAGE]
    assert engine.stage_states == [timeout_state]
    assert engine.events == [
        ("enter", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
        ("call", "sample_tokens.result"),
        ("exit", engine_core_module.SAMPLE_TOKENS_WAIT_STAGE),
    ]


def test_parse_stack_trace_signal_accepts_names_and_numbers(monkeypatch):
    monkeypatch.setattr(dump_input.signal, "SIGUSR1", 10, raising=False)

    assert dump_input._parse_stack_trace_signal("SIGUSR1") == 10
    assert dump_input._parse_stack_trace_signal("usr1") == 10
    assert dump_input._parse_stack_trace_signal("10") == 10
    assert dump_input._parse_stack_trace_signal("") is None
    assert dump_input._parse_stack_trace_signal("SIG_DOES_NOT_EXIST") is None


def test_install_stack_trace_signal_handler_registers_once(monkeypatch):
    registered: list[dict[str, Any]] = []

    def record_register(signum: int, **kwargs: Any) -> None:
        registered.append(
            {
                "all_threads": kwargs["all_threads"],
                "chain": kwargs["chain"],
                "file": kwargs["file"],
                "signum": signum,
            }
        )

    monkeypatch.setattr(dump_input.signal, "SIGUSR1", 10, raising=False)
    monkeypatch.setattr(
        dump_input.signal, "getsignal", lambda _signum: dump_input.signal.SIG_DFL
    )
    monkeypatch.setattr(
        dump_input.envs,
        "VLLM_DEBUG_STACK_TRACE_SIGNAL",
        "SIGUSR1",
        raising=False,
    )
    monkeypatch.setattr(dump_input.faulthandler, "register", record_register)

    assert dump_input.install_stack_trace_signal_handler("EngineCore")
    assert dump_input.install_stack_trace_signal_handler("Worker_0")
    assert registered == [
        {
            "all_threads": True,
            "chain": False,
            "file": dump_input.sys.stderr,
            "signum": 10,
        }
    ]


def test_install_stack_trace_signal_handler_rejects_unsafe_signal(monkeypatch):
    registered: list[tuple[Any, ...]] = []

    def record_unexpected_register(*args: Any, **_kwargs: Any) -> None:
        registered.append(args)

    monkeypatch.setattr(
        dump_input.envs,
        "VLLM_DEBUG_STACK_TRACE_SIGNAL",
        "SIGTERM",
        raising=False,
    )
    monkeypatch.setattr(
        dump_input.faulthandler,
        "register",
        record_unexpected_register,
    )

    assert not dump_input.install_stack_trace_signal_handler("EngineCore")
    assert not registered


def test_install_stack_trace_signal_handler_preserves_existing_handler(monkeypatch):
    registered: list[tuple[Any, ...]] = []

    monkeypatch.setattr(dump_input.signal, "SIGUSR1", 10, raising=False)
    monkeypatch.setattr(dump_input.signal, "getsignal", lambda _signum: object())
    monkeypatch.setattr(
        dump_input.envs,
        "VLLM_DEBUG_STACK_TRACE_SIGNAL",
        "SIGUSR1",
        raising=False,
    )
    monkeypatch.setattr(
        dump_input.faulthandler,
        "register",
        lambda *args, **_kwargs: registered.append(args),
    )

    assert not dump_input.install_stack_trace_signal_handler("EngineCore")
    assert not registered


@pytest.mark.skipif(
    not hasattr(os, "fork") or not hasattr(signal, "SIGUSR1"),
    reason="requires fork and SIGUSR1",
)
def test_install_stack_trace_signal_handler_reregisters_after_fork():
    script = """
import os
import signal
from vllm.logging_utils import dump_input

dump_input.envs.VLLM_DEBUG_STACK_TRACE_SIGNAL = "SIGUSR1"
assert dump_input.install_stack_trace_signal_handler("APIServer")
child_pid = os.fork()
if child_pid == 0:
    registrations = []
    dump_input.faulthandler.register = (
        lambda signum, **_kwargs: registrations.append(signum)
    )
    try:
        assert dump_input.install_stack_trace_signal_handler("EngineCore")
        assert registrations == [signal.SIGUSR1]
    except BaseException:
        os._exit(1)
    os._exit(0)

_, child_status = os.waitpid(child_pid, 0)
assert os.waitstatus_to_exitcode(child_status) == 0
"""

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(not hasattr(signal, "SIGUSR1"), reason="requires SIGUSR1")
def test_stack_trace_signal_handler_dumps_real_subprocess():
    script = """
import os
import signal
from vllm.logging_utils.dump_input import install_stack_trace_signal_handler

assert install_stack_trace_signal_handler("test-process")
os.kill(os.getpid(), signal.SIGUSR1)
print("signal handled")
"""
    child_env = os.environ.copy()
    child_env["VLLM_DEBUG_STACK_TRACE_SIGNAL"] = "SIGUSR1"

    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[3],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "signal handled" in result.stdout
    assert "Current thread" in result.stderr
    assert 'File "<string>"' in result.stderr


def test_worker_init_installs_stack_trace_signal_handler(monkeypatch):
    calls: list[str] = []

    def record_install(process_name: str) -> bool:
        calls.append(process_name)
        return True

    import vllm.plugins as plugins

    monkeypatch.setattr(
        dump_input,
        "install_stack_trace_signal_handler",
        record_install,
    )
    monkeypatch.setattr(
        plugins,
        "load_general_plugins",
        lambda: calls.append("plugins_loaded"),
    )

    config = SimpleNamespace(
        enable_trace_function_call_for_thread=lambda: None,
        model_config=SimpleNamespace(multimodal_config=None),
        parallel_config=SimpleNamespace(
            worker_cls=f"{__name__}.FakeDiagnosticWorker",
            worker_extension_cls=None,
        ),
    )
    wrapper = WorkerWrapperBase(rpc_rank=0, global_rank=3)

    wrapper.init_worker(all_kwargs=[{"vllm_config": config}])

    assert calls == ["plugins_loaded", "Worker_3"]
    assert isinstance(wrapper.worker, FakeDiagnosticWorker)


def test_api_server_init_installs_stack_trace_signal_handler(monkeypatch):
    from vllm.entrypoints.launchers.api_server import entry as api_entry
    from vllm.v1.engine.async_llm import AsyncLLM

    calls: list[str] = []
    fake_async_llm = SimpleNamespace(
        reset_mm_cache=lambda: None,
        shutdown=lambda timeout: None,
    )

    async def reset_mm_cache() -> None:
        return None

    fake_async_llm.reset_mm_cache = reset_mm_cache
    monkeypatch.setattr(
        dump_input,
        "install_stack_trace_signal_handler",
        lambda process_name: calls.append(process_name),
    )
    monkeypatch.setattr(
        AsyncLLM,
        "from_vllm_config",
        lambda **_kwargs: fake_async_llm,
    )
    engine_args = SimpleNamespace(
        create_engine_config=lambda **_kwargs: SimpleNamespace(shutdown_timeout=0),
        enable_log_requests=False,
        aggregate_engine_logging=False,
        disable_log_stats=True,
    )

    async def exercise() -> None:
        async with api_entry.build_async_engine_client_from_engine_args(
            engine_args,
            client_config={"client_count": 2, "client_index": 1},
        ) as engine:
            assert engine is fake_async_llm

    asyncio.run(exercise())

    assert calls == ["APIServer_1"]


def test_engine_core_init_installs_stack_trace_signal_handler(monkeypatch):
    calls: list[str] = []
    run_engine_core = engine_core_module.EngineCoreProc.run_engine_core
    monkeypatch.setattr(
        engine_core_module,
        "install_stack_trace_signal_handler",
        calls.append,
    )
    monkeypatch.setattr(
        engine_core_module,
        "maybe_register_config_serialize_by_value",
        lambda: None,
    )
    monkeypatch.setattr(engine_core_module, "set_process_title", lambda _title: None)
    monkeypatch.setattr(
        engine_core_module,
        "maybe_init_worker_tracer",
        lambda *_args: None,
    )
    monkeypatch.setattr(engine_core_module, "decorate_logs", lambda: None)
    monkeypatch.setattr(engine_core_module.signal, "signal", lambda *_args: None)
    monkeypatch.setattr(
        engine_core_module,
        "EngineCoreProc",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("stop")),
    )
    parallel_config = SimpleNamespace(
        data_parallel_size=1,
        data_parallel_rank_local=0,
        numa_bind=False,
        data_parallel_index=0,
        reconfigure_for_independent_dp_rank=lambda: None,
    )
    config = SimpleNamespace(
        parallel_config=parallel_config,
        kv_transfer_config=None,
        model_config=SimpleNamespace(is_moe=False),
    )

    with pytest.raises(RuntimeError, match="stop"):
        run_engine_core(vllm_config=config)

    assert calls == ["EngineCore"]


def test_ray_engine_core_actor_installs_stack_trace_signal_handler(monkeypatch):
    calls: list[str] = []
    actor = object.__new__(engine_core_module.EngineCoreActor)
    monkeypatch.setattr(
        engine_core_module,
        "install_stack_trace_signal_handler",
        calls.append,
    )
    monkeypatch.setattr(
        engine_core_module,
        "maybe_init_worker_tracer",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        engine_core_module.EngineCoreActorMixin,
        "_set_nixl_side_channel_host",
        lambda _self: None,
    )
    monkeypatch.setattr(
        engine_core_module.EngineCoreActorMixin,
        "_set_visible_devices",
        lambda _self, _config, _local_dp_rank: None,
    )
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_index=0,
            data_parallel_rank_local=0,
        )
    )

    engine_core_module.EngineCoreActorMixin.__init__(
        actor,
        config,
        addresses=SimpleNamespace(),
        dp_rank=2,
        local_dp_rank=0,
    )

    assert calls == ["EngineCoreActor_DP2"]


def test_engine_no_progress_dump_writes_diagnostic_bundle(tmp_path, monkeypatch):
    traceback_destinations: list[str] = []

    def record_traceback(file, all_threads):
        assert all_threads
        traceback_destinations.append("stderr" if file is sys.stderr else "bundle")
        try:
            file.write("stack dump\n")
        except TypeError:
            file.write(b"stack dump\n")

    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)

    dump_input.dump_engine_no_progress(
        config=enable_engine_diagnostic_bundles(monkeypatch, tmp_path),
        timeout_s=3.0,
        progress_snapshot={"stage": "engine_step", "progress_index": 1},
        scheduler_snapshot={"requests": {"running": {"count": 1}}},
    )

    dump_root = tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    bundles = list(dump_root.iterdir())
    assert len(bundles) == 1
    assert traceback_destinations == ["stderr", "bundle"]
    assert (bundles[0] / "stacks.txt").read_text(encoding="utf-8") == "stack dump\n"

    context = json.loads((bundles[0] / "context.json").read_text(encoding="utf-8"))
    assert context["reason"] == "no_progress"
    assert context["stage"] == dump_input.ENGINE_NO_PROGRESS_STAGE
    assert context["timeout_s"] == 3.0
    assert context["scheduler_output_text"] is None
    assert context["scheduler_stats_text"] is None
    assert context["scheduler_snapshot_file"] == "scheduler_snapshot.json"

    snapshot = json.loads(
        (bundles[0] / "scheduler_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["engine_progress"]["stage"] == "engine_step"
    assert snapshot["scheduler"]["requests"]["running"]["count"] == 1

    manifest = json.loads((bundles[0] / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"] is True
    assert manifest["artifacts"] == {
        "context.json": "written",
        "scheduler_snapshot.json": "written",
        "stacks.txt": "written",
    }


def test_progress_monitor_dumps_when_work_stalls(monkeypatch):
    now_s = 100.0
    emitted: list[tuple[Any, ...]] = []
    dispatched: list[tuple[Any, ...]] = []
    scheduler_snapshot = {"requests": {"waiting": {"count": 1}}}
    scheduler_snapshot_calls = 0

    def current_time() -> float:
        return now_s

    def make_scheduler_snapshot() -> dict[str, Any]:
        nonlocal scheduler_snapshot_calls
        scheduler_snapshot_calls += 1
        return scheduler_snapshot

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=2.0,
        has_work_fn=lambda: True,
        scheduler_snapshot_fn=make_scheduler_snapshot,
        time_fn=current_time,
    )
    monitor.record_progress("engine_step")
    now_s = 102.5
    monkeypatch.setattr(
        dump_input,
        "_emit_engine_no_progress",
        lambda *args: emitted.append(args),
    )
    monkeypatch.setattr(
        monitor,
        "_dispatch_no_progress_dump",
        lambda *args: dispatched.append(args) or True,
    )

    assert monitor.maybe_dump_no_progress()
    assert emitted[0][0] == 2.0
    assert emitted[0][1]["stage"] == "engine_step"
    assert emitted[0][1]["elapsed_since_progress_s"] == 2.5
    assert dispatched == [emitted[0]]
    assert scheduler_snapshot_calls == 0


def test_progress_monitor_does_not_dump_when_idle(monkeypatch):
    now_s = 100.0
    dumps: list[tuple[Any, ...]] = []

    def current_time() -> float:
        return now_s

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=1.0,
        has_work_fn=lambda: False,
        scheduler_snapshot_fn=lambda: None,
        time_fn=current_time,
    )
    monitor.record_progress("engine_step")
    now_s = 101.5
    monkeypatch.setattr(
        dump_input,
        "_emit_engine_no_progress",
        lambda *args: dumps.append(args),
    )

    assert not monitor.maybe_dump_no_progress()
    assert not dumps
    assert monitor.snapshot()["stage"] == "idle"


def test_progress_monitor_dumps_for_active_operation_without_scheduler_work(
    monkeypatch,
):
    now_s = 100.0
    emitted: list[tuple[Any, ...]] = []
    dispatched: list[tuple[Any, ...]] = []

    def fail_has_work() -> bool:
        raise AssertionError("active operation queried scheduler work")

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=1.0,
        has_work_fn=fail_has_work,
        scheduler_snapshot_fn=lambda: None,
        time_fn=lambda: now_s,
    )
    monitor.record_activity("client_request_utility")
    now_s = 101.5
    monkeypatch.setattr(
        dump_input,
        "_emit_engine_no_progress",
        lambda *args: emitted.append(args),
    )
    monkeypatch.setattr(
        monitor,
        "_dispatch_no_progress_dump",
        lambda *args: dispatched.append(args) or True,
    )

    assert monitor.maybe_dump_no_progress()
    assert emitted[0][1]["operation_active"] is True
    assert emitted[0][1]["stage"] == "client_request_utility"
    assert dispatched == [emitted[0]]


def test_progress_monitor_start_failure_is_fail_open(monkeypatch):
    class FailingThread:
        def start(self) -> None:
            raise RuntimeError("thread startup failed")

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=1.0,
        has_work_fn=lambda: True,
        scheduler_snapshot_fn=lambda: None,
    )
    monkeypatch.setattr(monitor, "_create_thread", FailingThread)

    monitor.start()

    assert not monitor.enabled
    assert monitor._thread is None


def test_progress_monitor_thread_survives_check_failure(monkeypatch):
    recovered = threading.Event()
    calls = 0
    logged_errors: list[tuple[Any, ...]] = []
    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=0.01,
        has_work_fn=lambda: True,
        scheduler_snapshot_fn=lambda: None,
    )

    def flaky_check() -> bool:
        nonlocal calls
        calls += 1
        if calls <= 2:
            raise RuntimeError("diagnostic failure")
        recovered.set()
        return False

    monkeypatch.setattr(monitor, "maybe_dump_no_progress", flaky_check)
    monkeypatch.setattr(
        dump_input.logger,
        "exception",
        lambda *args, **_kwargs: logged_errors.append(args),
    )
    monitor.start()
    try:
        assert recovered.wait(1.0)
    finally:
        monitor.stop()

    assert calls >= 2
    assert len(logged_errors) == 1
    assert monitor._thread is not None
    assert not monitor._thread.is_alive()


def test_progress_monitor_diagnostic_writer_is_nonblocking_and_single_flight(
    monkeypatch,
):
    now_s = 100.0
    snapshot_started = threading.Event()
    release_snapshot = threading.Event()
    check_finished = threading.Event()
    stop_finished = threading.Event()
    detailed_dumped = threading.Event()
    check_results: list[bool] = []
    diagnostic_events: list[str] = []

    def blocking_snapshot() -> dict[str, Any]:
        diagnostic_events.append("scheduler_snapshot")
        snapshot_started.set()
        assert release_snapshot.wait(timeout=2.0)
        return {"requests": {"running": {"count": 1}}}

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=1.0,
        has_work_fn=lambda: True,
        scheduler_snapshot_fn=blocking_snapshot,
        time_fn=lambda: now_s,
    )
    monitor.record_progress("engine_step")
    now_s = 101.5
    monkeypatch.setattr(
        dump_input,
        "_emit_engine_no_progress",
        lambda *_args: diagnostic_events.append("stderr"),
    )
    monkeypatch.setattr(
        dump_input,
        "dump_engine_no_progress",
        lambda *_args, **_kwargs: detailed_dumped.set(),
    )
    monkeypatch.setattr(
        dump_input,
        "ENGINE_NO_PROGRESS_MONITOR_STOP_TIMEOUT_S",
        0.01,
    )

    def check_no_progress() -> None:
        check_results.append(monitor.maybe_dump_no_progress())
        check_finished.set()

    check_thread = threading.Thread(target=check_no_progress)
    check_thread.start()
    try:
        assert check_finished.wait(timeout=1.0)
        check_thread.join(timeout=1.0)
        assert check_results == [True]
        assert snapshot_started.wait(timeout=1.0)
        assert diagnostic_events[:2] == ["stderr", "scheduler_snapshot"]
        diagnostic_thread = monitor._diagnostic_thread
        assert diagnostic_thread is not None
        assert diagnostic_thread.is_alive()
        assert not monitor._dispatch_no_progress_dump(1.0, {"stage": "engine_step"})

        stop_thread = threading.Thread(
            target=lambda: (monitor.stop(), stop_finished.set())
        )
        stop_thread.start()
        assert stop_finished.wait(timeout=1.0)
        stop_thread.join(timeout=1.0)
        assert diagnostic_thread.is_alive()
    finally:
        release_snapshot.set()
        check_thread.join(timeout=1.0)
        monitor.stop()

    diagnostic_thread.join(timeout=1.0)
    assert not diagnostic_thread.is_alive()
    assert detailed_dumped.is_set()


def test_process_engine_step_progress_requires_model_or_outputs(monkeypatch):
    progress_monitor = FakeProgressMonitor()
    queued_outputs: list[tuple[int, Any]] = []
    sleep_calls: list[float] = []
    engine = SimpleNamespace(
        progress_monitor=progress_monitor,
        output_queue=SimpleNamespace(put_nowait=queued_outputs.append),
        post_step=lambda model_executed: None,
        scheduler=SimpleNamespace(has_requests=lambda: True),
        step_fn=lambda: ({}, False),
    )

    monkeypatch.setattr(engine_core_module.time, "sleep", sleep_calls.append)

    assert not engine_core_module.EngineCoreProc._process_engine_step(engine)
    assert progress_monitor.activities == [("engine_step", None)]
    assert progress_monitor.progress == []
    assert sleep_calls == [0.001]


def test_dp_housekeeping_does_not_mask_stalled_local_request():
    progress_monitor = FakeProgressMonitor()
    loop_conditions = iter((True, False))

    @contextmanager
    def capture_iteration_details(_scheduler_output):
        yield None

    engine = SimpleNamespace(
        enable_fault_tolerance=False,
        _handle_shutdown=lambda: next(loop_conditions),
        engines_running=True,
        _process_input_queue=lambda: None,
        _maybe_publish_request_counts=lambda: None,
        eep_scaling_state=None,
        _process_engine_step=lambda: False,
        scheduler=SimpleNamespace(has_unfinished_requests=lambda: True),
        model_executor=SimpleNamespace(is_sleeping=False),
        capture_iteration_details=capture_iteration_details,
        progress_monitor=progress_monitor,
        execute_dummy_batch=lambda: None,
        has_coordinator=True,
        _has_global_unfinished_reqs=lambda _local_unfinished: True,
    )

    with pytest.raises(SystemExit):
        engine_core_module.DPEngineCoreProc.run_busy_loop(engine)

    assert progress_monitor.activities == [
        ("execute_dummy_batch", None),
        ("dp_global_sync", None),
    ]
    assert progress_monitor.progress == []


def test_process_engine_step_records_output_progress():
    progress_monitor = FakeProgressMonitor()
    queued_outputs: list[tuple[int, Any]] = []
    engine_outputs = EngineCoreOutputs(
        outputs=[EngineCoreOutput(request_id="req-1", new_token_ids=[1])]
    )
    engine = SimpleNamespace(
        progress_monitor=progress_monitor,
        output_queue=SimpleNamespace(put_nowait=queued_outputs.append),
        post_step=lambda model_executed: None,
        scheduler=SimpleNamespace(has_requests=lambda: True),
        step_fn=lambda: ({0: engine_outputs}, False),
    )

    assert not engine_core_module.EngineCoreProc._process_engine_step(engine)
    assert queued_outputs == [(0, engine_outputs)]
    assert progress_monitor.activities == [("engine_step", None)]
    assert progress_monitor.progress == [
        (
            "engine_step",
            {
                "has_pending_work": True,
                "model_executed": False,
                "num_outputs": 1,
            },
        )
    ]


def test_process_engine_step_disabled_monitor_skips_diagnostic_work():
    def fail_has_requests() -> bool:
        raise AssertionError("disabled monitor queried scheduler state")

    engine = SimpleNamespace(
        progress_monitor=None,
        output_queue=SimpleNamespace(put_nowait=lambda _output: None),
        post_step=lambda model_executed: None,
        scheduler=SimpleNamespace(has_requests=fail_has_requests),
        step_fn=lambda: ({}, True),
    )

    assert engine_core_module.EngineCoreProc._process_engine_step(engine)


def test_describe_process_exit_identifies_signal_exit():
    signum = int(signal.SIGTERM)
    status = dump_input.describe_process_exit(-signum)

    assert status == {
        "exit_code": -signum,
        "signal_name": signal.Signals(signum).name,
        "signal_number": signum,
        "status": "signal",
    }
    assert dump_input.format_process_exit(-signum) == (
        f"signal {signal.Signals(signum).name} ({signum})"
    )


def test_process_exit_status_retry_has_one_shared_budget(monkeypatch):
    """Unavailable status must not indefinitely delay failure cleanup."""
    elapsed = 0.0
    budget = 0.0025
    procs = [
        FakeProc(f"worker-{i}", pid=i + 1, exitcode=None, sentinel=i) for i in range(3)
    ]

    def advance_clock(delay):
        nonlocal elapsed
        assert 0 < delay <= 0.001
        elapsed += delay
        assert elapsed <= budget

    monkeypatch.setattr(
        v1_utils,
        "time",
        SimpleNamespace(monotonic=lambda: elapsed, sleep=advance_clock),
    )

    v1_utils.wait_for_process_exit_status(cast(Any, procs), timeout_s=budget)

    assert elapsed == pytest.approx(budget)
    assert all(proc.exitcode is None for proc in procs)


def test_process_death_diagnostics_writes_bundle(tmp_path, monkeypatch):
    signum = int(signal.SIGTERM)
    config = enable_engine_diagnostic_bundles(monkeypatch, tmp_path)

    bundle_dir = dump_input.dump_process_death_diagnostics(
        config,
        process_kind="worker",
        process_name="WorkerProc-0",
        pid=1234,
        exitcode=-signum,
        details={"rank": 0, "world_size": 2},
    )

    assert bundle_dir is not None
    assert bundle_dir.parent == (
        tmp_path / "rank_0_dp_0" / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR
    )
    context = json.loads((bundle_dir / "context.json").read_text(encoding="utf-8"))
    assert context["reason"] == "process_death"
    assert context["stage"] == "worker"
    assert context["process_death"] == {
        "details": {"rank": 0, "world_size": 2},
        "exit_code": -signum,
        "pid": 1234,
        "process_kind": "worker",
        "process_name": "WorkerProc-0",
        "signal_name": signal.Signals(signum).name,
        "signal_number": signum,
        "status": "signal",
    }
    manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["complete"]


def test_core_engine_proc_manager_dumps_one_bundle_for_failed_processes(monkeypatch):
    signum = int(signal.SIGTERM)
    proc = FakeProc(
        "EngineCore",
        pid=1234,
        exitcode=0,
        sentinel=17,
    )
    other_proc = FakeProc(
        "EngineCore_DP1",
        pid=1235,
        exitcode=-signum,
        sentinel=18,
        pending_exitcode_reads=1,
    )
    manager = cast(Any, object.__new__(engine_utils.CoreEngineProcManager))
    manager.processes = [proc, other_proc]
    manager.manager_stopped = SimpleNamespace(is_set=lambda: False)
    manager.failed_proc_name = None
    manager.vllm_config = SimpleNamespace()
    shutdown_calls: list[float | None] = []
    manager.shutdown = lambda timeout=None: shutdown_calls.append(timeout)
    diagnostics: list[dict[str, Any]] = []

    def record_diagnostics(config, **kwargs):
        diagnostics.append(kwargs)

    monkeypatch.setattr(
        engine_utils.connection,
        "wait",
        lambda sentinels, timeout: [proc.sentinel, other_proc.sentinel],
    )
    monkeypatch.setattr(
        engine_utils,
        "dump_process_death_diagnostics",
        record_diagnostics,
    )

    engine_utils.CoreEngineProcManager.monitor_engine_liveness(manager)

    assert manager.failed_proc_name == "EngineCore_DP1"
    assert shutdown_calls == [None]
    assert diagnostics == [
        {
            "process_kind": "engine_core",
            "process_name": "EngineCore_DP1",
            "pid": 1235,
            "exitcode": -signum,
            "details": {
                "finished_processes": {
                    "EngineCore": 0,
                    "EngineCore_DP1": -signum,
                },
                "local_engine_count": 2,
            },
        }
    ]


def test_core_engine_proc_manager_dumps_unexpected_clean_exit(monkeypatch):
    proc = FakeProc("EngineCore", pid=1234, exitcode=0, sentinel=17)
    manager = cast(Any, object.__new__(engine_utils.CoreEngineProcManager))
    manager.processes = [proc]
    manager.manager_stopped = SimpleNamespace(is_set=lambda: False)
    manager.failed_proc_name = None
    manager.vllm_config = SimpleNamespace()
    manager.shutdown = lambda timeout=None: None
    diagnostics: list[dict[str, Any]] = []

    monkeypatch.setattr(
        engine_utils.connection,
        "wait",
        lambda sentinels, timeout: [proc.sentinel],
    )
    monkeypatch.setattr(
        engine_utils,
        "dump_process_death_diagnostics",
        lambda config, **kwargs: diagnostics.append(kwargs),
    )

    engine_utils.CoreEngineProcManager.monitor_engine_liveness(manager)

    assert manager.failed_proc_name == "EngineCore"
    assert diagnostics[0]["exitcode"] == 0


def test_core_engine_proc_manager_suppresses_expected_shutdown(monkeypatch):
    proc = FakeProc("EngineCore", pid=1234, exitcode=0, sentinel=17)
    stopped = threading.Event()
    manager = cast(Any, object.__new__(engine_utils.CoreEngineProcManager))
    manager.processes = [proc]
    manager.manager_stopped = stopped
    manager.failed_proc_name = None
    manager.shutdown = lambda timeout=None: None

    def stop_while_waiting(sentinels, timeout):
        stopped.set()
        return [proc.sentinel]

    monkeypatch.setattr(engine_utils.connection, "wait", stop_while_waiting)
    monkeypatch.setattr(
        engine_utils,
        "dump_process_death_diagnostics",
        lambda *args, **kwargs: pytest.fail("unexpected diagnostic"),
    )

    engine_utils.CoreEngineProcManager.monitor_engine_liveness(manager)

    assert manager.failed_proc_name is None


def test_process_death_diagnostics_failure_is_fail_open(monkeypatch):
    def fail_dump_root(config):
        raise RuntimeError("diagnostic path failure")

    monkeypatch.setattr(dump_input, "_engine_diagnostic_dump_root", fail_dump_root)

    assert (
        dump_input.dump_process_death_diagnostics(
            SimpleNamespace(),
            process_kind="worker",
            process_name="WorkerProc-0",
            pid=1234,
            exitcode=1,
        )
        is None
    )


def test_multiproc_worker_monitor_dumps_failed_worker(monkeypatch):
    signum = int(signal.SIGTERM)
    clean_proc = FakeProc(
        "WorkerProc-1",
        pid=2344,
        exitcode=0,
        sentinel=28,
    )
    proc = FakeProc(
        "WorkerProc-2",
        pid=2345,
        exitcode=-signum,
        sentinel=29,
        pending_exitcode_reads=1,
    )
    clean_worker = SimpleNamespace(proc=clean_proc, rank=1)
    worker = SimpleNamespace(proc=proc, rank=2)
    executor = cast(Any, object.__new__(multiproc_executor_module.MultiprocExecutor))
    executor.workers = [clean_worker, worker]
    executor.vllm_config = SimpleNamespace()
    executor.local_world_size = 4
    executor.world_size = 8
    executor.is_failed = False
    executor.shutting_down = False
    shutdown_calls: list[bool] = []
    callback_calls: list[bool] = []
    executor.shutdown = lambda: shutdown_calls.append(True)
    executor.failure_callback = lambda: callback_calls.append(True)
    diagnostics: list[dict[str, Any]] = []

    def record_diagnostics(config, **kwargs):
        diagnostics.append(kwargs)

    monkeypatch.setattr(
        multiproc_executor_module.multiprocessing.connection,
        "wait",
        lambda sentinels: [clean_proc.sentinel, proc.sentinel],
    )
    monkeypatch.setattr(
        multiproc_executor_module,
        "dump_process_death_diagnostics",
        record_diagnostics,
    )

    multiproc_executor_module.MultiprocExecutor.start_worker_monitor(
        executor, inline=True
    )

    assert executor.is_failed
    assert executor.failure_callback is None
    assert shutdown_calls == [True]
    assert callback_calls == [True]
    assert diagnostics == [
        {
            "process_kind": "worker",
            "process_name": "WorkerProc-2",
            "pid": 2345,
            "exitcode": -signum,
            "details": {
                "finished_workers": {1: 0, 2: -signum},
                "local_world_size": 4,
                "rank": 2,
                "world_size": 8,
            },
        }
    ]


def test_multiproc_worker_monitor_suppresses_expected_shutdown(monkeypatch):
    proc = FakeProc("WorkerProc-2", pid=2345, exitcode=0, sentinel=29)
    worker = SimpleNamespace(proc=proc, rank=2)
    executor = cast(Any, object.__new__(multiproc_executor_module.MultiprocExecutor))
    executor.workers = [worker]
    executor.shutting_down = True
    executor.is_failed = False

    monkeypatch.setattr(
        multiproc_executor_module.multiprocessing.connection,
        "wait",
        lambda sentinels: [proc.sentinel],
    )
    monkeypatch.setattr(
        multiproc_executor_module,
        "dump_process_death_diagnostics",
        lambda *args, **kwargs: pytest.fail("unexpected diagnostic"),
    )

    multiproc_executor_module.MultiprocExecutor.start_worker_monitor(
        executor, inline=True
    )

    assert not executor.is_failed


def test_mp_client_monitor_supports_actor_manager_without_finished_procs(
    monkeypatch,
):
    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self._target = target

        def start(self):
            self._target()

    manager = SimpleNamespace(
        failed_proc_name="Actor actor-id",
        monitor_engine_liveness=lambda: None,
    )
    client = object.__new__(core_client_module.MPClient)
    client.resources = SimpleNamespace(engine_manager=manager, engine_dead=False)
    client._finalizer = SimpleNamespace(alive=True)
    shutdown_calls: list[bool] = []
    client.shutdown = lambda: shutdown_calls.append(True)
    monkeypatch.setattr(core_client_module, "Thread", ImmediateThread)

    core_client_module.MPClient.start_engine_core_monitor(client)

    assert client.resources.engine_dead
    assert shutdown_calls == [True]


def test_wait_for_completion_supports_actor_manager_without_finished_procs(
    monkeypatch,
):
    class Endpoint:
        def close(self):
            pass

    class ImmediateThread:
        def __init__(self, *, target, **kwargs):
            self._target = target

        def start(self):
            self._target()

    recv, send = Endpoint(), Endpoint()
    manager = SimpleNamespace(
        failed_proc_name="Actor actor-id",
        monitor_engine_liveness=lambda: None,
    )
    monkeypatch.setattr(v1_utils.connection, "Pipe", lambda duplex: (recv, send))
    monkeypatch.setattr(v1_utils.connection, "wait", lambda sentinels: [recv])
    monkeypatch.setattr(v1_utils.threading, "Thread", ImmediateThread)

    with pytest.raises(RuntimeError, match="Actor actor-id"):
        v1_utils.wait_for_completion_or_failure(
            api_server_manager=SimpleNamespace(processes=[]),
            engine_manager=manager,
        )
