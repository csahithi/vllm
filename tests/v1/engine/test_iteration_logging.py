# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import time
from collections import deque
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import vllm.v1.core.sched.scheduler as scheduler_module
import vllm.v1.engine.core as engine_core_module
from vllm.logging_utils import dump_input
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs
from vllm.v1.engine.core import EngineCore
from vllm.v1.metrics.stats import SchedulerIterationDetails, SchedulerStats
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


class FakeTimer:
    def __init__(self, interval, function):
        self.interval = interval
        self.function = function
        self.daemon = False
        self.started = False
        self.cancelled = False

    def start(self):
        self.started = True

    def cancel(self):
        self.cancelled = True


class FakeFuture:
    def __init__(self, result):
        self._result = result

    def result(self):
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
        self.execute_future = FakeFuture(execute_result)
        self.sample_future = FakeFuture(sample_result)
        self.sample_result = sample_result
        self.sample_calls = []

    def execute_model(self, scheduler_output, non_block=False):
        return self.execute_future

    def sample_tokens(self, grammar_output, non_block=False):
        self.sample_calls.append((grammar_output, non_block))
        return self.sample_future if non_block else self.sample_result


class FakeStageEngine:
    def __init__(self, scheduler, model_executor=None):
        self.scheduler = scheduler
        self.model_executor = model_executor
        self.stages = []
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
    def dump_on_slow_execution(self, scheduler_output, stage):
        self.stages.append(stage)
        yield

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


def clear_engine_execution_timeout_dump_throttle():
    with dump_input._engine_execution_timeout_dump_lock:
        dump_input._engine_execution_timeout_dump_last_s.clear()


def clear_stack_trace_signal_handler_state():
    with dump_input._stack_trace_signal_handler_lock:
        dump_input._stack_trace_signal_handlers.clear()


def make_debug_dump_config(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(compile_debug_dump_path=lambda: tmp_path)


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

    monkeypatch.setattr(scheduler_module.time, "time", lambda: 1000.0)

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


def test_engine_execution_timeout_dumper_cancels_timer(monkeypatch):
    timers = []
    dumps = []

    def timer_factory(interval, function):
        timer = FakeTimer(interval, function)
        timers.append(timer)
        return timer

    monkeypatch.setattr(dump_input.threading, "Timer", timer_factory)
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumps.append(args),
    )

    with dump_input.EngineExecutionTimeoutDumper(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=3.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    ):
        pass

    assert len(timers) == 1
    assert timers[0].interval == 3.0
    assert timers[0].daemon
    assert timers[0].started
    assert timers[0].cancelled
    assert not dumps


def test_engine_execution_timeout_dumper_dumps_when_timer_fires(monkeypatch):
    timers = []
    dumps = []
    scheduler_snapshot = {"requests": {"running": {"count": 1}}}

    def timer_factory(interval, function):
        timer = FakeTimer(interval, function)
        timers.append(timer)
        return timer

    monkeypatch.setattr(dump_input.threading, "Timer", timer_factory)
    monkeypatch.setattr(
        dump_input,
        "dump_engine_execution_timeout",
        lambda *args: dumps.append(args),
    )

    scheduler_output = SimpleNamespace()
    scheduler_stats = SchedulerStats(num_running_reqs=1)
    with dump_input.EngineExecutionTimeoutDumper(
        config=SimpleNamespace(),
        scheduler_output=scheduler_output,
        scheduler_stats=scheduler_stats,
        timeout_s=2.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
        scheduler_snapshot_fn=lambda: scheduler_snapshot,
    ):
        timers[0].function()

    assert len(dumps) == 1
    assert dumps[0][1] is scheduler_output
    assert dumps[0][2] is scheduler_stats
    assert dumps[0][3] == 2.0
    assert dumps[0][4] == engine_core_module.EXECUTE_MODEL_WAIT_STAGE
    assert dumps[0][5] is scheduler_snapshot


def test_engine_execution_timeout_dump_is_throttled_by_stage(monkeypatch):
    contexts: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
    tracebacks: list[dict[str, Any]] = []
    times = iter([100.0, 101.0, 102.0, 401.0])

    def record_context(*args: Any, **kwargs: Any) -> None:
        contexts.append((args, kwargs))

    def record_traceback(*args: Any, **kwargs: Any) -> None:
        tracebacks.append(kwargs)

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(dump_input, "_dump_engine_execution_context", record_context)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)

    dump_input.dump_engine_execution_timeout(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=1.0,
        stage=engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )
    dump_input.dump_engine_execution_timeout(
        config=SimpleNamespace(),
        scheduler_output=SimpleNamespace(),
        scheduler_stats=SchedulerStats(),
        timeout_s=1.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    assert len(contexts) == 3
    assert len(tracebacks) == 3
    assert all(context[0][0] == "timeout" for context in contexts)
    assert all("stage" in context[1] for context in contexts)
    assert all("timeout_s" in context[1] for context in contexts)

    clear_engine_execution_timeout_dump_throttle()


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

    clear_stack_trace_signal_handler_state()
    monkeypatch.setattr(
        dump_input.envs,
        "VLLM_DEBUG_STACK_TRACE_SIGNAL",
        "42",
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
            "signum": 42,
        }
    ]

    clear_stack_trace_signal_handler_state()


def test_install_stack_trace_signal_handler_rejects_protected_signal(monkeypatch):
    registered: list[tuple[Any, ...]] = []

    def record_unexpected_register(*args: Any, **_kwargs: Any) -> None:
        registered.append(args)

    clear_stack_trace_signal_handler_state()
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

    clear_stack_trace_signal_handler_state()


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
    monkeypatch.setattr(plugins, "load_general_plugins", lambda: None)

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

    assert calls == ["Worker_3"]
    assert isinstance(wrapper.worker, FakeDiagnosticWorker)


def test_engine_no_progress_dump_writes_diagnostic_bundle(tmp_path, monkeypatch):
    def record_traceback(file, all_threads):
        assert all_threads
        file_name = str(getattr(file, "name", ""))
        if file_name.endswith("stacks.txt"):
            file.write("stack dump\n")

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)

    dump_input.dump_engine_no_progress(
        config=make_debug_dump_config(tmp_path),
        scheduler_stats=SchedulerStats(num_running_reqs=1),
        timeout_s=3.0,
        progress_snapshot={"stage": "engine_step", "progress_index": 1},
        scheduler_snapshot={"requests": {"running": {"count": 1}}},
    )

    bundles = list((tmp_path / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR).iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "stacks.txt").read_text(encoding="utf-8") == "stack dump\n"

    context = json.loads((bundles[0] / "context.json").read_text(encoding="utf-8"))
    assert context["reason"] == "no_progress"
    assert context["stage"] == dump_input.ENGINE_NO_PROGRESS_STAGE
    assert context["timeout_s"] == 3.0
    assert context["scheduler_output"] is None
    assert "num_running_reqs=1" in context["scheduler_stats"]
    assert context["scheduler_snapshot_file"] == "scheduler_snapshot.json"

    snapshot = json.loads(
        (bundles[0] / "scheduler_snapshot.json").read_text(encoding="utf-8")
    )
    assert snapshot["engine_progress"]["stage"] == "engine_step"
    assert snapshot["scheduler"]["requests"]["running"]["count"] == 1

    clear_engine_execution_timeout_dump_throttle()


def test_progress_monitor_dumps_when_work_stalls(monkeypatch):
    now_s = 100.0
    dumps: list[tuple[Any, ...]] = []
    scheduler_stats = SchedulerStats(num_waiting_reqs=1)
    scheduler_snapshot = {"requests": {"waiting": {"count": 1}}}

    def current_time() -> float:
        return now_s

    monitor = dump_input.EngineCoreProgressMonitor(
        config=SimpleNamespace(),
        process_name="EngineCore_0",
        timeout_s=2.0,
        has_work_fn=lambda: True,
        scheduler_stats_fn=lambda: scheduler_stats,
        scheduler_snapshot_fn=lambda: scheduler_snapshot,
        time_fn=current_time,
    )
    monitor.record_progress("engine_step")
    now_s = 102.5
    monkeypatch.setattr(
        dump_input,
        "dump_engine_no_progress",
        lambda *args: dumps.append(args),
    )

    assert monitor.maybe_dump_no_progress()
    assert len(dumps) == 1
    assert dumps[0][1] is scheduler_stats
    assert dumps[0][2] == 2.0
    assert dumps[0][3]["stage"] == "engine_step"
    assert dumps[0][3]["elapsed_since_progress_s"] == 2.5
    assert dumps[0][4] is scheduler_snapshot


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
        scheduler_stats_fn=lambda: SchedulerStats(),
        scheduler_snapshot_fn=lambda: None,
        time_fn=current_time,
    )
    monitor.record_progress("engine_step")
    now_s = 101.5
    monkeypatch.setattr(
        dump_input,
        "dump_engine_no_progress",
        lambda *args: dumps.append(args),
    )

    assert not monitor.maybe_dump_no_progress()
    assert not dumps
    assert monitor.snapshot()["stage"] == "idle"


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


def test_process_engine_step_records_output_progress():
    progress_monitor = FakeProgressMonitor()
    queued_outputs: list[tuple[int, Any]] = []
    engine_outputs = EngineCoreOutputs(
        outputs=[
            EngineCoreOutput(request_id="req-1", new_token_ids=[1])
        ]
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


def test_engine_execution_context_writes_diagnostic_bundle(tmp_path):
    scheduler_output = SimpleNamespace(request_id="req-1", token_ids=[1, 2, 3])
    scheduler_stats = SchedulerStats(num_running_reqs=1)
    error = RuntimeError("model execution failed")

    bundle_dir = dump_input._dump_engine_execution_context(
        reason="exception",
        config=make_debug_dump_config(tmp_path),
        scheduler_output=scheduler_output,
        scheduler_stats=scheduler_stats,
        error=error,
        scheduler_snapshot={"kv_cache": {"usage": 0.5}},
    )

    assert bundle_dir is not None
    assert bundle_dir.parent == tmp_path / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR

    context = json.loads((bundle_dir / "context.json").read_text(encoding="utf-8"))
    assert context["bundle_version"] == dump_input.ENGINE_DIAGNOSTIC_BUNDLE_VERSION
    assert context["reason"] == "exception"
    assert context["stage"] is None
    assert context["timeout_s"] is None
    assert "req-1" in context["scheduler_output"]
    assert "num_running_reqs=1" in context["scheduler_stats"]
    assert context["exception"]["type"] == "builtins.RuntimeError"
    assert context["exception"]["message"] == "model execution failed"
    assert context["scheduler_snapshot_file"] == "scheduler_snapshot.json"
    scheduler_snapshot = json.loads(
        (bundle_dir / "scheduler_snapshot.json").read_text(encoding="utf-8")
    )
    assert scheduler_snapshot == {"kv_cache": {"usage": 0.5}}


def test_engine_execution_context_skips_bundle_without_debug_dump_path(tmp_path):
    bundle_dir = dump_input._dump_engine_execution_context(
        reason="exception",
        config=SimpleNamespace(compile_debug_dump_path=lambda: None),
        scheduler_output=SimpleNamespace(request_id="req-1"),
        scheduler_stats=None,
    )

    assert bundle_dir is None
    assert not (tmp_path / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR).exists()


def test_engine_execution_timeout_writes_stack_bundle(tmp_path, monkeypatch):
    stack_dumps: list[str] = []

    def record_traceback(file, all_threads):
        assert all_threads
        file.write("stack dump\n")
        file_name = str(getattr(file, "name", ""))
        if file_name.endswith("stacks.txt"):
            stack_dumps.append(file_name)

    clear_engine_execution_timeout_dump_throttle()
    monkeypatch.setattr(dump_input.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(dump_input.faulthandler, "dump_traceback", record_traceback)

    dump_input.dump_engine_execution_timeout(
        config=make_debug_dump_config(tmp_path),
        scheduler_output=SimpleNamespace(request_id="req-2"),
        scheduler_stats=SchedulerStats(num_waiting_reqs=2),
        timeout_s=2.0,
        stage=engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
    )

    bundles = list((tmp_path / dump_input.ENGINE_DIAGNOSTIC_DUMP_DIR).iterdir())
    assert len(bundles) == 1
    assert (bundles[0] / "stacks.txt").read_text(encoding="utf-8") == "stack dump\n"

    context = json.loads((bundles[0] / "context.json").read_text(encoding="utf-8"))
    assert context["reason"] == "timeout"
    assert context["stage"] == engine_core_module.EXECUTE_MODEL_WAIT_STAGE
    assert context["timeout_s"] == 2.0
    assert "req-2" in context["scheduler_output"]
    assert "num_waiting_reqs=2" in context["scheduler_stats"]
    assert context["exception"] is None
    assert context["scheduler_snapshot_file"] is None
    assert stack_dumps[0].endswith("stacks.txt")

    clear_engine_execution_timeout_dump_throttle()


def test_dump_on_slow_execution_uses_env_timeout_and_scheduler_stats(monkeypatch):
    calls: list[Any] = []
    scheduler_stats = SchedulerStats(num_waiting_reqs=2)
    scheduler_snapshot = {"requests": {"waiting": {"count": 2}}}

    class FakeDumper:
        def __init__(self, **kwargs):
            calls.append(kwargs)

        def __enter__(self):
            calls.append("enter")

        def __exit__(self, exc_type, exc_value, traceback):
            calls.append("exit")

    engine = SimpleNamespace(
        vllm_config=SimpleNamespace(),
        scheduler=SimpleNamespace(make_stats=lambda: scheduler_stats),
        make_scheduler_diagnostic_snapshot=lambda: scheduler_snapshot,
    )
    scheduler_output = SimpleNamespace()

    monkeypatch.setattr(engine_core_module, "EngineExecutionTimeoutDumper", FakeDumper)
    monkeypatch.setattr(engine_core_module.envs, "VLLM_ENGINE_ITERATION_TIMEOUT_S", 7)

    with EngineCore.dump_on_slow_execution(
        engine, scheduler_output, engine_core_module.EXECUTE_MODEL_WAIT_STAGE
    ):
        calls.append("body")

    assert calls == [
        {
            "config": engine.vllm_config,
            "scheduler_output": scheduler_output,
            "scheduler_stats": scheduler_stats,
            "timeout_s": 7,
            "stage": engine_core_module.EXECUTE_MODEL_WAIT_STAGE,
            "scheduler_snapshot_fn": engine.make_scheduler_diagnostic_snapshot,
        },
        "enter",
        "body",
        "exit",
    ]


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
    future, queued_scheduler_output, exec_future, future_stage = engine.batch_queue[0]
    assert future is model_executor.sample_future
    assert queued_scheduler_output is scheduler_output
    assert exec_future is model_executor.execute_future
    assert future_stage == engine_core_module.SAMPLE_TOKENS_WAIT_STAGE
    assert model_executor.sample_calls == [("grammar", True)]


def test_step_with_batch_queue_uses_queued_future_stage():
    scheduler_output = SimpleNamespace(total_num_scheduled_tokens=1)
    scheduler = FakeStageScheduler(scheduler_output, has_requests=False)
    model_output = SimpleNamespace()
    exec_future = FakeFuture(SimpleNamespace())
    engine = FakeStageEngine(scheduler)
    engine.batch_queue = deque(
        [
            (
                FakeFuture(model_output),
                scheduler_output,
                exec_future,
                engine_core_module.SAMPLE_TOKENS_WAIT_STAGE,
            )
        ],
        maxlen=engine.batch_queue_size,
    )

    outputs, model_executed = EngineCore.step_with_batch_queue(engine)

    assert outputs == {}
    assert not model_executed
    assert scheduler.updated_with == (scheduler_output, model_output)
    assert engine.stages == [engine_core_module.SAMPLE_TOKENS_WAIT_STAGE]
