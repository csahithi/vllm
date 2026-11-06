# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Spawn targets for test_api_server_process_manager.py.

APIServerProcessManager starts its workers with the "spawn" start method, so
each child re-imports the module that defines its target. Keeping the targets
here, with only standard-library imports at module level, spares each child
the torch + vllm import that the test module would pull in (about 10s on an
idle host and several times that on a loaded CI host), which otherwise eats
most of the tests' wall-clock budgets. Workers that need vllm import it
lazily, and the tests give them a longer bound.
"""

import time


def mock_run_api_server_worker(listen_address, sock, args, client_config=None):
    """Mock run_api_server_worker that runs for `args.runtime` seconds.

    The runtime comes in through `args` because a spawned child never sees
    module globals that the parent test mutated.
    """
    print(f"Mock worker started with client_config: {client_config}")
    time.sleep(args.runtime)
    print("Mock worker completed successfully")


def update_admission_stats_worker(listen_address, sock, args, client_config):
    from vllm.v1.engine.admission_control import SharedAdmissionStats

    stats = SharedAdmissionStats(
        client_config,
        client_config["client_count"],
        client_config["client_index"],
    )
    value = client_config["client_index"] + 1
    stats.set_num_requests(value)
    args.barrier.wait()
    args.results.put(stats.get_num_requests())


def adopt_zmq_listener_worker(listen_address, sock, args, client_config):
    import zmq

    from vllm.utils.network_utils import make_zmq_socket

    ctx = zmq.Context()
    router = make_zmq_socket(
        ctx,
        client_config["input_address"],
        zmq.ROUTER,
        bind=True,
        listener=client_config["input_listener"],
    )
    pull = make_zmq_socket(
        ctx,
        client_config["output_address"],
        zmq.PULL,
        bind=True,
        listener=client_config["output_listener"],
    )
    try:
        identity, payload = router.recv_multipart()
        assert payload == b"input"
        assert pull.recv() == b"output"
        router.send_multipart([identity, b"ok"])
    finally:
        router.close(linger=0)
        pull.close(linger=0)
        ctx.term()
