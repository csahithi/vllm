# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ...utils import RemoteOpenAIServer

# Advanced tests may need different server configurations
# This provides a factory pattern for flexibility
DEFAULT_SERVER_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "1024",
    "--enforce-eager",
    "--max-num-seqs", "16",
    "--gpu-memory-utilization", "0.7",
    "--disable-log-stats",
    "--disable-log-requests"
]

@pytest.fixture(scope="package")
def server_factory():
    """Factory fixture for creating servers with custom models and args."""
    def create_server(model_name, server_args=None):
        if server_args is None:
            server_args = DEFAULT_SERVER_ARGS.copy()
        server = RemoteOpenAIServer(model_name, server_args, max_wait_seconds=120)
        return server
    return create_server

@pytest.fixture(scope="package")
def default_server():
    """Default server for tests that don't need special configuration."""
    server = RemoteOpenAIServer("microsoft/DialoGPT-small", DEFAULT_SERVER_ARGS, max_wait_seconds=120)
    yield server
    server.__exit__(None, None, None)
