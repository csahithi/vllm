# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ...utils import RemoteOpenAIServer

# Basic tests use a simple, lightweight server configuration
BASIC_SERVER_ARGS = [
    "--dtype", "bfloat16",
    "--max-model-len", "1024",
    "--enforce-eager",
    "--max-num-seqs", "32",
    "--gpu-memory-utilization", "0.7",
    "--disable-log-stats",
    "--disable-log-requests",
    "--enable-server-load-tracking"
]

@pytest.fixture(scope="package")
def server():
    """Package-scoped server for basic API tests."""
    server = RemoteOpenAIServer("microsoft/DialoGPT-small", BASIC_SERVER_ARGS, max_wait_seconds=120)
    yield server
    server.__exit__(None, None, None)
