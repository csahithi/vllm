# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ...utils import RemoteOpenAIServer

# Embedding tests use pooling runner and optimized settings
EMBEDDING_SERVER_ARGS = [
    "--runner", "pooling",
    "--dtype", "bfloat16",
    "--enforce-eager",
    "--max-model-len", "512",
    "--gpu-memory-utilization", "0.7",
    "--disable-log-stats",
    "--disable-log-requests"
]

@pytest.fixture(scope="package")
def embedding_server():
    """Package-scoped embedding server."""
    server = RemoteOpenAIServer("intfloat/multilingual-e5-small", EMBEDDING_SERVER_ARGS, max_wait_seconds=120)
    yield server
    server.__exit__(None, None, None)
