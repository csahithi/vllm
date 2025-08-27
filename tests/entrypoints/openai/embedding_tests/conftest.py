# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ...utils import RemoteOpenAIServer

# Single, flexible embedding server configuration that works for all tests
# This server can handle multilingual-e5-small, matryoshka models, and middleware tests
UNIVERSAL_EMBEDDING_ARGS = [
    "--runner", "pooling",
    "--dtype", "bfloat16",  # Use bfloat16 for better compatibility
    "--enforce-eager",
    "--max-model-len", "512",
    "--gpu-memory-utilization", "0.7",
    "--max-num-seqs", "4",  # Increased to handle multiple test scenarios
    "--disable-log-stats",
    "--disable-log-requests"
]

@pytest.fixture(scope="package")
def embedding_server():
    """Single package-scoped embedding server for all embedding tests.
    
    This server uses intfloat/multilingual-e5-small which is:
    - Compatible with most embedding test scenarios
    - Supports the pooling runner
    - Has reasonable memory requirements
    - Can handle the test workloads efficiently
    """
    server = RemoteOpenAIServer("intfloat/multilingual-e5-small", UNIVERSAL_EMBEDDING_ARGS, max_wait_seconds=120)
    yield server
    server.__exit__(None, None, None)
