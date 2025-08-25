# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from ...utils import RemoteOpenAIServer

# Shared models for testing - using smaller, faster models
CHAT_MODEL = "microsoft/DialoGPT-small"  # Much smaller than zephyr-7b-beta
EMBEDDING_MODEL = "intfloat/multilingual-e5-small"
MULTIMODAL_MODEL = "microsoft/Phi-4-multimodal-instruct"

@pytest.fixture(scope="session")
def shared_chat_server():
    """Shared server for chat and completion tests - session scope for maximum reuse"""
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "2048",  # Reduced from 8192 for faster loading
        "--enforce-eager",
        "--max-num-seqs", "64",     # Reduced from 128 for memory efficiency
        "--disable-log-stats",      # Reduce logging overhead
        "--disable-log-requests",   # Reduce logging overhead
    ]
    
    with RemoteOpenAIServer(CHAT_MODEL, args, max_wait_seconds=120) as server:
        yield server

@pytest.fixture(scope="session") 
def shared_embedding_server():
    """Shared server for embedding tests - session scope for maximum reuse"""
    args = [
        "--runner", "pooling",
        "--dtype", "bfloat16",
        "--enforce-eager",
        "--max-model-len", "512",
        "--disable-log-stats",
        "--disable-log-requests",
    ]
    
    with RemoteOpenAIServer(EMBEDDING_MODEL, args, max_wait_seconds=120) as server:
        yield server

@pytest.fixture(scope="session")
def shared_multimodal_server():
    """Shared server for multimodal tests - session scope for maximum reuse"""
    args = [
        "--dtype", "bfloat16",
        "--max-model-len", "2048",  # Reduced from 12800
        "--enforce-eager",
        "--max-num-seqs", "32",     # Reduced from 2 for better throughput
        "--disable-log-stats",
        "--disable-log-requests",
        "--gpu-memory-utilization", "0.7",  # Reduced from 0.8
    ]
    
    with RemoteOpenAIServer(MULTIMODAL_MODEL, args, max_wait_seconds=180) as server:
        yield server

# Backward compatibility fixtures for existing tests
@pytest.fixture(scope="module")
def server(shared_chat_server):
    """Backward compatibility - redirects to shared server"""
    return shared_chat_server

@pytest.fixture(scope="module")
def embedding_server(shared_embedding_server):
    """Backward compatibility - redirects to shared embedding server"""
    return shared_embedding_server
