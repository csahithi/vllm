# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from ...utils import RemoteOpenAIServer

# Optimized server arguments for test groups (maximizing compatibility)
# 
# SESSION-SCOPED STRATEGY (Single Worker):
# 1. All tests share the same server instances (session-scoped)
# 2. Maximum model reuse - no model switching between tests
# 3. Single worker execution - no race conditions or conflicts
# 4. Tests are stateless - safe to share servers
#
DEFAULT_ARGS = {
    # Group 1: Tiny Model Tests (Most Compatible) - 11 test files can share this
    # Tests: test_chat.py, test_completion.py, test_basic.py, test_uds.py, test_chunked_prompt.py, 
    #        test_sleep.py, test_shutdown.py, test_metrics.py, test_root_path.py, test_chat_echo.py, test_chat_logit_bias_validation.py
    "tiny_model": ["--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests"],
    
    # Group 2: Embedding Tests - 3 test files can share this
    # Tests: test_embedding.py, test_embedding_long_text.py, test_optional_middleware.py
    "embedding": ["--runner", "pooling", "--dtype", "bfloat16", "--enforce-eager", "--max-model-len", "512", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests"],
    
    # Group 3: Specialized Tests (Keep Separate) - Individual configurations
    "vision": ["--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"image": 2}', "--disable-log-stats", "--disable-log-requests"],
    "audio": ["--dtype", "float32", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"audio": 2}', "--disable-log-stats", "--disable-log-requests"],
    "video": ["--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"video": 1}', "--disable-log-stats", "--disable-log-requests"],
    "multimodal": ["--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "16", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests"]
}

# Global model cache directory
MODEL_CACHE_DIR = None

def get_model_cache_dir():
    """Get or create the global model cache directory."""
    global MODEL_CACHE_DIR
    if MODEL_CACHE_DIR is None:
        MODEL_CACHE_DIR = tempfile.mkdtemp(prefix="vllm_test_models_")
        print(f"📁 Model cache directory: {MODEL_CACHE_DIR}")
    return MODEL_CACHE_DIR

def download_model_once(model_name, cache_dir):
    """Download model once and cache it for reuse."""
    model_path = os.path.join(cache_dir, model_name.replace("/", "_"))
    
    if os.path.exists(model_path):
        print(f"✅ Using cached model: {model_name}")
        return model_path
    
    print(f"📥 Downloading model: {model_name}")
    
    # Use huggingface_hub to download
    try:
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=model_name, local_dir=model_path)
        print(f"✅ Model downloaded and cached: {model_name}")
        return model_path
    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        # Fallback to original model name - this should work for local models
        print(f"🔄 Falling back to original model name: {model_name}")
        return model_name

import threading

class DynamicModelServer:
    """Single server that can switch between different models."""
    
    def __init__(self):
        self.current_model = None
        self.current_server = None
        self.cache_dir = get_model_cache_dir()
        self._lock = threading.Lock()  # Thread safety for parallel workers
        
    def get_server(self, model_name, server_args=None, model_type=None):
        """Get or create server for the specified model.
        
        Args:
            model_name: HuggingFace model name or path
            server_args: Custom server arguments (optional)
            model_type: Type for default args (optional, e.g., 'chat', 'embedding')
        """
        # Create a unique key for this model+args combination
        model_key = f"{model_name}_{hash(str(server_args))}"
        
        with self._lock:  # Thread-safe access
            if self.current_model == model_key and self.current_server is not None:
                return self.current_server
            
            # Close existing server if switching models
            if self.current_server is not None:
                print(f"🔄 Switching from {self.current_model} to {model_key}")
                self.current_server.__exit__(None, None, None)
            
            # Use custom args or default args based on model type
            if server_args is None and model_type in DEFAULT_ARGS:
                args = DEFAULT_ARGS[model_type].copy()
            elif server_args is None:
                # Default to tiny_model args for unknown model types
                args = DEFAULT_ARGS["tiny_model"].copy()
            else:
                args = server_args.copy()
            
            # Add memory optimization flags
            memory_flags = [
                "--gpu-memory-utilization", "0.7",  # Use only 70% of GPU memory
                "--max-model-len", "1024",  # Reduce from 2048 to save memory
                "--max-num-seqs", "32",     # Reduce from 64 to save memory
                "--disable-log-stats",
                "--disable-log-requests"
            ]
            
            # Merge memory flags with existing args
            for flag in memory_flags:
                if flag not in args:
                    args.append(flag)
            
            # Download and cache model
            cached_model_path = download_model_once(model_name, self.cache_dir)
            
            # Create new server
            print(f"🚀 Starting server for model: {model_name}")
            self.current_server = RemoteOpenAIServer(cached_model_path, args, max_wait_seconds=120)
            self.current_model = model_key
            
            return self.current_server
    
    def cleanup(self):
        """Clean up all resources."""
        if self.current_server is not None:
            self.current_server.__exit__(None, None, None)
        
        # Optionally clean up cache (comment out to keep models between test runs)
        # if MODEL_CACHE_DIR and os.path.exists(MODEL_CACHE_DIR):
        #     shutil.rmtree(MODEL_CACHE_DIR)

# Global server instance
_dynamic_server = None

def get_dynamic_server():
    """Get the global dynamic server instance."""
    global _dynamic_server
    if _dynamic_server is None:
        _dynamic_server = DynamicModelServer()
    return _dynamic_server

# Removed helper functions - no longer needed with server_factory approach

# Removed individual server fixtures - use server_factory instead for flexibility

@pytest.fixture(scope="session")
def dynamic_server():
    """Direct access to the dynamic server for advanced usage."""
    return get_dynamic_server()

@pytest.fixture(scope="session")
def server_factory():
    """Factory fixture for creating servers with custom models and args.
    
    Usage:
        def test_my_model(server_factory):
            server = server_factory("my/model", model_type="chat")
            # or
            server = server_factory("my/model", server_args=["--custom", "args"])
    """
    def create_server(model_name, server_args=None, model_type=None):
        server = get_dynamic_server()
        return server.get_server(model_name, server_args, model_type)
    return create_server

# Session-scoped server fixtures for maximum efficiency (single worker)
@pytest.fixture(scope="session")
def server(server_factory):
    """Session-scoped server with tiny model - ALL test files share this server."""
    return server_factory("hmellor/tiny-random-LlamaForCausalLM", model_type="tiny_model")

@pytest.fixture(scope="session")
def embedding_server(server_factory):
    """Session-scoped embedding server - ALL embedding tests share this server."""
    return server_factory("intfloat/multilingual-e5-small", model_type="embedding")

# Additional session-scoped fixtures for specialized tests
@pytest.fixture(scope="session")
def vision_server(server_factory):
    """Session-scoped vision server for multimodal tests."""
    return server_factory("microsoft/Phi-3.5-vision-instruct", model_type="vision")

@pytest.fixture(scope="session")
def audio_server(server_factory):
    """Session-scoped audio server for audio processing tests."""
    return server_factory("fixie-ai/ultravox-v0_5-llama-3_2-1b", model_type="audio")

# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_models():
    """Clean up models after all tests complete."""
    yield
    if _dynamic_server is not None:
        _dynamic_server.cleanup()
