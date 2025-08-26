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
    """Smart server that caches models separately from server configurations."""
    
    def __init__(self):
        self.model_cache = {}  # model_name -> cached_path
        self.server_cache = {}  # (model_name, args_hash) -> server_instance
        self.cache_dir = get_model_cache_dir()
        self._lock = threading.Lock()  # Thread safety for parallel workers
        
    def get_server(self, model_name, server_args=None, model_type=None):
        """Get or create server for the specified model.
        
        Args:
            model_name: HuggingFace model name or path
            server_args: Custom server arguments (optional)
            model_type: Type for default args (optional, e.g., 'tiny_model', 'embedding')
        """
        # Create a unique key for this model+args combination
        args_hash = hash(str(server_args) if server_args else str(model_type))
        server_key = (model_name, args_hash)
        
        with self._lock:  # Thread-safe access
            # Check if we already have this exact server configuration
            if server_key in self.server_cache:
                print(f"✅ Reusing existing server for {model_name} with args hash {args_hash}")
                return self.server_cache[server_key]
            
            # Check if we already have this model downloaded
            if model_name in self.model_cache:
                cached_model_path = self.model_cache[model_name]
                print(f"✅ Reusing cached model: {model_name}")
            else:
                # Download and cache model (only once per model name)
                cached_model_path = download_model_once(model_name, self.cache_dir)
                self.model_cache[model_name] = cached_model_path
                print(f"📥 Downloaded and cached model: {model_name}")
            
            # Use custom args or default args based on model type
            if server_args is None and model_type in DEFAULT_ARGS:
                args = DEFAULT_ARGS[model_type].copy()
            elif server_args is None:
                # Default to tiny_model args for unknown model types
                args = DEFAULT_ARGS["tiny_model"].copy()
            else:
                args = server_args.copy()
            
            # Add memory optimization flags if not already present
            memory_flags = [
                "--gpu-memory-utilization", "0.7",  # Use only 70% of GPU memory
                "--max-model-len", "1024",  # Reduce from 2048 to save memory
                "--max-num-seqs", "32",     # Reduce from 64 to save memory
                "--disable-log-stats",
                "--disable-log-requests"
            ]
            
            # Merge memory flags with existing args
            for i in range(0, len(memory_flags), 2):
                flag = memory_flags[i]
                value = memory_flags[i + 1]
                if flag not in args:
                    args.append(flag)
                    args.append(value)
            
            # Create new server
            print(f"🚀 Starting server for {model_name} with args hash {args_hash}")
            new_server = RemoteOpenAIServer(cached_model_path, args, max_wait_seconds=120)
            
            # Cache the server instance
            self.server_cache[server_key] = new_server
            
            return new_server
    
    def cleanup(self):
        """Clean up all resources."""
        # Clean up all server instances
        for server_key, server in self.server_cache.items():
            print(f"🧹 Cleaning up server for {server_key}")
            try:
                server.__exit__(None, None, None)
            except Exception as e:
                print(f"⚠️ Warning: Error cleaning up server {server_key}: {e}")
        
        # Clear caches
        self.server_cache.clear()
        self.model_cache.clear()
        
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

# Session-scoped server fixtures for maximum efficiency (single worker)
@pytest.fixture(scope="session")
def server():
    """Session-scoped server with tiny model - ALL test files share this server."""
    server = get_dynamic_server()
    return server.get_server("hmellor/tiny-random-LlamaForCausalLM", model_type="tiny_model")

@pytest.fixture(scope="session")
def embedding_server():
    """Session-scoped embedding server - ALL embedding tests share this server."""
    server = get_dynamic_server()
    return server.get_server("intfloat/multilingual-e5-small", model_type="embedding")

# Additional session-scoped fixtures for specialized tests
@pytest.fixture(scope="session")
def vision_server():
    """Session-scoped vision server for multimodal tests."""
    server = get_dynamic_server()
    return server.get_server("microsoft/Phi-3.5-vision-instruct", model_type="vision")

@pytest.fixture(scope="session")
def audio_server():
    """Session-scoped audio server for audio processing tests."""
    server = get_dynamic_server()
    return server.get_server("fixie-ai/ultravox-v0_5-llama-3_2-1b", model_type="audio")

# Factory fixture for advanced usage
@pytest.fixture(scope="session")
def server_factory():
    """Factory fixture for creating servers with custom models and args.
    
    Usage:
        def test_my_model(server_factory):
            server = server_factory("my/model", model_type="tiny_model")
            # or
            server = server_factory("my/model", server_args=["--custom", "args"])
    """
    def create_server(model_name, server_args=None, model_type=None):
        server = get_dynamic_server()
        return server.get_server(model_name, server_args, model_type)
    return create_server

# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_models():
    """Clean up models after all tests complete."""
    yield
    if _dynamic_server is not None:
        _dynamic_server.cleanup()
