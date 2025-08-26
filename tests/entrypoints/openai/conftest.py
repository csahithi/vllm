# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import pytest
import tempfile
import shutil
from pathlib import Path
from ...utils import RemoteOpenAIServer

# Default server arguments for common configurations (memory-optimized)
DEFAULT_ARGS = {
    "chat": ["--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "32", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests", "--disable-log-errors", "--max-paddings", "32"],
    "embedding": ["--runner", "pooling", "--dtype", "bfloat16", "--enforce-eager", "--max-model-len", "512", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests", "--disable-log-errors", "--max-paddings", "16"],
    "vision": ["--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"image": 2}', "--disable-log-stats", "--disable-log-requests", "--disable-log-errors"],
    "audio": ["--dtype", "float32", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"audio": 2}', "--disable-log-stats", "--disable-log-requests", "--disable-log-errors"],
    "video": ["--runner", "generate", "--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "4", "--gpu-memory-utilization", "0.7", "--trust-remote-code", "--limit-mm-per-prompt", '{"video": 1}', "--disable-log-stats", "--disable-log-requests", "--disable-log-errors"],
    "multimodal": ["--dtype", "bfloat16", "--max-model-len", "1024", "--enforce-eager", "--max-num-seqs", "16", "--gpu-memory-utilization", "0.7", "--disable-log-stats", "--disable-log-requests", "--disable-log-errors", "--max-paddings", "8"]
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
        # Fallback to original model name
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
                # Default to chat args for unknown model types
                args = DEFAULT_ARGS["chat"].copy()
            else:
                args = server_args.copy()
            
            # Add memory optimization flags
            memory_flags = [
                "--gpu-memory-utilization", "0.7",  # Use only 70% of GPU memory
                "--max-model-len", "1024",  # Reduce from 2048 to save memory
                "--max-num-seqs", "32",     # Reduce from 64 to save memory
                "--disable-log-stats",
                "--disable-log-requests",
                "--disable-log-errors"
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

# Backward compatibility fixtures
@pytest.fixture(scope="module")
def server(server_factory):
    """Backward compatibility - creates server with default chat model."""
    return server_factory("hmellor/tiny-random-LlamaForCausalLM", model_type="chat")

@pytest.fixture(scope="module")
def embedding_server(server_factory):
    """Backward compatibility - creates server with default embedding model."""
    return server_factory("intfloat/multilingual-e5-small", model_type="embedding")

# Cleanup fixture
@pytest.fixture(scope="session", autouse=True)
def cleanup_models():
    """Clean up models after all tests complete."""
    yield
    if _dynamic_server is not None:
        _dynamic_server.cleanup()
