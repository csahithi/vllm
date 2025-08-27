# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from tempfile import TemporaryDirectory

import httpx
import pytest

from vllm.version import __version__ as VLLM_VERSION

# RemoteOpenAIServer import removed - now using conftest fixtures

MODEL_NAME = "microsoft/DialoGPT-small"  # Compatible model for testing


# Use the conftest server fixture instead of local implementation
# The server fixture is now session-scoped and shared across all tests
# Note: This test requires custom server args for UDS testing
# We'll use the server_factory fixture for this specific case


@pytest.mark.asyncio
async def test_show_version(server_factory):
    with TemporaryDirectory() as tmpdir:
        # Create server with custom UDS configuration
        custom_args = [
            "--dtype", "bfloat16",
            "--max-model-len", "8192",
            "--enforce-eager",
            "--max-num-seqs", "128",
            "--uds", f"{tmpdir}/vllm.sock",
        ]
        
        # Use server_factory to create server with custom args
        with server_factory(MODEL_NAME, custom_args) as custom_server:
            transport = httpx.HTTPTransport(uds=custom_server.uds)
            client = httpx.Client(transport=transport)
            response = client.get(custom_server.url_for("version"))
            response.raise_for_status()

            assert response.json() == {"version": VLLM_VERSION}
