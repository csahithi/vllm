# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio
from huggingface_hub import snapshot_download

from ...conftest import AudioTestAssets
from ...utils import RemoteOpenAIServer

# NOTE - the tests in this module are currently analogous to test_chat, but are
# separated to avoid OOM killing due to module-scoped servers, since we
# need a multimodal model for these tests.

# Contains a modality specific lora alongside the base model
MULTIMODAL_MODEL_NAME = "microsoft/Phi-4-multimodal-instruct"  # Use model name for server
MULTIMODAL_MODEL_PATH = snapshot_download(MULTIMODAL_MODEL_NAME)  # Use local path for LoRA
AUDIO_LORA_PATH = os.path.join(MULTIMODAL_MODEL_PATH, "speech-lora")

ACTIVE_MM_LORA_RESPONSE = "Certainly! Here is the transcription of the provided audio:\n\nThe first words I spoke in the original phonograph record: A little piece of practical poetry. Mary had a little lamb, its fleece was white as snow, and everywhere that Mary went, the lamb was sure to go."


@pytest.fixture(scope="module")
def multimodal_server():
    """Test-specific server for this test with proper LoRA configuration"""
    # Get the model path for LoRA configuration
    model_path = snapshot_download("microsoft/Phi-4-multimodal-instruct")
    audio_lora_path = os.path.join(model_path, "speech-lora")
    
    args = [
        "--dtype", "half",  # use half precision for speed and memory savings in CI environment
        "--max-model-len", "12800",
        "--enforce-eager",
        # lora config below
        "--enable-lora",
        "--lora-modules", f"speech={audio_lora_path}",
        "--max-lora-rank", "320",
        "--max-num-seqs", "2",
        "--trust-remote-code",
        "--gpu-memory-utilization", "0.8",
        "--default-mm-loras", f'{{"audio": "{audio_lora_path}"}}',
    ]
    
    with RemoteOpenAIServer(MULTIMODAL_MODEL_NAME, args, max_wait_seconds=180) as server:
        yield server


@pytest_asyncio.fixture
async def multi_modal_client(multimodal_server):
    async with multimodal_server.get_async_client() as async_client:
        yield async_client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    # base model with default lora should give the same response as lora model
    "model_name",
    [MULTIMODAL_MODEL_NAME, "speech"],
)
async def test_default_mm_lora_chat_completions(
    model_name: str,
    multi_modal_client: openai.AsyncOpenAI,
    audio_assets: AudioTestAssets,
):
    messages = [{
        "role":
        "user",
        "content": [{
            "type": "text",
            "text": "Can you transcribe this audio?",
        }, {
            "type": "audio_url",
            "audio_url": {
                "url": audio_assets[0].url
            },
        }]
    }]

    chat_completion = await multi_modal_client.chat.completions.create(
        model=model_name,
        messages=messages,
        max_completion_tokens=128,
        temperature=0.0)

    assert len(chat_completion.choices) > 0

    message = chat_completion.choices[0].message
    assert message.content is not None and len(message.content) >= 0
    assert message.content == ACTIVE_MM_LORA_RESPONSE
