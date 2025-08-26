# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing import NamedTuple

import openai  # use the official client for correctness check
import pytest
import pytest_asyncio

# # any model with a chat template should work here
MODEL_NAME = "hmellor/tiny-random-LlamaForCausalLM"  # Tiny model for fast testing


@pytest.fixture(scope="module")
def server(server_factory):
    # Use server_factory with custom args for this specific test
    custom_args = [
        # use half precision for speed and memory savings in CI environment
        "--dtype", "float16",
        "--enforce-eager",
        "--max-model-len", "1024",  # Reduced for tiny model
        "--disable-log-stats",
        "--disable-log-requests"
    ]
    
    return server_factory(MODEL_NAME, server_args=custom_args)


@pytest_asyncio.fixture
async def client(server):
    async with server.get_async_client() as async_client:
        yield async_client


class TestCase(NamedTuple):
    model_name: str
    echo: bool


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test_case",
    [
        TestCase(model_name=MODEL_NAME, echo=True),
        TestCase(model_name=MODEL_NAME, echo=False)
    ],
)
async def test_chat_session_with_echo_and_continue_final_message(
        client: openai.AsyncOpenAI, test_case: TestCase):
    saying: str = "Here is a common saying about apple. An apple a day, keeps"
    # test echo with continue_final_message parameter
    chat_completion = await client.chat.completions.create(
        model=test_case.model_name,
        messages=[{
            "role": "user",
            "content": "tell me a common saying"
        }, {
            "role": "assistant",
            "content": saying
        }],
        extra_body={
            "echo": test_case.echo,
            "continue_final_message": True,
            "add_generation_prompt": False
        })
    assert chat_completion.id is not None
    assert len(chat_completion.choices) == 1

    choice = chat_completion.choices[0]
    assert choice.finish_reason == "stop"

    message = choice.message
    if test_case.echo:
        assert message.content is not None and saying in message.content
    else:
        assert message.content is not None and saying not in message.content
    assert message.role == "assistant"
