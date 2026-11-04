# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math

import numpy as np
import pytest
import torch

pytest.importorskip("cutlass")

from vllm.model_executor.layers.attention.sparse_mla_attention import (
    _build_topk_mask,
    _scatter_topk_kernel,
    _scatter_topk_single_req_kernel,
    _topk_mask_shape,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_topk_mask_single_request_matches_generic_path() -> None:
    topk = torch.tensor(
        [[0, 31, 32, 63, -1], [1, 64, 127, -1, -1]],
        dtype=torch.int32,
        device="cuda",
    )

    num_words = (128 + 31) // 32
    single_out = torch.zeros(1, 2, num_words, dtype=torch.int32, device="cuda")
    generic_out = torch.zeros(2, 1, num_words, dtype=torch.int32, device="cuda")
    single_req = _build_topk_mask([topk], [2], 2, 128, single_out)
    generic = _build_topk_mask([topk[:1], topk[1:]], [1, 1], 1, 128, generic_out)

    torch.testing.assert_close(single_req[0, 0], generic[0, 0])
    torch.testing.assert_close(single_req[0, 1], generic[1, 0])


def test_topk_mask_rows_are_aligned_for_vectorized_loads() -> None:
    assert _topk_mask_shape(2, 129, 129) == (2, 256, 8)
    assert _topk_mask_shape(2, 129, 129, reserve_key_starts_word=True) == (
        2,
        256,
        8,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_topk_mask_preserves_aligned_row_storage() -> None:
    shape = _topk_mask_shape(1, 1, 129)
    out = torch.zeros(shape, dtype=torch.int32, device="cuda")
    topk = torch.tensor([[0, 128]], dtype=torch.int32, device="cuda")

    mask = _build_topk_mask([topk], [1], 1, 129, out)

    assert mask.shape == (1, 1, 8)
    assert mask[0, 0].tolist() == [1, 0, 0, 0, 1, 0, 0, 0]


def _causal_topk(
    q_len: int, seq_len: int, num_topk: int, generator: torch.Generator
) -> torch.Tensor:
    """Random top-k indices for the last `q_len` queries of a sequence.

    Each query selects only keys up to its own position; its unused slots hold
    -1 and are interleaved with the valid ones.
    """
    positions = torch.arange(seq_len - q_len, seq_len)
    scores = torch.rand(q_len, seq_len, generator=generator)
    scores[torch.arange(seq_len) > positions[:, None]] = -1.0
    num_selected = min(num_topk, seq_len)
    values, indices = scores.topk(num_selected, dim=-1)
    topk = torch.full((q_len, num_topk), -1, dtype=torch.int32)
    topk[:, :num_selected] = torch.where(values >= 0, indices, -1)
    order = torch.rand(q_len, num_topk, generator=generator).argsort(dim=-1)
    return topk.gather(1, order)


def _reference_build_topk_mask(
    topk_per_req: list[torch.Tensor],
    q_lens: list[int],
    max_q_len: int,
    max_seq_len: int,
    out: torch.Tensor,
) -> torch.Tensor:
    """What `_build_topk_mask` leaves in `out`, computed without its kernels.

    Row (request, query position) is a bitmap over the keys: key `k` is bit
    `k % 32` of int32 word `k // 32`, as the masked-MHA mask_mods read it. The
    first `cdiv(max_seq_len, 32)` words of every row are rewritten, so rows past
    a request's query length come out empty, while the rest of each row, such as
    the reserved key-start word, keeps its contents.
    """
    num_words = (max_seq_len + 31) // 32
    selected = np.zeros((len(q_lens), max_q_len, num_words * 32), dtype=bool)
    for req, (topk, q_len) in enumerate(zip(topk_per_req, q_lens)):
        indices = topk[:q_len].cpu().numpy()
        # Negative indices are padding.
        query, slot = np.nonzero(indices >= 0)
        selected[req, query, indices[query, slot]] = True
    words = np.packbits(selected, axis=-1, bitorder="little").view("<i4")
    expected = out.to("cpu", copy=True)
    expected[: len(q_lens), :max_q_len, :num_words] = torch.from_numpy(
        words.astype(np.int32)
    )
    return expected


def _check_build_topk_mask(
    q_lens: list[int], seq_lens: list[int], num_topk: int, layout: str
) -> None:
    """Build a top-k mask in a stale workspace and compare it to the reference.

    `layout` is "padded" for the masked-MHA mask shape, "padded_key_starts" for
    the global mask, whose rows reserve a key-start word, or "unpadded_q" to give
    each request only as many rows as the longest request has queries.
    """
    generator = torch.Generator().manual_seed(0)
    topk_per_req = [
        _causal_topk(q_len, seq_len, num_topk, generator).cuda()
        for q_len, seq_len in zip(q_lens, seq_lens)
    ]
    max_seq_len = max(seq_lens)
    batch_size, max_q_len, row_words = _topk_mask_shape(
        len(q_lens),
        max(q_lens),
        max_seq_len,
        reserve_key_starts_word=layout == "padded_key_starts",
    )
    if layout == "unpadded_q":
        max_q_len = max(q_lens)
    shape = (batch_size, max_q_len, row_words)

    # The shared workspace still holds earlier masks, so start from stale words,
    # including some past the mask that must be left alone.
    needed = math.prod(shape)
    workspace = torch.randint(
        -(2**31), 2**31 - 1, (needed + 64,), dtype=torch.int32, generator=generator
    ).cuda()
    out = workspace[:needed].view(shape)
    expected = _reference_build_topk_mask(
        topk_per_req, q_lens, max_q_len, max_seq_len, out
    )
    expected_tail = workspace[needed:].cpu()

    mask = _build_topk_mask(topk_per_req, q_lens, max_q_len, max_seq_len, out)

    assert mask.shape == shape
    assert mask.data_ptr() == out.data_ptr() and mask.stride() == out.stride()
    torch.testing.assert_close(mask.cpu(), expected, rtol=0, atol=0)
    torch.testing.assert_close(workspace[needed:].cpu(), expected_tail, rtol=0, atol=0)


# Triton compiles separate variants for integer arguments equal to 1 and for
# multiples of 16. Across the cases below, num_words and max_q_len or total_q
# take values of all three kinds, and mask_row_stride, always a multiple of 4,
# both multiples of 16 and others. Most key lengths end partway through a word.
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "q_lens, seq_lens, num_topk, layout",
    [
        pytest.param(
            [37, 1, 64, 100],
            [37, 1500, 4100, 100],
            2048,
            "padded_key_starts",
            id="q128-words129-stride132",
        ),
        pytest.param(
            [37, 1, 64, 100],
            [37, 1500, 3001, 100],
            2048,
            "unpadded_q",
            id="q100-words94-stride96",
        ),
        pytest.param(
            [16, 3, 48], [1000, 600, 48], 100, "unpadded_q", id="q48-words32-stride32"
        ),
        pytest.param([1, 1, 1], [20, 7, 29], 100, "unpadded_q", id="q1-words1-stride4"),
    ],
)
def test_build_topk_mask_multi_request_matches_reference(
    q_lens: list[int], seq_lens: list[int], num_topk: int, layout: str
) -> None:
    _check_build_topk_mask(q_lens, seq_lens, num_topk, layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "q_len, seq_len, num_topk, layout",
    [
        pytest.param(37, 3001, 2048, "padded", id="total37-words94-stride96"),
        pytest.param(
            48, 4096, 2048, "padded_key_starts", id="total48-words128-stride132"
        ),
        pytest.param(1, 129, 100, "padded", id="total1-words5-stride8"),
        pytest.param(5, 20, 100, "padded", id="total5-words1-stride4"),
        # 300 of 512 rows hold queries.
        pytest.param(300, 5000, 2048, "padded", id="total300-words157-stride160"),
    ],
)
def test_build_topk_mask_single_request_matches_reference(
    q_len: int, seq_len: int, num_topk: int, layout: str
) -> None:
    _check_build_topk_mask([q_len], [seq_len], num_topk, layout)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_build_topk_mask_reuses_jit_across_lengths() -> None:
    """A new prompt or chunk length must not compile new mask kernels."""

    def compiled_variants() -> set:
        return {
            (kernel.fn.__name__, key)
            for kernel in (_scatter_topk_kernel, _scatter_topk_single_req_kernel)
            for caches in kernel.device_caches.values()
            for key in caches[0]
        }

    _check_build_topk_mask([37, 1, 64, 100], [37, 1500, 3001, 100], 2048, "unpadded_q")
    _check_build_topk_mask([37], [3001], 2048, "padded")
    warmed = compiled_variants()

    # New values of num_words (94 -> 93) and of max_q_len or total_q (100 or
    # 37 -> 45), with the same BLOCK_WORDS and integer specializations.
    _check_build_topk_mask([45, 7], [45, 2970], 2048, "unpadded_q")
    _check_build_topk_mask([45], [2970], 2048, "padded")
    assert compiled_variants() == warmed
