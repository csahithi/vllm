# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
from typing import Any

DIAGNOSTIC_STRING_MAX_CHARS = 256


def make_request_id_summary(request_id: str) -> dict[str, Any]:
    if len(request_id) <= DIAGNOSTIC_STRING_MAX_CHARS:
        return {"request_id": request_id}

    digest = hashlib.sha256(
        request_id.encode("utf-8", errors="surrogatepass")
    ).hexdigest()
    return {
        "request_id": f"{request_id[:160]}...{request_id[-64:]}",
        "request_id_length": len(request_id),
        "request_id_sha256": digest,
        "request_id_truncated": True,
    }
