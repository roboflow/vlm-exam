# Copyright 2026 Roboflow, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import httpx
import pytest

from vlm_exam.providers.base import (
    call_with_retries,
    is_retryable_error,
)


def test_call_with_retries_success_first_try() -> None:
    result, stats = call_with_retries(lambda: "ok")

    assert result == "ok"
    assert stats.attempts == 1
    assert stats.transient_error_types == ()
    assert stats.inference_seconds >= 0.0


def test_call_with_retries_retries_then_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vlm_exam.providers.base.time.sleep", lambda _: None)
    calls = {"count": 0}

    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ReadTimeout("stall")
        return "recovered"

    result, stats = call_with_retries(flaky)

    assert result == "recovered"
    assert calls["count"] == 3
    assert stats.attempts == 3
    assert stats.transient_error_types == ("ReadTimeout", "ReadTimeout")


def test_call_with_retries_propagates_non_retryable() -> None:
    def broken() -> str:
        raise ValueError("bad request")

    with pytest.raises(ValueError, match="bad request"):
        call_with_retries(broken)


def test_call_with_retries_exhausts_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("vlm_exam.providers.base.time.sleep", lambda _: None)

    def always_times_out() -> str:
        raise httpx.ConnectTimeout("stall")

    with pytest.raises(httpx.ConnectTimeout):
        call_with_retries(always_times_out)


def test_is_retryable_error_classification() -> None:
    class _Overloaded(Exception):
        status_code = 529

    class _GenaiServerError(Exception):
        code = 503

    assert is_retryable_error(httpx.ReadTimeout("x"))
    assert is_retryable_error(_Overloaded())
    assert is_retryable_error(_GenaiServerError())
    assert not is_retryable_error(ValueError("nope"))
