"""
Regression tests for HF_TOKEN configuration visibility.

Context: a real HTTP 401 was traced to a long-running process holding a stale
HF_TOKEN in memory (settings are read once from .env at process startup and
are not hot-reloaded — see app/core/embedding_service.py's 401/403 log
message for the operational guidance this implies). Separately, a genuinely
missing HF_TOKEN previously only surfaced as an error on first ingestion
attempt. _check_hf_token() makes it visible in boot logs immediately.
See app/core/config.py.
"""
import pytest

from app.core.config import _check_hf_token


def test_missing_hf_token_warns():
    with pytest.warns(UserWarning, match="HF_TOKEN is not set"):
        _check_hf_token(None)


def test_empty_hf_token_warns():
    with pytest.warns(UserWarning, match="HF_TOKEN is not set"):
        _check_hf_token("")


def test_present_hf_token_does_not_warn(recwarn):
    _check_hf_token("hf_dummy_valid_looking_token")
    assert not any("HF_TOKEN is not set" in str(w.message) for w in recwarn.list)
