"""Tests for structured log formatting."""

import json
import logging

from app.core.logging import JsonFormatter


def _record(**extra: object) -> logging.LogRecord:
    """Build a log record carrying the given ``extra`` fields."""
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="recognized plates",
        args=(),
        exc_info=None,
    )
    record.__dict__.update(extra)
    return record


def test_formatter_emits_standard_fields() -> None:
    """Every record carries a timestamp, level, logger name and message."""
    payload = json.loads(JsonFormatter().format(_record()))

    assert payload["level"] == "INFO"
    assert payload["logger"] == "test"
    assert payload["message"] == "recognized plates"
    assert "timestamp" in payload


def test_formatter_emits_extra_fields() -> None:
    """Fields passed via ``extra=`` reach the log line.

    Without this, per-stage latency logged by the recognize service would be
    silently dropped and the timings would never be observable.
    """
    payload = json.loads(JsonFormatter().format(_record(plates=2, total_ms=812.5)))

    assert payload["plates"] == 2
    assert payload["total_ms"] == 812.5


def test_formatter_omits_internal_record_attributes() -> None:
    """Logging's own record attributes stay out of the payload."""
    payload = json.loads(JsonFormatter().format(_record()))

    assert "pathname" not in payload
    assert "args" not in payload
    assert "msg" not in payload
