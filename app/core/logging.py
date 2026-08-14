"""Structured (JSON) logging configuration."""

import json
import logging
from datetime import UTC, datetime

# Attributes logging puts on every record. Anything else in a record's
# ``__dict__`` was passed by the caller as ``extra=`` and belongs in the output.
_RESERVED_ATTRIBUTES = frozenset(
    logging.LogRecord("", 0, "", 0, "", (), None).__dict__
) | {"asctime", "message", "taskName"}


class JsonFormatter(logging.Formatter):
    """Format log records as single-line JSON objects."""

    def format(self, record: logging.LogRecord) -> str:
        """Render ``record`` as a compact JSON string.

        Fields passed via ``extra=`` are merged into the payload, which is the
        point of structured logging: per-stage latency and counts stay
        queryable instead of being flattened into the message text.

        Args:
            record: The log record to serialize.

        Returns:
            A JSON string with timestamp, level, logger name, message and any
            caller-supplied fields.
        """
        payload: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        payload.update(
            {
                key: value
                for key, value in record.__dict__.items()
                if key not in _RESERVED_ATTRIBUTES
            }
        )
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


def configure_logging(level: str = "INFO") -> None:
    """Configure the root logger to emit structured JSON to stdout.

    Idempotent: existing handlers on the root logger are replaced so that
    repeated calls (e.g. in tests) do not duplicate log output.

    Args:
        level: Logging level name (e.g. ``"INFO"``, ``"DEBUG"``).
    """
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level.upper())
