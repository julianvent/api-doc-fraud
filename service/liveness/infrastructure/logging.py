"""Structured logging for the liveness module.

Two design choices: (1) stdlib `logging` only — no new dependency; a
deployment can swap in a JSON formatter without touching call sites.
(2) Records from inside a request carry `request_id` in `extra`; the
orchestrator wraps its logger in a LoggerAdapter for `run()` so the id
need not be threaded explicitly.

Usage:

    from service.liveness.infrastructure.logging import get_logger
    log = get_logger(__name__)
    log.info("face.located", extra={"width": 224, "height": 224})

Set env `LIVENESS_LOG_FORMAT=json` before the first `configure()` for
one-JSON-object-per-line output.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


_LOGGER_PREFIX = "liveness"
_CONFIGURED = False


class _JsonFormatter(logging.Formatter):
    """Emits one JSON object per log record.

    Standard fields use reserved keys (`ts`, `level`, `logger`, `event`,
    `msg`); `extra={...}` is merged at top level so indexers can query
    by e.g. `request_id`.
    """

    _RESERVED = {
        "args", "asctime", "created", "exc_info", "exc_text", "filename",
        "funcName", "levelname", "levelno", "lineno", "message", "module",
        "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "thread", "threadName",
        "taskName",
    }

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key in self._RESERVED or key.startswith("_"):
                continue
            payload[key] = _coerce_jsonable(value)
        return json.dumps(payload, default=str)


def _coerce_jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _coerce_jsonable(v) for k, v in value.items()}
    return str(value)


def configure(*, level: int | None = None, fmt: str | None = None) -> None:
    """Attach a handler to the `liveness` root logger (idempotent).

    Resolution is argument → env var → default:
      level: LIVENESS_LOG_LEVEL (DEBUG/INFO/WARNING/ERROR) | INFO
      fmt:   LIVENESS_LOG_FORMAT (text/json)               | text

    Configures only once and only the `liveness` namespace (never the
    root logger), so host apps can manage their own logging stack.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    resolved_level = level
    if resolved_level is None:
        env_level = os.environ.get("LIVENESS_LOG_LEVEL", "INFO").upper()
        resolved_level = getattr(logging, env_level, logging.INFO)

    resolved_fmt = (fmt or os.environ.get("LIVENESS_LOG_FORMAT", "text")).lower()

    handler = logging.StreamHandler(stream=sys.stderr)
    if resolved_fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s %(levelname)-5s %(name)s %(message)s",
                datefmt="%H:%M:%S",
            )
        )

    logger = logging.getLogger(_LOGGER_PREFIX)
    logger.setLevel(resolved_level)
    logger.addHandler(handler)
    # Avoid duplicate emission if the host app configured the root logger.
    logger.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a logger under the `liveness` namespace.

    `name` is typically `__name__`. Used as-is if it already starts with
    `liveness.`, otherwise prefixed.
    """
    configure()
    if name == _LOGGER_PREFIX or name.startswith(_LOGGER_PREFIX + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{_LOGGER_PREFIX}.{name}")


class _MergingAdapter(logging.LoggerAdapter):
    """LoggerAdapter that MERGES `extra` instead of overwriting it.

    The stdlib default replaces `kwargs['extra']` with `self.extra`,
    dropping per-call fields. We keep the adapter's `request_id` PLUS
    whatever the call site passed.
    """

    def process(self, msg, kwargs):
        call_extra = kwargs.get("extra") or {}
        merged = {**self.extra, **call_extra}
        kwargs["extra"] = merged
        return msg, kwargs


def request_logger(base: logging.Logger, request_id: str) -> logging.LoggerAdapter:
    """Bind a request_id to every record emitted through the adapter.

    Use inside `Orchestrator.run()` so each pipeline log carries
    `request_id=<uuid>` without threading it through every call. Per-call
    `extra={...}` fields are preserved (see `_MergingAdapter`).
    """
    return _MergingAdapter(base, extra={"request_id": request_id})
