"""Centralized logging configuration for the api-doc-fraud service.

Usage in any module:

    from service.logging_config import get_logger
    log = get_logger(__name__)
    log.info("processing %s", path)
    log.warning("template missing — falling back")
    log.error("VLM call failed: %s", err)

Configuration is applied once via configure_logging(), called from main.py
on app startup. The log level honors the LOG_LEVEL env var (default INFO).
LOG_FORMAT=json enables JSON output for production log aggregation; otherwise
human-friendly text format is used."""

from __future__ import annotations

import json
import logging
import os
import sys
from typing import Any


_LOG_LEVEL  = os.getenv("LOG_LEVEL", "INFO").upper()
_LOG_FORMAT = os.getenv("LOG_FORMAT", "text").lower()

_TEXT_FORMAT = "%(asctime)s [%(levelname)8s] %(name)s:%(lineno)d - %(message)s"


class _JsonFormatter(logging.Formatter):
    """Minimal JSON formatter — no extra deps. Compatible with most log
    aggregators (Datadog, ELK, CloudWatch)."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts":      self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level":   record.levelname,
            "logger":  record.name,
            "msg":     record.getMessage(),
            "line":    record.lineno,
            "module":  record.module,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        # Allow extra fields via log.info("msg", extra={"request_id": ...})
        for key, value in record.__dict__.items():
            if key in payload or key.startswith("_") or key in {
                "args", "asctime", "created", "exc_info", "exc_text",
                "filename", "funcName", "levelname", "levelno", "lineno",
                "message", "module", "msecs", "msg", "name", "pathname",
                "process", "processName", "relativeCreated", "stack_info",
                "thread", "threadName", "taskName",
            }:
                continue
            payload[key] = value
        return json.dumps(payload, ensure_ascii=False)


_configured = False


def configure_logging(level: str | None = None, fmt: str | None = None) -> None:
    """Idempotent root logger setup. Call once at process startup.

    level: 'DEBUG' / 'INFO' / 'WARNING' / 'ERROR'. Falls back to LOG_LEVEL env.
    fmt:   'text' (default) or 'json'. Falls back to LOG_FORMAT env."""
    global _configured
    if _configured:
        return

    effective_level = (level or _LOG_LEVEL).upper()
    effective_fmt   = (fmt or _LOG_FORMAT).lower()

    root = logging.getLogger()
    root.setLevel(effective_level)

    # Replace any pre-existing handlers (uvicorn or python defaults) so we get
    # a consistent format across the whole process.
    for h in list(root.handlers):
        root.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if effective_fmt == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(logging.Formatter(_TEXT_FORMAT))
    root.addHandler(handler)

    # Tame noisy third-party libs at INFO+.
    for noisy in ("urllib3", "httpx", "httpcore", "PIL", "paddle"):
        logging.getLogger(noisy).setLevel(max(logging.WARNING, root.level))

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger; ensure configure_logging has been called at least once."""
    if not _configured:
        configure_logging()
    return logging.getLogger(name)
