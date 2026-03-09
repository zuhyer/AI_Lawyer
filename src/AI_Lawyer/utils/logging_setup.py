"""Central logging configuration for AI_Lawyer.

This file now contains the fully featured logging setup that was
originally placed in `logging_setup_v2.py`.  The old file was simple
and inflexible; the new implementation supports JSON formatting,
rotating file handlers, and respects environment variables or
`LoggingConfig` settings when invoked via `configure_logger`.

The separate `logging_setup_v2.py` remains in the tree only as a
deprecated reference.
"""

import os
import sys
import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

_BOOTSTRAP_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
_BOOTSTRAP_FORMAT = os.environ.get("LOG_FORMAT", "text")


def _build_logger(name: str = "AI_Lawyer_Logger") -> logging.Logger:
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    level = getattr(logging, _BOOTSTRAP_LEVEL, logging.INFO)
    lg.setLevel(level)
    if _BOOTSTRAP_FORMAT.lower() == "json":
        fmt = _JSONFormatter()
    else:
        fmt = logging.Formatter(
            "[%(asctime)s: %(levelname)s: %(module)s: %(message)s]"
        )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    lg.addHandler(ch)
    log_file = os.environ.get("LOG_FILE_PATH", "logs/ai_lawyer.log")
    if log_file and log_file.lower() != "null":
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        max_bytes = int(os.environ.get("LOG_MAX_BYTES", 10_485_760))
        backup = int(os.environ.get("LOG_BACKUP_COUNT", 5))
        fh = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup)
        fh.setFormatter(fmt)
        lg.addHandler(fh)
    return lg


class _JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "module": record.module,
            "message": record.getMessage(),
        }
        if record.exc_info:
            entry["exception"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in (
                "name", "msg", "args", "levelname", "levelno", "pathname",
                "filename", "module", "exc_info", "exc_text", "stack_info",
                "lineno", "funcName", "created", "msecs", "relativeCreated",
                "thread", "threadName", "processName", "process", "message",
            ):
                if not key.startswith("_"):
                    entry[key] = val
        return json.dumps(entry, default=str)


# module-level logger object
logger = _build_logger()


