import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv


class JsonFormatter(logging.Formatter):
    """Write structured logs as JSON lines."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=True)


@dataclass
class MT5Credentials:
    login: Optional[int] = None
    password: Optional[str] = None
    server: Optional[str] = None

    def is_complete(self) -> bool:
        return self.login is not None and bool(self.password) and bool(self.server)


def parse_bool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def parse_csv_env(value: Optional[str], default: List[str]) -> List[str]:
    if not value:
        return default
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value not in (None, "") else default


def parse_float_env(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value not in (None, "") else default


def parse_str_env(name: str, default: str) -> str:
    return os.getenv(name, default)


def load_env_file() -> None:
    load_dotenv()


def load_mt5_credentials(prefix: str = "MT5") -> MT5Credentials:
    load_env_file()
    login_raw = os.getenv(f"{prefix}_LOGIN")
    login = int(login_raw) if login_raw else None
    return MT5Credentials(
        login=login,
        password=os.getenv(f"{prefix}_PASSWORD"),
        server=os.getenv(f"{prefix}_SERVER"),
    )


def configure_logging(log_level: str, log_file: str) -> None:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    root_logger.handlers.clear()

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=5, encoding="utf-8")
    file_handler.setFormatter(JsonFormatter())

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
