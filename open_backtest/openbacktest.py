#!/usr/bin/env python3
"""OpenBacktest.ai — AI-powered cryptocurrency backtesting platform.

Single-file application with:
  - OpenRouter AI (only provider)
  - PySide6 UI
  - Plotly visualization (via QWebEngineView)
  - CCXT market data
  - Polars data processing
  - TA-Lib indicators
  - Vectorized backtest engine (NautilusTrader optional)
  - Central orchestrator architecture

Launch:  python openbacktest.py

Author: Kowsigan R | AI & Data Science
"""
from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════
# STANDARD LIBRARY
# ═══════════════════════════════════════════════════════════════════
import ast
import base64
import hashlib
import json
import logging
import math
import os
import queue
import re
import shutil
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Callable, Generator

# ═══════════════════════════════════════════════════════════════════
# THIRD-PARTY (always required)
# ═══════════════════════════════════════════════════════════════════
import numpy as np
import polars as pl
import requests as http_requests

# ═══════════════════════════════════════════════════════════════════
# CONDITIONAL IMPORTS
# ═══════════════════════════════════════════════════════════════════
try:
    import ccxt as _ccxt
    HAS_CCXT = True
except ImportError:
    HAS_CCXT = False

try:
    import talib as _talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False

try:
    import plotly.graph_objects as go
    import plotly.io as pio
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

try:
    from cryptography.fernet import Fernet
    HAS_CRYPTO = True
except ImportError:
    HAS_CRYPTO = False

try:
    from scipy import stats as scipy_stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ═══════════════════════════════════════════════════════════════════
# SECTION 1 — CONSTANTS & CONFIGURATION
# ═══════════════════════════════════════════════════════════════════

APP_NAME = "OpenBacktest.ai"
APP_VERSION = "1.0.0"
APP_TITLE = f"{APP_NAME} v{APP_VERSION}"
APP_CREATOR = "Kowsigan R | AI & Data Science"

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"
CONFIG_DIR = PROJECT_ROOT / "config"
for _d in (DATA_DIR, RESULTS_DIR, LOGS_DIR, CONFIG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# OpenRouter
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
OPENROUTER_CHAT_URL = f"{OPENROUTER_BASE_URL}/chat/completions"
DEFAULT_MODEL = "anthropic/claude-sonnet-5"

# Defaults
DEFAULT_ASSETS: list[str] = ["BTC/USDT", "ETH/USDT", "DOGE/USDT"]
DEFAULT_TIMEFRAME = "1d"
DEFAULT_EXCHANGE = "binance"
DEFAULT_INITIAL_CAPITAL = 10_000.0
DEFAULT_POSITION_SIZE_PCT = 0.90
DEFAULT_MAKER_FEE_PCT = 0.001
DEFAULT_TAKER_FEE_PCT = 0.001
DEFAULT_SLIPPAGE_PCT = 0.0005

SUPPORTED_TIMEFRAMES = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]
SUPPORTED_EXCHANGES = ["binance", "bybit", "coinbase", "kraken", "okx", "bitget"]

SUPPORTED_INDICATORS = [
    {"name": "RSI", "default_period": 14, "source": "close"},
    {"name": "EMA", "default_period": 20, "source": "close"},
    {"name": "SMA", "default_period": 20, "source": "close"},
    {"name": "MACD", "default_period": 12, "source": "close"},
    {"name": "ATR", "default_period": 14, "source": "hlc"},
    {"name": "ADX", "default_period": 14, "source": "hlc"},
    {"name": "CCI", "default_period": 20, "source": "hlc"},
    {"name": "BBANDS", "default_period": 20, "source": "close"},
    {"name": "STOCH", "default_period": 14, "source": "hlc"},
    {"name": "ROC", "default_period": 10, "source": "close"},
    {"name": "MFI", "default_period": 14, "source": "hlcv"},
    {"name": "OBV", "default_period": 0, "source": "close_volume"},
]

CCXT_OHLCV_LIMIT = 1000
CCXT_MAX_RETRIES = 3
CCXT_RETRY_DELAY = 2.0

LOG_FORMAT = "%(asctime)s  %(name)-24s  %(levelname)-8s  %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
MAX_LOG_FILE_SIZE_MB = 10
LOG_BACKUP_COUNT = 3

MAX_AI_REPAIR_ATTEMPTS = 3

# Security — forbidden patterns in AI-generated code
FORBIDDEN_PATTERNS = [
    r"\bos\.system\b", r"\bsubprocess\b", r"\b__import__\b",
    r"\beval\s*\(", r"\bexec\s*\(", r"\bcompile\s*\(",
    r"\bopen\s*\(", r"\bglobals\s*\(", r"\blocals\s*\(",
    r"\bgetattr\s*\(", r"\bsetattr\s*\(", r"\bdelattr\s*\(",
    r"\bbreakpoint\s*\(", r"\bexit\s*\(", r"\bquit\s*\(",
    r"\brequests\b", r"\burllib\b", r"\bsocket\b",
    r"\bshutil\b", r"\bpickle\b", r"\bshelve\b",
]

ALLOWED_IMPORTS = {"numpy", "np", "math", "datetime", "collections"}


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 — LOGGING & SECURITY
# ═══════════════════════════════════════════════════════════════════

class SensitiveDataFilter(logging.Filter):
    """Redacts API keys and secrets from log output."""
    _patterns = [
        re.compile(r"(sk-[a-zA-Z0-9]{20,})"),
        re.compile(r"(AIza[a-zA-Z0-9_-]{35})"),
        re.compile(r"(sk-ant-[a-zA-Z0-9_-]{20,})"),
        re.compile(r"(sk-or-[a-zA-Z0-9_-]{20,})"),
        re.compile(r"(api[_-]?key\s*[:=]\s*['\"]?)(\S+)", re.IGNORECASE),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        if isinstance(record.msg, str):
            for p in self._patterns:
                record.msg = p.sub("[REDACTED]", record.msg)
        return True


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application-wide logging."""
    logger = logging.getLogger("openbacktest")
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False
    sf = SensitiveDataFilter()
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    ch.addFilter(sf)
    logger.addHandler(ch)
    try:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d")
        fh = RotatingFileHandler(
            LOGS_DIR / f"openbacktest_{ts}.log",
            maxBytes=MAX_LOG_FILE_SIZE_MB * 1024 * 1024,
            backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        fh.addFilter(sf)
        logger.addHandler(fh)
    except Exception:
        pass
    return logger


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"openbacktest.{name}")


# Initialize global logger
_root_logger = setup_logging()
_log = get_logger("core")


class KeyStore:
    """Encrypted local storage for the OpenRouter API key."""

    def __init__(self) -> None:
        self._path = CONFIG_DIR / "credentials.enc"
        self._key_path = CONFIG_DIR / ".keyfile"

    def _get_fernet(self) -> Any:
        if not HAS_CRYPTO:
            return None
        if self._key_path.exists():
            key = self._key_path.read_bytes()
        else:
            key = Fernet.generate_key()
            self._key_path.write_bytes(key)
        return Fernet(key)

    def _load_data(self) -> dict:
        if not self._path.exists():
            return {"active": None, "keys": {}}
        try:
            f = self._get_fernet()
            if f:
                content = f.decrypt(self._path.read_bytes()).decode()
            else:
                content = base64.b64decode(self._path.read_text()).decode()
            try:
                data = json.loads(content)
                if isinstance(data, dict) and "keys" in data:
                    return data
            except json.JSONDecodeError:
                if content.strip():
                    return {"active": "Default Key", "keys": {"Default Key": content.strip()}}
        except Exception:
            pass
        return {"active": None, "keys": {}}

    def _save_data(self, data: dict) -> None:
        content = json.dumps(data)
        f = self._get_fernet()
        if f:
            self._path.write_bytes(f.encrypt(content.encode()))
        else:
            self._path.write_text(base64.b64encode(content.encode()).decode())

    def save_key(self, api_key: str, name: str = "Default Key") -> None:
        data = self._load_data()
        data["keys"][name] = api_key
        data["active"] = name
        self._save_data(data)

    def load_key(self) -> str:
        data = self._load_data()
        active = data.get("active")
        return data["keys"].get(active, "") if active else ""

    def get_all_keys(self) -> dict:
        return self._load_data().get("keys", {})

    def set_active_key(self, name: str) -> None:
        data = self._load_data()
        if name in data.get("keys", {}):
            data["active"] = name
            self._save_data(data)

    def delete_key(self) -> None:
        if self._path.exists():
            self._path.unlink()


_key_store = KeyStore()


def get_key_store() -> KeyStore:
    return _key_store


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 — DATA MODELS & ENUMS
# ═══════════════════════════════════════════════════════════════════

class Stage(Enum):
    INITIALIZING = "initializing"
    CHECKING_API = "checking_api"
    DISCOVERING_MARKETS = "discovering_markets"
    DOWNLOADING = "downloading"
    VALIDATING_DATA = "validating_data"
    INTERPRETING_STRATEGY = "interpreting_strategy"
    VALIDATING_CODE = "validating_code"
    RUNNING_BACKTEST = "running_backtest"
    ANALYZING = "analyzing"
    GENERATING_CHARTS = "generating_charts"
    SAVING_RESULTS = "saving_results"
    COMPLETE = "complete"
    FAILED = "failed"
    CANCELLED = "cancelled"


class EventType(Enum):
    PROGRESS = "progress"
    RUN_STARTED = "run_started"
    RUN_COMPLETE = "run_complete"
    RUN_FAILED = "run_failed"
    RUN_CANCELLED = "run_cancelled"
    AI_COMPLETE = "ai_complete"
    AI_ERROR = "ai_error"


class UserIntent(Enum):
    STRATEGY_CHANGE = "strategy_change"
    CONFIG_CHANGE = "config_change"
    ANALYTICAL_QUERY = "analytical_query"
    GENERAL_CHAT = "general_chat"


@dataclass
class FeeConfig:
    maker_fee_pct: float = DEFAULT_MAKER_FEE_PCT
    taker_fee_pct: float = DEFAULT_TAKER_FEE_PCT


@dataclass
class SlippageConfig:
    slippage_pct: float = DEFAULT_SLIPPAGE_PCT


@dataclass
class BacktestConfig:
    assets: list[str] = field(default_factory=lambda: list(DEFAULT_ASSETS))
    exchange: str = DEFAULT_EXCHANGE
    timeframe: str = DEFAULT_TIMEFRAME
    initial_capital: float = DEFAULT_INITIAL_CAPITAL
    position_size_pct: float = DEFAULT_POSITION_SIZE_PCT
    fee_config: FeeConfig = field(default_factory=FeeConfig)
    slippage_config: SlippageConfig = field(default_factory=SlippageConfig)
    strategy_text: str = ""
    start_date: Any = None
    end_date: Any = None

    def config_hash(self) -> str:
        d = {
            "assets": sorted(self.assets), "exchange": self.exchange,
            "timeframe": self.timeframe, "capital": self.initial_capital,
            "pos_pct": self.position_size_pct, "strategy": self.strategy_text,
        }
        return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


@dataclass
class TradeRecord:
    trade_id: int = 0
    asset: str = ""
    side: str = "LONG"
    entry_time: str = ""
    entry_price: float = 0.0
    exit_time: str = ""
    exit_price: float = 0.0
    quantity: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class MetricsReport:
    initial_capital: float = 0.0
    final_value: float = 0.0
    net_pnl: float = 0.0
    return_pct: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    avg_trade_pnl: float = 0.0
    best_trade: float = 0.0
    worst_trade: float = 0.0
    total_fees: float = 0.0
    total_slippage: float = 0.0
    total_charges: float = 0.0
    avg_holding_time: str = ""
    expectancy: float = 0.0


@dataclass
class BacktestResult:
    run_id: str = ""
    config: BacktestConfig = field(default_factory=BacktestConfig)
    metrics: MetricsReport = field(default_factory=MetricsReport)
    trades: list[TradeRecord] = field(default_factory=list)
    equity_curve: list[dict] = field(default_factory=list)
    run_directory: str = ""
    strategy_hash: str = ""
    code_hash: str = ""
    files: list[str] = field(default_factory=list)
    datasets: dict[str, Any] = field(default_factory=dict)


@dataclass
class ChatMessage:
    role: str = "user"
    content: str = ""
    metrics: MetricsReport | None = None
    charts: list[Any] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class AppConfig:
    model: str = DEFAULT_MODEL
    exchange: str = DEFAULT_EXCHANGE
    api_key: str = ""

    def save(self) -> None:
        d = {"model": self.model, "exchange": self.exchange}
        (CONFIG_DIR / "app_config.json").write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls) -> "AppConfig":
        p = CONFIG_DIR / "app_config.json"
        cfg = cls()
        if p.exists():
            try:
                d = json.loads(p.read_text())
                cfg.model = d.get("model", DEFAULT_MODEL)
                cfg.exchange = d.get("exchange", DEFAULT_EXCHANGE)
            except Exception:
                pass
        cfg.api_key = get_key_store().load_key()
        return cfg


# ═══════════════════════════════════════════════════════════════════
# SECTION 4 — OPENROUTER CLIENT
# ═══════════════════════════════════════════════════════════════════

class OpenRouterClient:
    """OpenRouter API client using OpenAI-compatible endpoints."""

    def __init__(self, api_key: str, model: str = DEFAULT_MODEL) -> None:
        self._api_key = api_key
        self.model = model
        self._log = get_logger("openrouter")
        self._session = http_requests.Session()
        self._session.headers.update({
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://openbacktest.ai",
            "X-Title": APP_NAME,
        })

    def validate_connection(self) -> tuple[bool, str]:
        """Send a minimal request to verify the API key works."""
        try:
            resp = self._session.post(
                OPENROUTER_CHAT_URL,
                json={
                    "model": self.model,
                    "messages": [{"role": "user", "content": "Reply with OK"}],
                    "max_tokens": 5,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                if data.get("choices"):
                    return True, "OpenRouter connected"
                return False, "No response from model"
            return False, self._parse_error(resp)
        except http_requests.exceptions.Timeout:
            return False, "Connection timed out"
        except http_requests.exceptions.ConnectionError:
            return False, "Could not reach OpenRouter servers"
        except Exception as e:
            return False, f"Connection error: {e}"

    def chat_completion(
        self, messages: list[dict], temperature: float = 0.3,
        max_tokens: int = 2048,
    ) -> str:
        """Non-streaming chat completion."""
        try:
            resp = self._session.post(
                OPENROUTER_CHAT_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                },
                timeout=120,
            )
            if resp.status_code != 200:
                raise RuntimeError(self._parse_error(resp))
            data = resp.json()
            choices = data.get("choices", [])
            if not choices:
                raise RuntimeError("Empty response from OpenRouter")
            return choices[0]["message"]["content"]
        except http_requests.exceptions.Timeout:
            raise RuntimeError("OpenRouter request timed out. Please try again.")
        except http_requests.exceptions.ConnectionError:
            raise RuntimeError("Cannot reach OpenRouter. Check your internet connection.")

    def chat_completion_stream(
        self, messages: list[dict], temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> Generator[str, None, None]:
        """Streaming chat completion via SSE."""
        try:
            resp = self._session.post(
                OPENROUTER_CHAT_URL,
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "stream": True,
                },
                timeout=120,
                stream=True,
            )
            if resp.status_code != 200:
                raise RuntimeError(self._parse_error(resp))
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                    delta = chunk.get("choices", [{}])[0].get("delta", {})
                    content = delta.get("content", "")
                    if content:
                        yield content
                except json.JSONDecodeError:
                    continue
        except http_requests.exceptions.Timeout:
            raise RuntimeError("Streaming request timed out.")
        except http_requests.exceptions.ConnectionError:
            raise RuntimeError("Connection lost during streaming.")

    def generate_strategy_code(
        self, strategy_text: str, master_template: str, context: dict,
    ) -> str:
        """Generate strategy code from user description."""
        system_prompt = textwrap.dedent(f"""\
        You are an expert quantitative trading strategy developer.

        The user will describe a trading strategy in plain English.
        You must generate a Python class called `UserStrategy` that implements it.

        RULES:
        1. Return ONLY the Python source code. No markdown, no explanation.
        2. The class MUST have this exact structure:
           - __init__(self) that sets self.indicators (list of dicts)
           - should_enter(self, idx, row, indicators) -> bool
           - should_exit(self, idx, row, indicators) -> bool
        3. `row` is a dict with keys: open, high, low, close, volume, timestamp
        4. `indicators` is a dict mapping indicator names to numpy arrays
           e.g. indicators["RSI_14"], indicators["EMA_50"], indicators["SMA_20"]
        5. `idx` is the current bar index (int)
        6. Only use: numpy, math, datetime, collections
        7. Do NOT import os, subprocess, requests, or any I/O modules
        8. Do NOT use eval, exec, open, or any dangerous functions
        9. If the request is NOT a valid trading strategy, return exactly: NA

        Available indicators: {json.dumps([i["name"] for i in SUPPORTED_INDICATORS])}
        Indicator format in self.indicators: {{"name": "RSI", "period": 14, "source": "close"}}

        Context:
        - Assets: {context.get("assets", [])}
        - Timeframe: {context.get("timeframe", "1h")}
        - Initial capital: ${context.get("initial_capital", 10000):,.0f}
        - Position size: {context.get("position_size_pct", 0.9) * 100:.0f}%
        """)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": strategy_text},
        ]
        return self.chat_completion(messages, temperature=0.2, max_tokens=2048)

    def repair_code(self, code: str, errors: list[str], attempt: int) -> str:
        """Ask AI to repair code that failed validation."""
        messages = [
            {"role": "system", "content": textwrap.dedent("""\
                You are fixing a Python UserStrategy class that failed validation.
                Return ONLY the corrected Python source code. No markdown, no explanation.
                The class must have: __init__, should_enter, should_exit methods.
                Do not use forbidden functions (os, subprocess, eval, exec, open, etc).
            """)},
            {"role": "user", "content": f"Code:\n```\n{code}\n```\n\nErrors:\n" + "\n".join(errors)
             + f"\n\nAttempt {attempt}/{MAX_AI_REPAIR_ATTEMPTS}. Fix all errors."},
        ]
        return self.chat_completion(messages, temperature=0.1, max_tokens=2048)

    def analyze_results(self, metrics_summary: str, question: str) -> str:
        """Generate AI analysis of backtest results."""
        messages = [
            {"role": "system", "content": textwrap.dedent("""\
                You are an expert trading analyst. The user has run a backtest.
                Analyze the results and answer their question.
                Use the ACTUAL metrics provided — do NOT invent numbers.
                Be concise and insightful.
            """)},
            {"role": "user", "content": f"Results:\n{metrics_summary}\n\nQuestion: {question}"},
        ]
        return self.chat_completion(messages, temperature=0.5, max_tokens=1500)

    def classify_intent(self, message: str) -> UserIntent:
        """Classify what the user wants."""
        msg_lower = message.lower()
        strategy_keywords = [
            "buy when", "sell when", "entry", "exit", "crossover", "cross above",
            "cross below", "strategy", "rsi above", "rsi below", "ema", "sma",
            "macd", "bollinger", "stop loss", "take profit", "trailing",
            "add", "change", "modify", "use", "implement",
        ]
        config_keywords = [
            "use eth", "use btc", "use 4h", "use 1h", "use 1d",
            "capital", "position size", "$", "timeframe",
        ]
        analysis_keywords = [
            "show", "display", "list", "losing trades", "winning trades",
            "monthly returns", "drawdown", "biggest", "worst", "best",
            "how many", "what was", "average",
        ]
        if any(k in msg_lower for k in config_keywords):
            return UserIntent.CONFIG_CHANGE
        if any(k in msg_lower for k in strategy_keywords):
            return UserIntent.STRATEGY_CHANGE
        if any(k in msg_lower for k in analysis_keywords):
            return UserIntent.ANALYTICAL_QUERY
        return UserIntent.GENERAL_CHAT

    def _parse_error(self, resp: http_requests.Response) -> str:
        code = resp.status_code
        try:
            body = resp.json()
            msg = body.get("error", {}).get("message", resp.text[:200])
        except Exception:
            msg = resp.text[:200]
        errors = {
            401: "Authentication failed. Please check your API key.",
            403: "Access denied. Your API key may not have access to this model.",
            429: "Rate limit exceeded. Please wait a moment and try again.",
            500: "OpenRouter server error. Please try again later.",
            502: "OpenRouter gateway error. Please try again later.",
            503: "OpenRouter is temporarily unavailable.",
        }
        return errors.get(code, f"OpenRouter error ({code}): {msg}")


# ═══════════════════════════════════════════════════════════════════
# SECTION 5 — STRATEGY & CODE MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

MASTER_CODE_TEMPLATE = textwrap.dedent('''\
import numpy as np
import math

class UserStrategy:
    """AI-generated trading strategy."""

    def __init__(self):
        self.indicators = [
            {"name": "EMA", "period": 10, "source": "close"},
            {"name": "EMA", "period": 21, "source": "close"},
        ]

    def should_enter(self, idx: int, row: dict, indicators: dict) -> bool:
        """Return True to open a LONG position."""
        if idx < 25:
            return False
        ema_fast = indicators.get("EMA_10")
        ema_slow = indicators.get("EMA_21")
        if ema_fast is None or ema_slow is None:
            return False
        
        # Fast EMA crosses above Slow EMA
        return (
            ema_fast[idx-1] <= ema_slow[idx-1]
            and ema_fast[idx] > ema_slow[idx]
        )

    def should_exit(self, idx: int, row: dict, indicators: dict) -> bool:
        """Return True to close the current position."""
        ema_fast = indicators.get("EMA_10")
        ema_slow = indicators.get("EMA_21")
        if ema_fast is None or ema_slow is None:
            return False
            
        # Fast EMA crosses below Slow EMA
        return (
            ema_fast[idx-1] >= ema_slow[idx-1]
            and ema_fast[idx] < ema_slow[idx]
        )
''')


class MasterCodeManager:
    """Manages the canonical strategy code."""

    def __init__(self) -> None:
        self._code = MASTER_CODE_TEMPLATE
        self._log = get_logger("code_mgr")

    @property
    def template(self) -> str:
        return MASTER_CODE_TEMPLATE

    @property
    def current_code(self) -> str:
        return self._code

    @property
    def code_hash(self) -> str:
        return hashlib.sha256(self._code.encode()).hexdigest()[:16]

    def apply_ai_code(self, code: str) -> tuple[bool, list[str]]:
        """Replace the current strategy code with AI-generated code."""
        code = self._clean_code(code)
        if code.strip().upper() == "NA":
            return False, ["AI returned NA — not a valid trading strategy."]
        validator = CodeValidator()
        ok, errors = validator.validate(code)
        if ok:
            self._code = code
            self._log.info("Strategy code updated (hash=%s)", self.code_hash)
        return ok, errors

    def extract_indicator_config(self) -> list[dict]:
        """Extract indicator configuration from current strategy code."""
        try:
            ns: dict[str, Any] = {}
            exec(self._code, {"np": np, "numpy": np, "math": math}, ns)
            strategy_cls = ns.get("UserStrategy")
            if strategy_cls:
                instance = strategy_cls()
                return getattr(instance, "indicators", [])
        except Exception as e:
            self._log.warning("Failed to extract indicators: %s", e)
        return []

    def create_strategy_instance(self) -> Any:
        """Instantiate the current UserStrategy."""
        ns: dict[str, Any] = {}
        exec(self._code, {"np": np, "numpy": np, "math": math}, ns)
        cls = ns.get("UserStrategy")
        if cls is None:
            raise RuntimeError("UserStrategy class not found in code")
        return cls()

    def _clean_code(self, code: str) -> str:
        """Remove markdown fences and leading/trailing whitespace."""
        code = code.strip()
        if code.startswith("```"):
            lines = code.split("\n")
            lines = lines[1:]  # remove opening fence
            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]
            code = "\n".join(lines)
        return code.strip()


class CodeValidator:
    """Validates AI-generated strategy code for safety and correctness."""

    def __init__(self) -> None:
        self._log = get_logger("validator")

    def validate(self, code: str) -> tuple[bool, list[str]]:
        errors: list[str] = []
        # 1. Syntax
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            errors.append(f"Syntax error at line {e.lineno}: {e.msg}")
            return False, errors

        # 2. Required structure
        class_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        if "UserStrategy" not in class_names:
            errors.append("Missing required class: UserStrategy")

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "UserStrategy":
                methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                for req in ("__init__", "should_enter", "should_exit"):
                    if req not in methods:
                        errors.append(f"Missing required method: {req}")

        # 3. Security
        for pattern in FORBIDDEN_PATTERNS:
            matches = re.findall(pattern, code)
            if matches:
                errors.append(f"Forbidden pattern: {matches[0]}")

        # 4. Import validation
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in ALLOWED_IMPORTS:
                        errors.append(f"Forbidden import: {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                if node.module and node.module.split(".")[0] not in ALLOWED_IMPORTS:
                    errors.append(f"Forbidden import: {node.module}")

        if errors:
            return False, errors

        # 5. Dry initialization
        try:
            ns: dict[str, Any] = {}
            exec(code, {"np": np, "numpy": np, "math": math}, ns)
            cls = ns.get("UserStrategy")
            if cls:
                instance = cls()
                if not hasattr(instance, "indicators"):
                    errors.append("UserStrategy.__init__ must set self.indicators")
                if not callable(getattr(instance, "should_enter", None)):
                    errors.append("should_enter must be callable")
                if not callable(getattr(instance, "should_exit", None)):
                    errors.append("should_exit must be callable")
        except Exception as e:
            errors.append(f"Initialization failed: {e}")

        return len(errors) == 0, errors


class StrategyInterpreter:
    """Handles the AI code generation pipeline with repair loop."""

    def __init__(self, client: OpenRouterClient, code_mgr: MasterCodeManager) -> None:
        self._client = client
        self._code_mgr = code_mgr
        self._log = get_logger("strategy")

    def interpret(self, strategy_text: str, context: dict) -> tuple[bool, str, list[str]]:
        """Generate and validate strategy code, with repair loop.

        Returns: (success, final_code, errors)
        """
        self._log.info("Interpreting strategy: %s", strategy_text[:80])
        code = self._client.generate_strategy_code(
            strategy_text, self._code_mgr.template, context,
        )
        code = self._code_mgr._clean_code(code)

        if code.strip().upper() == "NA":
            return False, "", ["AI determined this is not a valid trading strategy."]

        validator = CodeValidator()
        for attempt in range(1, MAX_AI_REPAIR_ATTEMPTS + 1):
            ok, errors = validator.validate(code)
            if ok:
                self._log.info("Code validated on attempt %d", attempt)
                return True, code, []
            if attempt < MAX_AI_REPAIR_ATTEMPTS:
                self._log.warning("Validation failed (attempt %d): %s", attempt, errors)
                code = self._client.repair_code(code, errors, attempt + 1)
                code = self._code_mgr._clean_code(code)

        return False, code, errors


# ═══════════════════════════════════════════════════════════════════
# SECTION 6 — DATA PIPELINE
# ═══════════════════════════════════════════════════════════════════

class CCXTDataManager:
    """Downloads market data via CCXT."""

    def __init__(self, exchange_id: str = DEFAULT_EXCHANGE) -> None:
        self._exchange_id = exchange_id
        self._exchange: Any = None
        self._log = get_logger("ccxt")

    def initialize(self) -> None:
        if not HAS_CCXT:
            raise RuntimeError("ccxt is not installed. Run: pip install ccxt")
        exchange_cls = getattr(_ccxt, self._exchange_id, None)
        if exchange_cls is None:
            raise RuntimeError(f"Unknown exchange: {self._exchange_id}")
        self._exchange = exchange_cls({"enableRateLimit": True})

    def load_markets(self) -> None:
        if self._exchange:
            self._exchange.load_markets()

    def download(
        self, symbol: str, timeframe: str,
        start: int | None = None,
        end: int | None = None,
        progress_callback: Callable | None = None,
        cancel_check: Callable | None = None,
    ) -> pl.DataFrame:
        """Download OHLCV data for a symbol."""
        if not self._exchange:
            self.initialize()
            self.load_markets()
        if symbol not in self._exchange.markets:
            raise RuntimeError(f"Symbol {symbol} not found on {self._exchange_id}")

        all_rows: list[list] = []
        since = start
        if end is None:
            end = int(datetime.now(timezone.utc).timestamp() * 1000)

        for retry in range(CCXT_MAX_RETRIES):
            try:
                while True:
                    if cancel_check and cancel_check():
                        raise RuntimeError("Download cancelled")
                    ohlcv = self._exchange.fetch_ohlcv(
                        symbol, timeframe, since=since, limit=CCXT_OHLCV_LIMIT,
                    )
                    if not ohlcv:
                        break
                    all_rows.extend(ohlcv)
                    if progress_callback:
                        progress_callback(len(all_rows), symbol)
                    last_ts = ohlcv[-1][0]
                    if last_ts >= end:
                        break
                    since = last_ts + 1
                    if len(ohlcv) < CCXT_OHLCV_LIMIT:
                        break
                    time.sleep(self._exchange.rateLimit / 1000)
                break
            except _ccxt.RateLimitExceeded:
                self._log.warning("Rate limit hit, waiting %ds...", CCXT_RETRY_DELAY * (retry + 1))
                time.sleep(CCXT_RETRY_DELAY * (retry + 1))
            except _ccxt.NetworkError as e:
                if retry == CCXT_MAX_RETRIES - 1:
                    raise RuntimeError(f"Network error downloading {symbol}: {e}")
                time.sleep(CCXT_RETRY_DELAY)

        if not all_rows:
            raise RuntimeError(f"No data downloaded for {symbol}")

        df = pl.DataFrame(
            all_rows,
            schema=["timestamp", "open", "high", "low", "close", "volume"],
            orient="row",
        )
        df = df.with_columns(
            pl.col("timestamp").cast(pl.Int64),
            pl.col("open").cast(pl.Float64),
            pl.col("high").cast(pl.Float64),
            pl.col("low").cast(pl.Float64),
            pl.col("close").cast(pl.Float64),
            pl.col("volume").cast(pl.Float64),
        )
        df = df.filter(pl.col("timestamp") <= end)
        df = df.unique(subset=["timestamp"]).sort("timestamp")
        self._log.info("Downloaded %d candles for %s", len(df), symbol)
        return df


class PolarsDataManager:
    """Data normalization, validation, and storage with Polars."""

    def __init__(self) -> None:
        self._log = get_logger("data")

    def normalize(self, df: pl.DataFrame, symbol: str) -> pl.DataFrame:
        df = df.with_columns([
            (pl.col("timestamp") * 1_000_000).cast(pl.Datetime("ns", "UTC")).alias("datetime"),
            pl.lit(symbol).alias("symbol"),
        ])
        return df

    def validate(self, df: pl.DataFrame, symbol: str, timeframe: str) -> list[str]:
        anomalies: list[str] = []
        if df.is_empty():
            anomalies.append(f"{symbol}: empty dataset")
            return anomalies
        zero_vol = df.filter(pl.col("volume") == 0).height
        if zero_vol > 0:
            anomalies.append(f"{symbol}: {zero_vol} zero-volume bars")
        null_count = df.null_count().sum_horizontal()[0]
        if null_count > 0:
            anomalies.append(f"{symbol}: {null_count} null values")
        dup_count = df.height - df.unique(subset=["timestamp"]).height
        if dup_count > 0:
            anomalies.append(f"{symbol}: {dup_count} duplicate timestamps")
        return anomalies

    def clean(self, df: pl.DataFrame) -> pl.DataFrame:
        df = df.unique(subset=["timestamp"]).sort("timestamp")
        df = df.drop_nulls()
        return df

    def save_parquet(self, df: pl.DataFrame, path: Path) -> None:
        df.write_parquet(path)

    def load_parquet(self, path: Path) -> pl.DataFrame:
        return pl.read_parquet(path)


# ═══════════════════════════════════════════════════════════════════
# SECTION 7 — INDICATORS
# ═══════════════════════════════════════════════════════════════════

class IndicatorManager:
    """Computes technical indicators using TA-Lib."""

    def __init__(self) -> None:
        self._log = get_logger("indicators")

    def compute(
        self, df: pl.DataFrame, name: str, period: int = 14,
        source: str = "close", params: dict | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute a single indicator, return dict of arrays."""
        close = df["close"].to_numpy().astype(np.float64)
        high = df["high"].to_numpy().astype(np.float64)
        low = df["low"].to_numpy().astype(np.float64)
        volume = df["volume"].to_numpy().astype(np.float64)

        key = f"{name}_{period}" if period > 0 else name
        result: dict[str, np.ndarray] = {}
        name_upper = name.upper()

        if not HAS_TALIB:
            if name_upper == "MACD":
                fast_p, slow_p, sig_p = 12, 26, 9
                fast_ema = self._fallback_indicator("EMA", close, high, low, volume, fast_p)
                slow_ema = self._fallback_indicator("EMA", close, high, low, volume, slow_p)
                macd = fast_ema - slow_ema
                
                signal = np.full(len(close), np.nan)
                valid_idx = slow_p - 1
                if len(close) > valid_idx + sig_p:
                    valid_macd = macd[valid_idx:]
                    sig_ema = self._fallback_indicator("EMA", valid_macd, valid_macd, valid_macd, valid_macd, sig_p)
                    signal[valid_idx:] = sig_ema
                    
                hist = macd - signal
                result[f"MACD_{period}"] = macd
                result[f"MACD_SIGNAL_{period}"] = signal
                result[f"MACD_HIST_{period}"] = hist
                return result
            elif name_upper == "BBANDS":
                sma = self._fallback_indicator("SMA", close, high, low, volume, period)
                std = np.full(len(close), np.nan)
                for i in range(period - 1, len(close)):
                    std[i] = np.std(close[i - period + 1 : i + 1])
                result[f"BBANDS_UPPER_{period}"] = sma + 2 * std
                result[f"BBANDS_MIDDLE_{period}"] = sma
                result[f"BBANDS_LOWER_{period}"] = sma - 2 * std
                return result

            result[key] = self._fallback_indicator(name_upper, close, high, low, volume, period)
            return result

        try:
            if name_upper == "RSI":
                result[key] = _talib.RSI(close, timeperiod=period)
            elif name_upper == "EMA":
                result[key] = _talib.EMA(close, timeperiod=period)
            elif name_upper == "SMA":
                result[key] = _talib.SMA(close, timeperiod=period)
            elif name_upper == "MACD":
                macd, signal, hist = _talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
                result[f"MACD_{period}"] = macd
                result[f"MACD_SIGNAL_{period}"] = signal
                result[f"MACD_HIST_{period}"] = hist
            elif name_upper == "ATR":
                result[key] = _talib.ATR(high, low, close, timeperiod=period)
            elif name_upper == "ADX":
                result[key] = _talib.ADX(high, low, close, timeperiod=period)
            elif name_upper == "CCI":
                result[key] = _talib.CCI(high, low, close, timeperiod=period)
            elif name_upper == "BBANDS":
                upper, middle, lower = _talib.BBANDS(close, timeperiod=period)
                result[f"BBANDS_UPPER_{period}"] = upper
                result[f"BBANDS_MIDDLE_{period}"] = middle
                result[f"BBANDS_LOWER_{period}"] = lower
            elif name_upper == "STOCH":
                k, d = _talib.STOCH(high, low, close, fastk_period=period)
                result[f"STOCH_K_{period}"] = k
                result[f"STOCH_D_{period}"] = d
            elif name_upper == "ROC":
                result[key] = _talib.ROC(close, timeperiod=period)
            elif name_upper == "MFI":
                result[key] = _talib.MFI(high, low, close, volume, timeperiod=period)
            elif name_upper == "OBV":
                result["OBV"] = _talib.OBV(close, volume)
            else:
                self._log.warning("Unknown indicator: %s", name)
                result[key] = np.full(len(close), np.nan)
        except Exception as e:
            self._log.error("TA-Lib error for %s: %s", name, e)
            result[key] = np.full(len(close), np.nan)

        return result

    def compute_multiple(
        self, df: pl.DataFrame, configs: list[dict],
    ) -> dict[str, np.ndarray]:
        """Compute multiple indicators, merge results."""
        all_indicators: dict[str, np.ndarray] = {}
        for cfg in configs:
            result = self.compute(
                df, cfg.get("name", ""), cfg.get("period", 14),
                cfg.get("source", "close"), cfg.get("params"),
            )
            all_indicators.update(result)
        return all_indicators

    def _fallback_indicator(
        self, name: str, close: np.ndarray, high: np.ndarray,
        low: np.ndarray, volume: np.ndarray, period: int,
    ) -> np.ndarray:
        """Pure-numpy fallback when TA-Lib is not installed."""
        n = len(close)
        if name == "SMA":
            out = np.full(n, np.nan)
            for i in range(period - 1, n):
                out[i] = np.mean(close[i - period + 1: i + 1])
            return out
        elif name == "EMA":
            out = np.full(n, np.nan)
            k = 2.0 / (period + 1)
            out[period - 1] = np.mean(close[:period])
            for i in range(period, n):
                out[i] = close[i] * k + out[i - 1] * (1 - k)
            return out
        elif name == "RSI":
            out = np.full(n, np.nan)
            deltas = np.diff(close)
            gains = np.where(deltas > 0, deltas, 0.0)
            losses = np.where(deltas < 0, -deltas, 0.0)
            avg_gain = np.mean(gains[:period])
            avg_loss = np.mean(losses[:period])
            if avg_loss == 0:
                out[period] = 100.0
            else:
                out[period] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            for i in range(period + 1, n):
                avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
                avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
                if avg_loss == 0:
                    out[i] = 100.0
                else:
                    out[i] = 100.0 - 100.0 / (1.0 + avg_gain / avg_loss)
            return out
        elif name == "ATR":
            tr = np.maximum(
                high[1:] - low[1:],
                np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1]))
            )
            tr = np.concatenate([[high[0] - low[0]], tr])
            out = np.full(n, np.nan)
            out[period - 1] = np.mean(tr[:period])
            for i in range(period, n):
                out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
            return out
        elif name == "OBV":
            out = np.zeros(n)
            for i in range(1, n):
                if close[i] > close[i - 1]:
                    out[i] = out[i - 1] + volume[i]
                elif close[i] < close[i - 1]:
                    out[i] = out[i - 1] - volume[i]
                else:
                    out[i] = out[i - 1]
            return out
        return np.full(n, np.nan)


# ═══════════════════════════════════════════════════════════════════
# SECTION 8 — BACKTEST ENGINE
# ═══════════════════════════════════════════════════════════════════

class FeeManager:
    def __init__(self, config: FeeConfig) -> None:
        self.maker = config.maker_fee_pct
        self.taker = config.taker_fee_pct

    def calculate(self, notional: float, is_maker: bool = False) -> float:
        rate = self.maker if is_maker else self.taker
        return notional * rate


class SlippageManager:
    def __init__(self, config: SlippageConfig) -> None:
        self.pct = config.slippage_pct

    def apply(self, price: float, is_buy: bool) -> float:
        factor = 1 + self.pct if is_buy else 1 - self.pct
        return price * factor


class PortfolioManager:
    def __init__(self, initial_capital: float, position_size_pct: float) -> None:
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.position_size_pct = position_size_pct
        self.positions: dict[str, dict] = {}

    def available_capital(self) -> float:
        return self.cash

    def position_value(self, symbol: str, current_price: float) -> float:
        pos = self.positions.get(symbol)
        if not pos:
            return 0.0
        return pos["quantity"] * current_price

    def total_value(self, prices: dict[str, float]) -> float:
        total = self.cash
        for sym, pos in self.positions.items():
            total += pos["quantity"] * prices.get(sym, pos["entry_price"])
        return total

    def can_enter(self, symbol: str) -> bool:
        return symbol not in self.positions and self.cash > 0

    def enter(self, symbol: str, price: float, quantity: float) -> None:
        cost = price * quantity
        self.cash -= cost
        self.positions[symbol] = {
            "entry_price": price, "quantity": quantity,
            "entry_time": datetime.now(timezone.utc).isoformat(),
        }

    def exit(self, symbol: str, price: float) -> tuple[float, float]:
        pos = self.positions.pop(symbol, None)
        if not pos:
            return 0.0, 0.0
        proceeds = price * pos["quantity"]
        self.cash += proceeds
        pnl = (price - pos["entry_price"]) * pos["quantity"]
        return pnl, pos["quantity"]


class BacktestManager:
    """Vectorized backtest engine with signal-based execution."""

    def __init__(self) -> None:
        self._log = get_logger("backtest")

    def run(
        self, config: BacktestConfig, datasets: dict[str, pl.DataFrame],
        strategy: Any, indicators_map: dict[str, dict[str, np.ndarray]],
        progress_callback: Callable | None = None,
        cancel_check: Callable | None = None,
    ) -> dict:
        """Execute the backtest across all assets."""
        fee_mgr = FeeManager(config.fee_config)
        slip_mgr = SlippageManager(config.slippage_config)
        portfolio = PortfolioManager(config.initial_capital, config.position_size_pct)

        all_trades: list[TradeRecord] = []
        equity_curve: list[dict] = []
        trade_counter = 0
        total_bars = sum(len(df) for df in datasets.values())
        bars_processed = 0

        for symbol, df in datasets.items():
            indicators = indicators_map.get(symbol, {})
            rows = df.to_dicts()
            n = len(rows)

            for idx in range(n):
                if cancel_check and cancel_check():
                    raise RuntimeError("Backtest cancelled")

                row = rows[idx]
                current_price = row["close"]
                ts = row.get("timestamp", 0)
                dt_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else ""

                # Check exit first
                if symbol in portfolio.positions:
                    try:
                        should_exit = strategy.should_exit(idx, row, indicators)
                    except Exception:
                        should_exit = False
                    if should_exit:
                        exit_price = slip_mgr.apply(current_price, is_buy=False)
                        pos = portfolio.positions[symbol]
                        exit_fee = fee_mgr.calculate(exit_price * pos["quantity"])
                        pnl_raw, qty = portfolio.exit(symbol, exit_price)
                        entry_fee = fee_mgr.calculate(pos["entry_price"] * qty)
                        total_fee = entry_fee + exit_fee
                        slip_cost = abs(current_price - exit_price) * qty
                        net_pnl = pnl_raw - total_fee

                        trade_counter += 1
                        all_trades.append(TradeRecord(
                            trade_id=trade_counter, asset=symbol, side="LONG",
                            entry_time=pos.get("entry_time", ""),
                            entry_price=pos["entry_price"],
                            exit_time=dt_str, exit_price=exit_price,
                            quantity=qty, pnl=net_pnl,
                            pnl_pct=(net_pnl / (pos["entry_price"] * qty)) * 100 if qty > 0 else 0,
                            fees=total_fee, slippage=slip_cost,
                        ))

                # Check entry
                if portfolio.can_enter(symbol):
                    try:
                        should_enter = strategy.should_enter(idx, row, indicators)
                    except Exception:
                        should_enter = False
                    if should_enter:
                        entry_price = slip_mgr.apply(current_price, is_buy=True)
                        alloc = portfolio.cash * portfolio.position_size_pct
                        qty = alloc / entry_price
                        if qty > 0:
                            portfolio.enter(symbol, entry_price, qty)

                # Record equity
                bars_processed += 1
                if idx % max(1, n // 100) == 0 or idx == n - 1:
                    prices = {s: datasets[s]["close"][min(idx, len(datasets[s]) - 1)]
                              for s in datasets}
                    equity_curve.append({
                        "timestamp": ts,
                        "datetime": dt_str,
                        "value": portfolio.total_value(prices),
                        "cash": portfolio.cash,
                    })

                if progress_callback and bars_processed % max(1, total_bars // 50) == 0:
                    progress_callback(bars_processed / total_bars * 100, f"Processing {symbol}...")

        # Close any remaining positions at last price
        for symbol in list(portfolio.positions.keys()):
            df = datasets[symbol]
            last_price = df["close"][-1]
            pos = portfolio.positions[symbol]
            exit_fee = fee_mgr.calculate(last_price * pos["quantity"])
            pnl_raw, qty = portfolio.exit(symbol, last_price)
            entry_fee = fee_mgr.calculate(pos["entry_price"] * qty)
            trade_counter += 1
            all_trades.append(TradeRecord(
                trade_id=trade_counter, asset=symbol, side="LONG",
                entry_time=pos.get("entry_time", ""),
                entry_price=pos["entry_price"],
                exit_time="END", exit_price=last_price,
                quantity=qty, pnl=pnl_raw - entry_fee - exit_fee,
                pnl_pct=((last_price - pos["entry_price"]) / pos["entry_price"]) * 100,
                fees=entry_fee + exit_fee, slippage=0,
            ))

        return {
            "trades": all_trades,
            "equity_curve": equity_curve,
            "final_value": portfolio.cash,
        }


# ═══════════════════════════════════════════════════════════════════
# SECTION 9 — ANALYTICS
# ═══════════════════════════════════════════════════════════════════

class MetricsManager:
    """Calculates comprehensive backtest metrics."""

    def calculate(
        self, trades: list[TradeRecord], equity_curve: list[dict],
        initial_capital: float,
    ) -> MetricsReport:
        m = MetricsReport(initial_capital=initial_capital)
        if not trades:
            m.final_value = initial_capital
            return m

        pnls = [t.pnl for t in trades]
        m.total_trades = len(trades)
        m.winning_trades = sum(1 for p in pnls if p > 0)
        m.losing_trades = sum(1 for p in pnls if p <= 0)
        m.win_rate = (m.winning_trades / m.total_trades * 100) if m.total_trades else 0
        m.net_pnl = sum(pnls)
        m.total_fees = sum(t.fees for t in trades)
        m.total_slippage = sum(t.slippage for t in trades)
        m.total_charges = m.total_fees + m.total_slippage

        if equity_curve:
            m.final_value = equity_curve[-1]["value"]
        else:
            m.final_value = initial_capital + m.net_pnl
        m.return_pct = ((m.final_value - initial_capital) / initial_capital) * 100

        # Profit factor
        gross_profit = sum(p for p in pnls if p > 0)
        gross_loss = abs(sum(p for p in pnls if p < 0))
        m.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf") if gross_profit > 0 else 0

        # Best / Worst / Avg
        m.best_trade = max(pnls) if pnls else 0
        m.worst_trade = min(pnls) if pnls else 0
        m.avg_trade_pnl = np.mean(pnls) if pnls else 0

        # Consecutive wins/losses
        m.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        m.max_consecutive_losses = self._max_consecutive(pnls, positive=False)

        # Sharpe & Sortino (annualized, assuming daily returns)
        if len(equity_curve) > 1:
            values = [e["value"] for e in equity_curve]
            returns = np.diff(values) / values[:-1]
            returns = returns[np.isfinite(returns)]
            if len(returns) > 1:
                mean_r = np.mean(returns)
                std_r = np.std(returns, ddof=1)
                m.sharpe_ratio = round((mean_r / std_r * np.sqrt(252)) if std_r > 0 else 0, 2)
                downside = returns[returns < 0]
                down_std = np.std(downside, ddof=1) if len(downside) > 1 else 0
                m.sortino_ratio = round((mean_r / down_std * np.sqrt(252)) if down_std > 0 else 0, 2)

        # Max drawdown
        if equity_curve:
            values = np.array([e["value"] for e in equity_curve])
            peak = np.maximum.accumulate(values)
            dd = (values - peak) / peak * 100
            m.max_drawdown_pct = round(float(np.min(dd)), 2) if len(dd) > 0 else 0

        # Expectancy
        avg_win = np.mean([p for p in pnls if p > 0]) if m.winning_trades > 0 else 0
        avg_loss = abs(np.mean([p for p in pnls if p < 0])) if m.losing_trades > 0 else 0
        wr = m.win_rate / 100
        m.expectancy = round(avg_win * wr - avg_loss * (1 - wr), 2)

        return m

    def _max_consecutive(self, pnls: list[float], positive: bool) -> int:
        max_c = current = 0
        for p in pnls:
            if (positive and p > 0) or (not positive and p <= 0):
                current += 1
                max_c = max(max_c, current)
            else:
                current = 0
        return max_c


# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — VISUALIZATION
# ═══════════════════════════════════════════════════════════════════

CHART_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="#000000",
    plot_bgcolor="#000000",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#e2e8f0", size=12),
    margin=dict(l=50, r=30, t=40, b=40),
    legend=dict(bgcolor="rgba(0,0,0,0)"),
)


class VisualizationManager:
    """Generates a modern HTML dashboard using Tailwind CSS and ApexCharts."""

    def __init__(self) -> None:
        self._log = get_logger("viz")

    def generate_dashboard_html(self, result: "BacktestResult", is_dark: bool = False) -> str:
        import json
        from datetime import datetime
        m = result.metrics
        equity = result.equity_curve
        trades = result.trades

        # Process Dates and Duration
        start_date_str = equity[0].get("datetime", "")[:10] if equity else "N/A"
        end_date_str = equity[-1].get("datetime", "")[:10] if equity else "N/A"
        duration_days = 0
        if start_date_str != "N/A" and end_date_str != "N/A":
            try:
                d1 = datetime.strptime(start_date_str, "%Y-%m-%d")
                d2 = datetime.strptime(end_date_str, "%Y-%m-%d")
                duration_days = (d2 - d1).days
            except:
                pass

        # Process Equity & Drawdown (Underwater) data
        dates = [e.get("datetime", "")[:10] for e in equity]
        values = [e["value"] for e in equity]
        
        peak = 0
        uw_dd_values = []
        for v in values:
            if v > peak: peak = v
            dd = ((v - peak) / peak * 100) if peak > 0 else 0
            uw_dd_values.append(round(dd, 2)) # Negative drawdown values
            
        values = [round(v, 2) for v in values]

        # Process Monthly Returns
        monthly = {}
        for i, e in enumerate(equity):
            dt = e.get("datetime", "")
            if len(dt) >= 7:
                month_key = dt[:7]
                if month_key not in monthly:
                    monthly[month_key] = equity[max(0, i - 1)]["value"] if i > 0 else e["value"]
                monthly[month_key] = e["value"]

        months = list(monthly.keys())
        month_vals = list(monthly.values())
        monthly_rets = []
        monthly_labels = []
        for i in range(1, len(month_vals)):
            ret = (month_vals[i] - month_vals[i-1]) / month_vals[i-1] * 100
            monthly_rets.append(round(ret, 2))
            monthly_labels.append(months[i])

        # Heatmap Process (Month-wise Returns)
        years = {}
        for m_lbl, r in zip(monthly_labels, monthly_rets):
            yr = m_lbl[:4]
            mo = m_lbl[5:7]
            if yr not in years:
                years[yr] = {m: None for m in ['01','02','03','04','05','06','07','08','09','10','11','12']}
            years[yr][mo] = r
            
        heatmap_series = []
        month_names = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
        for yr in sorted(years.keys(), reverse=True):
            data_points = []
            for i, mo in enumerate(['01','02','03','04','05','06','07','08','09','10','11','12']):
                data_points.append({"x": month_names[i], "y": years[yr][mo]})
            heatmap_series.append({"name": yr, "data": data_points})

        wins = int(m.total_trades * m.win_rate / 100)
        losses = m.total_trades - wins
        win_pct = round(m.win_rate, 1) if m.total_trades > 0 else 0.0
        loss_pct = round(100 - m.win_rate, 1) if m.total_trades > 0 else 0.0
        
        asset_name = result.config.assets[0] if result.config.assets else "Portfolio"

        bg_page = "#000000" if is_dark else "#fefefe"
        bg_card = "#000000" if is_dark else "white"
        border_card = "#334155" if is_dark else "#e2e8f0"
        shadow_card = "0 1px 3px rgba(0,0,0,0.2)" if is_dark else "0 1px 3px rgba(0,0,0,0.05)"
        text_primary = "#f8fafc" if is_dark else "inherit" # inherit falls back to body which is black by default or set by tailwind
        text_label = "#94a3b8" if is_dark else "#64748b"
        tooltip_bg = "#1e293b" if is_dark else "#1e293b" # keep original
        tooltip_title_bg = "#0f172a" if is_dark else "#0f172a"
        tooltip_border = "#334155" if is_dark else "#334155"
        
        tw_text_main = "text-slate-100" if is_dark else "text-slate-800"
        tw_text_sub = "text-slate-400" if is_dark else "text-slate-500"
        tw_border = "border-slate-700/50" if is_dark else "border-slate-100"
        tw_emerald_text = "text-emerald-500" if is_dark else "text-emerald-600"
        tw_emerald_bg = "bg-emerald-900/30 border-emerald-800" if is_dark else "bg-emerald-50"
        
        apex_theme = "dark" if is_dark else "light"
        apex_grid_border = "#334155" if is_dark else "#f1f5f9"
        apex_donut_stroke = "#1e212b" if is_dark else "#fff"
        apex_text_color = "#94a3b8" if is_dark else "#94a3b8"
        
        heatmap_ranges = f"""
                            {{ from: -1000, to: -20, color: '#7f1d1d', name: 'Heavy Loss (<-20%)' }},
                            {{ from: -20, to: -5, color: '#b91c1c', name: 'Loss (-5% to -20%)' }},
                            {{ from: -5, to: -0.01, color: '#f87171', name: 'Small Loss (0% to -5%)' }},
                            {{ from: -0.01, to: 0.01, color: '{("#334155" if is_dark else "#cbd5e1")}', name: 'Break Even' }},
                            {{ from: 0.01, to: 5, color: '{("#059669" if is_dark else "#6ee7b7")}', name: 'Small Profit (0% to 5%)' }},
                            {{ from: 5, to: 20, color: '#10b981', name: 'Profit (5% to 20%)' }},
                            {{ from: 20, to: 10000, color: '{("#34d399" if is_dark else "#047857")}', name: 'High Profit (>20%)' }}
        """

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://cdn.jsdelivr.net/npm/apexcharts"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        ::-webkit-scrollbar {{ display: none; }}
        body {{ background-color: {bg_page}; font-family: 'Inter', sans-serif; -ms-overflow-style: none; scrollbar-width: none; }}
        .card {{ background-color: {bg_card}; border-radius: 12px; box-shadow: {shadow_card}; border: 1px solid {border_card}; padding: 24px; }}
        .stat-value {{ font-size: 24px; font-weight: 700; color: {text_primary}; }}
        .stat-label {{ font-size: 13px; color: {text_label}; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; }}
        .apexcharts-tooltip {{ background: {tooltip_bg} !important; color: white !important; border: none !important; border-radius: 8px !important; box-shadow: 0 4px 6px rgba(0,0,0,0.3) !important; }}
        .apexcharts-tooltip-title {{ background: {tooltip_title_bg} !important; border-bottom: 1px solid {tooltip_border} !important; font-family: 'Inter', sans-serif !important; font-weight: 600 !important; }}
    </style>
</head>
<body class="p-2">
    <div class="max-w-7xl mx-auto space-y-6">
        
        <!-- Header -->
        <div class="flex items-center justify-between mb-4">
            <div>
                <h1 class="text-2xl font-bold {tw_text_main}">Analytics Dashboard <span class="{tw_emerald_text}">— {asset_name}</span></h1>
                <p class="{tw_text_sub} font-medium text-sm mt-1">Cumulative Backtest: {start_date_str} to {end_date_str}</p>
            </div>
            <div class="{tw_emerald_text} font-semibold flex items-center {tw_emerald_bg} px-3 py-1 rounded-full text-sm border">
                <svg class="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M5 13l4 4L19 7"></path></svg>
                {duration_days} Days Analyzed
            </div>
        </div>

        <!-- Top Metrics -->
        <div class="grid grid-cols-4 gap-4">
            <div class="card p-5">
                <div class="stat-label mb-1">Cumulative Return</div>
                <div class="stat-value {tw_emerald_text}">{m.return_pct:+.2f}%</div>
            </div>
            <div class="card p-5">
                <div class="stat-label mb-1">Total Charges (Fees+Slip)</div>
                <div class="stat-value text-red-500">-${m.total_charges:,.2f}</div>
            </div>
            <div class="card p-5">
                <div class="stat-label mb-1">Total Trades</div>
                <div class="stat-value {tw_text_main}">{m.total_trades}</div>
            </div>
            <div class="card p-5">
                <div class="stat-label mb-1">Hit Rate (Win %)</div>
                <div class="stat-value {tw_text_main}">{m.win_rate:.1f}%</div>
            </div>
        </div>

        <!-- Main Charts Area -->
        <div class="grid grid-cols-12 gap-6">
            
            <!-- Donut -->
            <div class="col-span-5 card flex flex-col items-center justify-center p-6">
                <h3 class="w-full text-left font-bold {tw_text_main} mb-2">Hit / Win Rate</h3>
                <div id="donutChart" class="w-full mt-4 flex justify-center"></div>
            </div>

            <!-- Performance Table -->
            <div class="col-span-7 card p-6">
                <h3 class="font-bold {tw_text_main} mb-4">Cumulative Performance & Charges</h3>
                <table class="w-full text-sm">
                    <tbody>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub}">Gross P&L</td><td class="py-2.5 text-right font-semibold {tw_text_main}">${m.net_pnl + m.total_charges:,.2f}</td></tr>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub} text-red-500 font-medium">Real Charges (Fees/Slip)</td><td class="py-2.5 text-right font-semibold text-red-500">-${m.total_charges:,.2f}</td></tr>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub} font-bold">Net P&L</td><td class="py-2.5 text-right font-bold {tw_emerald_text}">{"+" if m.net_pnl >= 0 else ""}${m.net_pnl:,.2f}</td></tr>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub}">Profit Factor</td><td class="py-2.5 text-right font-semibold {tw_text_main}">{m.profit_factor:.2f}</td></tr>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub}">Sharpe Ratio</td><td class="py-2.5 text-right font-semibold {tw_text_main}">{m.sharpe_ratio:.2f}</td></tr>
                        <tr class="border-b {tw_border}"><td class="py-2.5 {tw_text_sub}">Max Drawdown</td><td class="py-2.5 text-right font-semibold text-red-500">{m.max_drawdown_pct:.2f}%</td></tr>
                        <tr><td class="py-2.5 {tw_text_sub}">Expectancy</td><td class="py-2.5 text-right font-semibold {tw_text_main}">${m.expectancy:,.2f}</td></tr>
                    </tbody>
                </table>
            </div>
        </div>

        <!-- Monthly Returns Heatmap (Advanced Visualization) -->
        <div class="card">
            <h3 class="font-bold {tw_text_main} mb-2">Monthly Returns Heatmap (%)</h3>
            <div id="heatmapChart" class="w-full h-[350px]"></div>
        </div>

        <!-- Equity Curve -->
        <div class="card">
            <h3 class="font-bold {tw_text_main} mb-2">Cumulative Equity Curve</h3>
            <div id="equityChart" class="w-full h-[300px]"></div>
        </div>

        <!-- Detailed Analytics section -->
        <div class="grid grid-cols-2 gap-6">
            <div class="card flex flex-col">
                <h3 class="font-bold {tw_text_main} mb-2">Underwater Drawdown Curve</h3>
                <div id="drawdownChart" class="w-full h-[250px]"></div>
            </div>
            <div class="card flex flex-col">
                <h3 class="font-bold {tw_text_main} mb-2">Month-wise Returns (%)</h3>
                <div id="monthlyChart" class="w-full h-[250px]"></div>
            </div>
        </div>

    </div>

    <script>
        const chartOptions = {{
            fontFamily: 'Inter, sans-serif',
            toolbar: {{ show: false }},
            animations: {{ enabled: true, easing: 'easeinout', speed: 800 }},
            theme: {{ mode: '{apex_theme}' }},
            background: 'transparent'
        }};

        // Pie/Donut Chart for Hit Rate
        new ApexCharts(document.querySelector("#donutChart"), {{
            ...chartOptions,
            series: [{win_pct}, {loss_pct}],
            labels: ['Wins (%)', 'Losses (%)'],
            chart: {{ type: 'pie', height: 260 }},
            colors: ['#10b981', '#ef4444'],
            plotOptions: {{
                pie: {{
                    dataLabels: {{ offset: -20 }},
                }}
            }},
            dataLabels: {{ 
                enabled: true, 
                formatter: function (val) {{ return val.toFixed(1) + "%" }},
                style: {{ fontSize: '14px', fontWeight: 'bold' }}
            }},
            stroke: {{ width: 2, colors: ['{apex_donut_stroke}'] }},
            legend: {{ position: 'bottom' }}
        }}).render();

        // Heatmap Chart for Monthly Returns
        new ApexCharts(document.querySelector("#heatmapChart"), {{
            ...chartOptions,
            series: {json.dumps(heatmap_series)},
            chart: {{ type: 'heatmap', height: 350, toolbar: {{ show: false }} }},
            plotOptions: {{
                heatmap: {{
                    shadeIntensity: 0.8,
                    radius: 4,
                    colorScale: {{
                        ranges: [
                            {heatmap_ranges}
                        ]
                    }}
                }}
            }},
            dataLabels: {{ 
                enabled: true, 
                style: {{ colors: ['#fff'] }},
                formatter: function(val) {{ return val !== null ? val.toFixed(1) + "%" : ""; }}
            }}
        }}).render();

        // Equity Curve
        new ApexCharts(document.querySelector("#equityChart"), {{
            ...chartOptions,
            series: [{{ name: 'Cumulative Portfolio Value', data: {json.dumps(values)} }}],
            chart: {{ type: 'area', height: 300, toolbar: {{ show: false }} }},
            colors: ['#3b82f6'],
            fill: {{ type: 'gradient', gradient: {{ shadeIntensity: 1, opacityFrom: 0.3, opacityTo: 0.0, stops: [0, 100] }} }},
            dataLabels: {{ enabled: false }},
            stroke: {{ curve: 'smooth', width: 2 }},
            xaxis: {{ categories: {json.dumps(dates)}, type: 'category', tickAmount: 6, labels: {{ style: {{ colors: '{apex_text_color}' }} }} }},
            yaxis: {{ labels: {{ formatter: (val) => '$' + val.toLocaleString(), style: {{ colors: '{apex_text_color}' }} }} }},
            grid: {{ borderColor: '{apex_grid_border}', strokeDashArray: 4 }}
        }}).render();

        // Underwater Drawdown Curve (Negative values)
        new ApexCharts(document.querySelector("#drawdownChart"), {{
            ...chartOptions,
            series: [{{ name: 'Drawdown', data: {json.dumps(uw_dd_values)} }}],
            chart: {{ type: 'area', height: 250, toolbar: {{ show: false }} }},
            colors: ['#ef4444'],
            fill: {{ type: 'solid', opacity: 0.2 }},
            dataLabels: {{ enabled: false }},
            stroke: {{ curve: 'smooth', width: 2 }},
            xaxis: {{ categories: {json.dumps(dates)}, type: 'category', tickAmount: 5, labels: {{ style: {{ colors: '{apex_text_color}' }} }} }},
            yaxis: {{ max: 0, labels: {{ formatter: (val) => val.toFixed(1) + '%', style: {{ colors: '{apex_text_color}' }} }} }},
            grid: {{ borderColor: '{apex_grid_border}', strokeDashArray: 4 }}
        }}).render();

        // Monthly Returns
        new ApexCharts(document.querySelector("#monthlyChart"), {{
            ...chartOptions,
            series: [{{ name: 'Return', data: {json.dumps(monthly_rets)} }}],
            chart: {{ type: 'bar', height: 250, toolbar: {{ show: false }} }},
            colors: [function({{ value }}) {{ return value >= 0 ? '#10b981' : '#ef4444'; }}],
            plotOptions: {{ bar: {{ borderRadius: 4, columnWidth: '60%' }} }},
            dataLabels: {{ enabled: false }},
            xaxis: {{ categories: {json.dumps(monthly_labels)}, tickAmount: 10, labels: {{ rotate: -45, hideOverlappingLabels: true, style: {{ colors: '{apex_text_color}' }} }} }},
            yaxis: {{ labels: {{ formatter: (val) => val.toFixed(1) + '%', style: {{ colors: '{apex_text_color}' }} }} }},
            grid: {{ borderColor: '{apex_grid_border}', strokeDashArray: 4 }}
        }}).render();
    </script>
</body>
</html>"""
        return html


# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — RESULT & CHAT MANAGEMENT
# ═══════════════════════════════════════════════════════════════════

class ResultManager:
    """Persists and retrieves backtest results."""

    def __init__(self) -> None:
        self._log = get_logger("results")
        self._cache: dict[str, BacktestResult] = {}

    def create_run_directory(self, run_id: str) -> Path:
        d = RESULTS_DIR / run_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_complete_result(
        self, run_dir: Path, result: BacktestResult,
        master_code: str, strategy_text: str,
    ) -> list[str]:
        files: list[str] = []
        # Trades CSV
        if result.trades:
            csv_path = run_dir / "trades.csv"
            rows = []
            for t in result.trades:
                rows.append({
                    "trade_id": t.trade_id, "asset": t.asset, "side": t.side,
                    "entry_time": t.entry_time, "entry_price": t.entry_price,
                    "exit_time": t.exit_time, "exit_price": t.exit_price,
                    "quantity": t.quantity, "pnl": round(t.pnl, 2),
                    "pnl_pct": round(t.pnl_pct, 2), "fees": round(t.fees, 4),
                })
            pl.DataFrame(rows).write_csv(csv_path)
            files.append(str(csv_path))

        # Metrics JSON
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(
            {k: v for k, v in result.metrics.__dict__.items()}, indent=2, default=str,
        ))
        files.append(str(metrics_path))

        # Strategy text
        (run_dir / "strategy.txt").write_text(strategy_text)
        files.append(str(run_dir / "strategy.txt"))

        # Master code
        (run_dir / "strategy_code.py").write_text(master_code)
        files.append(str(run_dir / "strategy_code.py"))

        # Equity CSV
        if result.equity_curve:
            eq_path = run_dir / "portfolio_value.csv"
            pl.DataFrame(result.equity_curve).write_csv(eq_path)
            files.append(str(eq_path))

        # Report
        report_path = run_dir / "report.csv"
        m = result.metrics
        report_data = [
            {"metric": k, "value": str(v)}
            for k, v in m.__dict__.items()
        ]
        pl.DataFrame(report_data).write_csv(report_path)
        files.append(str(report_path))

        self._cache[result.run_id] = result
        return files


class ChatHistoryManager:
    """Manages conversation state."""

    def __init__(self) -> None:
        self._messages: list[ChatMessage] = []
        self._current_result: BacktestResult | None = None

    def add(self, role: str, content: str, **kwargs: Any) -> ChatMessage:
        msg = ChatMessage(role=role, content=content, **kwargs)
        self._messages.append(msg)
        return msg

    def get_history(self) -> list[ChatMessage]:
        return list(self._messages)

    def get_api_messages(self, limit: int = 20) -> list[dict]:
        recent = self._messages[-limit:]
        return [{"role": m.role, "content": m.content} for m in recent]

    @property
    def current_result(self) -> BacktestResult | None:
        return self._current_result

    @current_result.setter
    def current_result(self, val: BacktestResult | None) -> None:
        self._current_result = val

    def clear(self) -> None:
        self._messages.clear()
        self._current_result = None


# ═══════════════════════════════════════════════════════════════════
# SECTION 12 — PROGRESS & EVENT SYSTEM
# ═══════════════════════════════════════════════════════════════════

class ProgressManager:
    """Thread-safe event queue for UI updates."""

    def __init__(self) -> None:
        self._queue: queue.Queue = queue.Queue()
        self._subscribers: dict[EventType, list[Callable]] = {}

    def emit(self, event_type: EventType, data: dict | None = None) -> None:
        self._queue.put((event_type, data or {}))

    def subscribe(self, event_type: EventType, callback: Callable) -> None:
        self._subscribers.setdefault(event_type, []).append(callback)

    def process_pending(self, max_events: int = 20) -> None:
        processed = 0
        while processed < max_events:
            try:
                event_type, data = self._queue.get_nowait()
                for cb in self._subscribers.get(event_type, []):
                    try:
                        cb(data)
                    except Exception as e:
                        get_logger("events").error("Event handler error: %s", e)
                processed += 1
            except queue.Empty:
                break


# ═══════════════════════════════════════════════════════════════════
# SECTION 13 — CENTRAL ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════

class OpenBacktestOrchestrator:
    """Central controller coordinating all application components."""

    def __init__(self, app_config: AppConfig) -> None:
        self.config = app_config
        self.openrouter: OpenRouterClient | None = None
        self.data_mgr = CCXTDataManager(app_config.exchange)
        self.polars_mgr = PolarsDataManager()
        self.indicator_mgr = IndicatorManager()
        self.backtest_mgr = BacktestManager()
        self.metrics_mgr = MetricsManager()
        self.viz_mgr = VisualizationManager()
        self.result_mgr = ResultManager()
        self.chat_mgr = ChatHistoryManager()
        self.code_mgr = MasterCodeManager()
        self.progress = ProgressManager()
        self.current_result: BacktestResult | None = None
        self._log = get_logger("orchestrator")

        if app_config.api_key:
            self.openrouter = OpenRouterClient(app_config.api_key, app_config.model)

    def set_api_key(self, key: str) -> None:
        self.config.api_key = key
        self.openrouter = OpenRouterClient(key, self.config.model)

    def set_model(self, model: str) -> None:
        self.config.model = model
        if self.openrouter:
            self.openrouter.model = model

    def classify_and_route(self, user_message: str) -> UserIntent:
        if not self.openrouter:
            return UserIntent.GENERAL_CHAT
        return self.openrouter.classify_intent(user_message)

    def run_full_pipeline(
        self, bt_config: BacktestConfig,
        cancel_check: Callable | None = None,
    ) -> BacktestResult:
        """Execute the complete backtest pipeline."""
        run_id = f"RUN_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        start_time = time.time()

        # 1. Initialize
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.INITIALIZING.value, "message": "Initializing...", "percent": 0,
        })

        # 2. Check AI connection
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.CHECKING_API.value, "message": "Checking AI connection...", "percent": 5,
        })
        if self.openrouter:
            ok, msg = self.openrouter.validate_connection()
            if not ok:
                raise RuntimeError(f"AI connection failed: {msg}")
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.CHECKING_API.value, "message": "AI connected ✓", "percent": 8,
        })

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        # 3. Download data
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.DOWNLOADING.value, "message": "Connecting to exchange...", "percent": 10,
        })
        self.data_mgr = CCXTDataManager(bt_config.exchange)
        self.data_mgr.initialize()
        self.data_mgr.load_markets()

        datasets: dict[str, pl.DataFrame] = {}
        total_assets = len(bt_config.assets)
        for i, symbol in enumerate(bt_config.assets):
            if cancel_check and cancel_check():
                raise RuntimeError("Cancelled")
            pct = 12 + (i / total_assets) * 30
            self.progress.emit(EventType.PROGRESS, {
                "stage": Stage.DOWNLOADING.value,
                "message": f"Downloading {symbol}...", "percent": pct,
                "asset": symbol,
            })

            def _dl_cb(count: int, sym: str) -> None:
                self.progress.emit(EventType.PROGRESS, {
                    "stage": Stage.DOWNLOADING.value,
                    "message": f"Downloading {sym}... {count} candles", "percent": pct,
                })

            
            start_ts, end_ts = None, None
            if bt_config.start_date:
                start_ts = int(datetime.combine(bt_config.start_date, datetime.min.time(), tzinfo=timezone.utc).timestamp() * 1000)
            if bt_config.end_date:
                end_ts = int(datetime.combine(bt_config.end_date, datetime.max.time(), tzinfo=timezone.utc).timestamp() * 1000)

            df = self.data_mgr.download(
                symbol, bt_config.timeframe,
                start=start_ts, end=end_ts,
                progress_callback=_dl_cb,
                cancel_check=cancel_check,
            )
            df = self.polars_mgr.normalize(df, symbol)
            anomalies = self.polars_mgr.validate(df, symbol, bt_config.timeframe)
            if anomalies:
                self.progress.emit(EventType.PROGRESS, {
                    "stage": Stage.VALIDATING_DATA.value,
                    "message": f"{symbol}: {len(anomalies)} warnings", "percent": pct + 2,
                })
            df = self.polars_mgr.clean(df)
            datasets[symbol] = df
            self.progress.emit(EventType.PROGRESS, {
                "stage": Stage.DOWNLOADING.value,
                "message": f"{symbol}: {len(df)} candles ✓", "percent": pct + 5,
                "asset": symbol,
            })

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        # 4. AI Strategy interpretation
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.INTERPRETING_STRATEGY.value,
            "message": "Sending strategy to AI...", "percent": 48,
        })

        if bt_config.strategy_text and self.openrouter:
            context = {
                "assets": bt_config.assets,
                "timeframe": bt_config.timeframe,
                "initial_capital": bt_config.initial_capital,
                "position_size_pct": bt_config.position_size_pct,
            }
            interpreter = StrategyInterpreter(self.openrouter, self.code_mgr)
            ok, code, errors = interpreter.interpret(bt_config.strategy_text, context)
            if not ok:
                raise RuntimeError(f"Strategy generation failed: {'; '.join(errors)}")
            self.code_mgr.apply_ai_code(code)

        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.VALIDATING_CODE.value,
            "message": "Code validated ✓", "percent": 55,
        })

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        # 5. Compute indicators
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.RUNNING_BACKTEST.value,
            "message": "Computing indicators...", "percent": 58,
        })
        indicator_configs = self.code_mgr.extract_indicator_config()
        indicators_map: dict[str, dict[str, np.ndarray]] = {}
        for symbol, df in datasets.items():
            indicators_map[symbol] = self.indicator_mgr.compute_multiple(df, indicator_configs)

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        # 6. Run backtest
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.RUNNING_BACKTEST.value,
            "message": "Running backtest engine...", "percent": 62,
        })
        strategy = self.code_mgr.create_strategy_instance()

        def _bt_progress(pct: float, msg: str) -> None:
            self.progress.emit(EventType.PROGRESS, {
                "stage": Stage.RUNNING_BACKTEST.value,
                "message": msg, "percent": 62 + pct * 0.18,
            })

        bt_result = self.backtest_mgr.run(
            bt_config, datasets, strategy, indicators_map,
            progress_callback=_bt_progress, cancel_check=cancel_check,
        )

        if cancel_check and cancel_check():
            raise RuntimeError("Cancelled")

        # 7. Analyze
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.ANALYZING.value,
            "message": "Calculating metrics...", "percent": 82,
        })
        metrics = self.metrics_mgr.calculate(
            bt_result["trades"], bt_result["equity_curve"], bt_config.initial_capital,
        )

        # 8. Save results
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.SAVING_RESULTS.value,
            "message": "Saving results...", "percent": 90,
        })
        run_dir = self.result_mgr.create_run_directory(run_id)
        result = BacktestResult(
            run_id=run_id, config=bt_config, metrics=metrics,
            trades=bt_result["trades"], equity_curve=bt_result["equity_curve"],
            run_directory=str(run_dir),
            strategy_hash=bt_config.config_hash(),
            code_hash=self.code_mgr.code_hash,
            datasets=datasets,
        )
        saved = self.result_mgr.save_complete_result(
            run_dir, result, self.code_mgr.current_code, bt_config.strategy_text,
        )

        # 9. Generate charts
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.GENERATING_CHARTS.value,
            "message": "Generating dashboard...", "percent": 94,
        })
        chart_files = []
        html_dashboard = self.viz_mgr.generate_dashboard_html(result)
        dash_path = run_dir / "dashboard.html"
        try:
            dash_path.write_text(html_dashboard, encoding="utf-8")
            chart_files.append(str(dash_path))
        except Exception as e:
            self._log.error("Failed to save dashboard html: %s", e)

        result.files = saved + chart_files
        self.current_result = result
        self.chat_mgr.current_result = result

        elapsed = time.time() - start_time
        self.progress.emit(EventType.PROGRESS, {
            "stage": Stage.COMPLETE.value,
            "message": f"Complete! {len(bt_result['trades'])} trades, "
                       f"${metrics.net_pnl:,.2f} P&L ({metrics.return_pct:.2f}%)",
            "percent": 100, "elapsed": elapsed,
        })

        return result

    def handle_followup(self, user_message: str) -> str:
        """Handle follow-up chat messages using existing results."""
        if not self.openrouter:
            return "OpenRouter not connected. Please configure your API key."

        intent = self.classify_and_route(user_message)

        if intent == UserIntent.ANALYTICAL_QUERY and self.current_result:
            # Answer from existing data
            m = self.current_result.metrics
            summary = json.dumps({k: v for k, v in m.__dict__.items()}, default=str, indent=2)
            return self.openrouter.analyze_results(summary, user_message)

        elif intent == UserIntent.GENERAL_CHAT:
            self.chat_mgr.add("user", user_message)
            messages = self.chat_mgr.get_api_messages()
            response = self.openrouter.chat_completion(messages, temperature=0.7, max_tokens=1500)
            self.chat_mgr.add("assistant", response)
            return response

        return "To modify the strategy, please use the Backtest page to configure and run a new backtest."

    def get_metrics_summary(self) -> str:
        if not self.current_result:
            return "No backtest results available."
        m = self.current_result.metrics
        return (
            f"Net P&L: ${m.net_pnl:,.2f}\n"
            f"Return: {m.return_pct:+.2f}%\n"
            f"Total Trades: {m.total_trades}\n"
            f"Win Rate: {m.win_rate:.1f}%\n"
            f"Profit Factor: {m.profit_factor:.2f}\n"
            f"Sharpe: {m.sharpe_ratio:.2f}\n"
            f"Sortino: {m.sortino_ratio:.2f}\n"
            f"Max Drawdown: {m.max_drawdown_pct:.2f}%\n"
            f"Total Charges: ${m.total_charges:,.2f}"
        )


# ═══════════════════════════════════════════════════════════════════
# SECTION 14 — BACKGROUND WORKER
# ═══════════════════════════════════════════════════════════════════

class BacktestWorker:
    """Runs the full pipeline in a background thread."""

    def __init__(self, orchestrator: OpenBacktestOrchestrator, config: BacktestConfig) -> None:
        self._orch = orchestrator
        self._config = config
        self._thread: threading.Thread | None = None
        self._cancelled = False
        self._log = get_logger("worker")

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        if self.is_running:
            return
        self._cancelled = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def cancel(self) -> None:
        self._cancelled = True

    def _run(self) -> None:
        self._orch.progress.emit(EventType.RUN_STARTED, {"run_id": "pending"})
        try:
            result = self._orch.run_full_pipeline(
                self._config, cancel_check=lambda: self._cancelled,
            )
            self._orch.progress.emit(EventType.RUN_COMPLETE, {
                "result": result, "run_id": result.run_id,
            })
        except Exception as e:
            tb = traceback.format_exc()
            self._log.error("Pipeline failed: %s\n%s", e, tb)
            evt = EventType.RUN_CANCELLED if self._cancelled else EventType.RUN_FAILED
            self._orch.progress.emit(evt, {"error": str(e), "traceback": tb})


# ═══════════════════════════════════════════════════════════════════
# SECTION 15 — PYSIDE6 UI
# ═══════════════════════════════════════════════════════════════════

from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QLineEdit, QComboBox, QCheckBox,
    QProgressBar, QStackedWidget, QFrame, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView, QDialog, QSplitter, QSpacerItem,
    QSizePolicy, QGroupBox, QGridLayout, QDoubleSpinBox, QSpinBox, QListWidget, QListWidgetItem, QDateEdit,
    QTabWidget, QInputDialog
)
from PySide6.QtCore import Qt, QTimer, Signal, QSize, QUrl, QDate
from PySide6.QtGui import QFont, QColor, QIcon, QPalette, QFontDatabase, QDesktopServices
from PySide6.QtCore import QUrl

try:
    from PySide6.QtWebEngineWidgets import QWebEngineView
    from PySide6.QtWebEngineCore import QWebEnginePage
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False


# ── Theme Constants ──────────────────────────────────────────────

class T_Light:
    """Design tokens (Light Theme matching mockups)."""
    BG_MAIN = "#fefefe"
    BG_SIDEBAR = "#f8fafc"
    BG_SURFACE = "#ffffff"
    BG_PANEL = "#f8fafc"
    BG_INPUT = "#f1f5f9"
    BG_HOVER = "#e2e8f0"
    PRIMARY = "#3b82f6"
    PRIMARY_HOVER = "#2563eb"
    PRIMARY_LIGHT = "#eff6ff"
    POSITIVE = "#059669"
    POSITIVE_BG = "#ecfdf5"
    NEGATIVE = "#dc2626"
    NEGATIVE_BG = "#fef2f2"
    WARNING = "#d97706"
    TEXT_PRIMARY = "#0f172a"
    TEXT_SECONDARY = "#334155"
    TEXT_MUTED = "#64748b"
    BORDER = "#e2e8f0"
    BORDER_LIGHT = "#f1f5f9"
    CORNER = 12
    FONT_FAMILY = "Inter, Segoe UI, Roboto, sans-serif"

class T_Dark:
    """Design tokens (Dark Theme)."""
    BG_MAIN = "#000000"
    BG_SIDEBAR = "#000000"
    BG_SURFACE = "#000000"
    BG_PANEL = "#000000"
    BG_INPUT = "#000000"
    BG_HOVER = "#111111"
    PRIMARY = "#3b82f6"
    PRIMARY_HOVER = "#2563eb"
    PRIMARY_LIGHT = "#1e3a8a"
    POSITIVE = "#10b981"
    POSITIVE_BG = "#064e3b"
    NEGATIVE = "#ef4444"
    NEGATIVE_BG = "#7f1d1d"
    WARNING = "#f59e0b"
    TEXT_PRIMARY = "#f8fafc"
    TEXT_SECONDARY = "#cbd5e1"
    TEXT_MUTED = "#94a3b8"
    BORDER = "#334155"
    BORDER_LIGHT = "#1e293b"
    CORNER = 12
    FONT_FAMILY = "Inter, Segoe UI, Roboto, sans-serif"

T = T_Light

def get_global_qss(t):
    return f"""
QWidget {{
    font-family: {t.FONT_FAMILY};
    color: {t.TEXT_PRIMARY};
    background-color: {t.BG_MAIN};
}}
QMainWindow {{
    background-color: {t.BG_MAIN};
}}
QPushButton {{
    background-color: {t.PRIMARY};
    color: #ffffff;
    border: none;
    border-radius: 8px;
    padding: 8px 18px;
    font-size: 13px;
    font-weight: 600;
}}
QPushButton:hover {{
    background-color: {t.PRIMARY_HOVER};
}}
QPushButton:pressed {{
    background-color: #1d4ed8;
}}
QPushButton:disabled {{
    background-color: {t.BG_HOVER};
    color: {t.TEXT_MUTED};
}}
QPushButton[class="secondary"] {{
    background-color: {t.BG_SURFACE};
    border: 1px solid {t.BORDER};
}}
QPushButton[class="secondary"]:hover {{
    background-color: {t.BG_HOVER};
}}
QPushButton[class="danger"] {{
    background-color: {t.NEGATIVE};
}}
QPushButton[class="danger"]:hover {{
    background-color: #dc2626;
}}
QPushButton[class="nav"] {{
    background-color: transparent;
    color: {t.TEXT_MUTED};
    text-align: left;
    padding: 12px 16px;
    border-radius: 8px;
    font-size: 13px;
    font-weight: 500;
}}
QPushButton[class="nav"]:hover {{
    background-color: {t.BG_HOVER};
    color: {t.TEXT_PRIMARY};
}}
QPushButton[class="nav"][active="true"] {{
    background-color: rgba(59,130,246,0.15);
    color: {t.PRIMARY};
    font-weight: 600;
}}
QLineEdit, QTextEdit, QComboBox, QDoubleSpinBox, QSpinBox {{
    background-color: {t.BG_INPUT};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER};
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 13px;
    selection-background-color: {t.PRIMARY};
}}
QLineEdit:focus, QTextEdit:focus {{
    border-color: {t.PRIMARY};
}}
QComboBox::drop-down {{
    border: none;
    padding-right: 8px;
}}
QComboBox QAbstractItemView {{
    background-color: {t.BG_SURFACE};
    color: {t.TEXT_PRIMARY};
    border: 1px solid {t.BORDER};
    selection-background-color: {t.PRIMARY};
}}
QCheckBox {{
    spacing: 8px;
    font-size: 13px;
}}
QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {t.BORDER};
    background-color: {t.BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background-color: {t.PRIMARY};
    border-color: {t.PRIMARY};
}}
QProgressBar {{
    background-color: {t.BG_INPUT};
    border: none;
    border-radius: 6px;
    height: 12px;
    text-align: center;
    font-size: 10px;
    color: {t.TEXT_MUTED};
}}
QProgressBar::chunk {{
    background-color: {t.PRIMARY};
    border-radius: 6px;
}}
QTableWidget {{
    background-color: {t.BG_SURFACE};
    color: {t.TEXT_PRIMARY};
    border: none;
    gridline-color: {t.BORDER};
    font-size: 12px;
}}
QTableWidget::item {{
    padding: 6px 8px;
}}
QHeaderView::section {{
    background-color: {t.BG_PANEL};
    color: {t.TEXT_SECONDARY};
    border: none;
    border-bottom: 1px solid {t.BORDER};
    padding: 8px;
    font-weight: 600;
    font-size: 12px;
}}
QScrollArea {{
    border: none;
    background-color: transparent;
}}
QScrollBar:vertical {{
    background: {t.BG_MAIN};
    width: 8px;
    border-radius: 4px;
}}
QScrollBar::handle:vertical {{
    background: {t.BORDER};
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QGroupBox {{
    background-color: {t.BG_PANEL};
    border: 1px solid {t.BORDER};
    border-radius: 10px;
    margin-top: 16px;
    padding: 16px;
    font-size: 14px;
    font-weight: 600;
    color: {t.TEXT_PRIMARY};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 16px;
    padding: 0 8px;
}}
QLabel {{
    background-color: transparent;
}}
"""

GLOBAL_QSS_LIGHT = get_global_qss(T_Light)
GLOBAL_QSS_DARK = get_global_qss(T_Dark)
GLOBAL_QSS = GLOBAL_QSS_LIGHT


# ── Utility Widgets ──────────────────────────────────────────────

class Card(QFrame):
    """Styled card container."""
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setStyleSheet(f"""
            Card {{
                background-color: {T.BG_PANEL};
                border: 1px solid {T.BORDER};
                border-radius: {T.CORNER}px;
            }}
        """)


class MetricCard(QFrame):
    """Displays a single metric with label and value."""
    def __init__(self, label: str, value: str, color: str = T.PRIMARY, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(90)
        self.setStyleSheet(f"""
            MetricCard {{
                background-color: {T.BG_PANEL};
                border: 1px solid {T.BORDER};
                border-radius: {T.CORNER}px;
                padding: 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 10, 14, 10)
        lbl = QLabel(label)
        lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px; font-weight: 500;")
        layout.addWidget(lbl)
        self._value_label = QLabel(value)
        self._value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700;")
        layout.addWidget(self._value_label)

    def set_value(self, value: str, color: str | None = None) -> None:
        self._value_label.setText(value)
        if color:
            self._value_label.setStyleSheet(f"color: {color}; font-size: 20px; font-weight: 700;")


class PlotlyWidget(QWidget):
    """Embeds a HTML figure via QWebEngineView."""
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.current_result = None
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if HAS_WEBENGINE:
            self._view = QWebEngineView()
            self._view.setStyleSheet(f"background-color: {T.BG_MAIN};")
            
            # Suppress Tailwind CDN warnings
            class QuietWebPage(QWebEnginePage):
                def javaScriptConsoleMessage(self, level, msg, line, sourceID):
                    if "cdn.tailwindcss.com" in msg or "Tailwind CSS" in msg:
                        return
                    super().javaScriptConsoleMessage(level, msg, line, sourceID)
            self._view.setPage(QuietWebPage(self._view))
            
            layout.addWidget(self._view)
        else:
            lbl = QLabel("QWebEngineView not available.\nInstall PySide6-WebEngine for interactive charts.")
            lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; padding: 20px;")
            lbl.setAlignment(Qt.AlignCenter)
            layout.addWidget(lbl)
            self._view = None

    def load_figure(self, fig: Any) -> None:
        if self._view is None or fig is None:
            return
        html = fig.to_html(include_plotlyjs="cdn", full_html=True)
        self._view.setHtml(html)

    def load_html(self, html: str) -> None:
        if self._view:
            self._view.setHtml(html)


# ── API Key Dialog ───────────────────────────────────────────────

class ApiKeyDialog(QDialog):
    """First-launch dialog for OpenRouter API key entry."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(APP_NAME)
        self.setFixedSize(480, 510)
        self.setStyleSheet(f"""
            QDialog {{
                background-color: {T.BG_SURFACE};
                border-radius: 16px;
            }}
        """)
        self.api_key = ""
        self.demo_mode = False  # True if user chose Demo instead of API key
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 30, 40, 30)
        layout.setSpacing(16)

        # Logo / Title
        title = QLabel(APP_NAME)
        title.setStyleSheet(f"color: {T.PRIMARY}; font-size: 28px; font-weight: 800;")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Connect your AI engine")
        subtitle.setStyleSheet(f"color: {T.TEXT_SECONDARY}; font-size: 14px;")
        subtitle.setAlignment(Qt.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(12)

        # Saved Keys combo
        key_label = QLabel("Saved API Keys")
        key_label.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px; font-weight: 600;")
        layout.addWidget(key_label)

        self._key_combo = QComboBox()
        self._key_combo.setFixedHeight(36)
        self._key_combo.setStyleSheet(f"QComboBox {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 8px; font-size: 13px; }}")
        self._key_combo.currentIndexChanged.connect(self._on_key_selected)
        layout.addWidget(self._key_combo)

        lbl1 = QLabel("OpenRouter API Key")
        lbl1.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px; font-weight: 600; margin-top: 10px;")
        layout.addWidget(lbl1)

        # API Key input
        self._key_input = QLineEdit()
        self._key_input.setPlaceholderText("sk-or-...")
        self._key_input.setEchoMode(QLineEdit.Password)
        self._key_input.setFixedHeight(44)
        self._key_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {T.BG_INPUT};
                border: 1px solid {T.BORDER};
                border-radius: 10px;
                padding: 0 16px;
                font-size: 14px;
                color: {T.TEXT_PRIMARY};
            }}
            QLineEdit:focus {{
                border-color: {T.PRIMARY};
            }}
        """)
        layout.addWidget(self._key_input)
        
        self._populate_combo()

        # Test button
        self._test_btn = QPushButton("⚡  Test && Continue")
        self._test_btn.setFixedHeight(44)
        self._test_btn.setCursor(Qt.PointingHandCursor)
        self._test_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {T.PRIMARY};
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 15px;
                font-weight: 700;
            }}
            QPushButton:hover {{ background-color: {T.PRIMARY_HOVER}; }}
            QPushButton:disabled {{ background-color: {T.BG_HOVER}; color: {T.TEXT_MUTED}; }}
        """)
        self._test_btn.clicked.connect(self._on_test)
        layout.addWidget(self._test_btn)

        # OR divider
        or_row = QHBoxLayout()
        or_row.setSpacing(8)
        line1 = QFrame()
        line1.setFrameShape(QFrame.HLine)
        line1.setStyleSheet(f"color: {T.BORDER};")
        or_lbl = QLabel("or")
        or_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        or_lbl.setAlignment(Qt.AlignCenter)
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet(f"color: {T.BORDER};")
        or_row.addWidget(line1, 1)
        or_row.addWidget(or_lbl)
        or_row.addWidget(line2, 1)
        layout.addLayout(or_row)

        # Demo button
        self._demo_btn = QPushButton("🎬  Try Demo  (no API key needed)")
        self._demo_btn.setFixedHeight(44)
        self._demo_btn.setCursor(Qt.PointingHandCursor)
        self._demo_btn.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0f766e, stop:1 #059669);
                color: white;
                border: none;
                border-radius: 10px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0d9488, stop:1 #10b981);
            }}
        """)
        self._demo_btn.clicked.connect(self._on_demo)
        layout.addWidget(self._demo_btn)

        # Status
        self._status = QLabel("")
        self._status.setAlignment(Qt.AlignCenter)
        self._status.setStyleSheet(f"font-size: 13px; color: {T.TEXT_MUTED};")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        layout.addStretch()

        btn_row = QHBoxLayout()
        self._dark_toggle_btn = QPushButton("🌙 Toggle Dark Mode")
        self._dark_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._dark_toggle_btn.setStyleSheet(f"background: {T.BG_PANEL}; color: {T.TEXT_PRIMARY}; border: 1px solid {T.BORDER}; border-radius: 6px; padding: 4px 8px;")
        self._dark_toggle_btn.clicked.connect(self._toggle_dark_mode)
        btn_row.addWidget(self._dark_toggle_btn)

        self._dm_btn = QPushButton("DM me for fun")
        self._dm_btn.setCursor(Qt.PointingHandCursor)
        self._dm_btn.setStyleSheet(f"color: {T.PRIMARY}; border: none; font-size: 12px; text-decoration: underline; background: transparent;")
        self._dm_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://kowsiganr.github.io/")))
        btn_row.addWidget(self._dm_btn)
        layout.addLayout(btn_row)

        layout.addStretch()

    def _toggle_dark_mode(self):
        if self.parent() and hasattr(self.parent(), "_toggle_dark_mode"):
            self.parent()._toggle_dark_mode()
    def _populate_combo(self):
        store = get_key_store()
        keys = store.get_all_keys()
        data = store._load_data()
        active = data.get("active")
        
        self._key_combo.blockSignals(True)
        self._key_combo.clear()
        self._key_combo.addItem("Add New Key...", "")
        for name, key_val in keys.items():
            self._key_combo.addItem(name, key_val)
            
        if active and active in keys:
            idx = self._key_combo.findText(active)
            if idx >= 0:
                self._key_combo.setCurrentIndex(idx)
                self._key_input.setText(keys[active])
        else:
            self._key_combo.setCurrentIndex(0)
            self._key_input.setText("")
        self._key_combo.blockSignals(False)

    def _on_key_selected(self, index):
        name = self._key_combo.itemText(index)
        key_val = self._key_combo.itemData(index)
        if key_val:
            self._key_input.setText(key_val)
            get_key_store().set_active_key(name)
        else:
            self._key_input.setText("")

    def _on_test(self) -> None:
        key = self._key_input.text().strip()
        if not key:
            self._status.setStyleSheet(f"font-size: 13px; color: {T.WARNING};")
            self._status.setText("Please enter your API key")
            return

        self._test_btn.setEnabled(False)
        self._status.setStyleSheet(f"font-size: 13px; color: {T.TEXT_MUTED};")
        self._status.setText("Testing connection...")
        QApplication.processEvents()

        client = OpenRouterClient(key)
        ok, msg = client.validate_connection()

        if ok:
            self._status.setStyleSheet(f"font-size: 13px; color: {T.POSITIVE};")
            self._status.setText("✓  OpenRouter connected")
            self.api_key = key
            self.demo_mode = False
            
            store = get_key_store()
            keys = store.get_all_keys()
            existing_name = None
            for n, v in keys.items():
                if v == key:
                    existing_name = n
                    break
            
            if existing_name:
                store.set_active_key(existing_name)
            else:
                from PySide6.QtWidgets import QInputDialog
                name, ok_input = QInputDialog.getText(self, "Save API Key", "Enter a name for this API key:", text="My API Key")
                if ok_input and name.strip():
                    store.save_key(key, name.strip())
                else:
                    store.save_key(key, "Default Key")
                    
            QTimer.singleShot(800, self.accept)
        else:
            self._status.setStyleSheet(f"font-size: 13px; color: {T.NEGATIVE};")
            self._status.setText(f"✗  {msg}")
            self._test_btn.setEnabled(True)


    def _filter_assets(self, text: str) -> None:
        text = text.upper()
        for i in range(self._asset_list.count()):
            item = self._asset_list.item(i)
            item.setHidden(text not in item.text())

    def _add_custom_asset(self) -> None:
        asset = self._custom_asset.text().strip().upper()
        if asset and asset not in self._asset_checks:
            item = QListWidgetItem(asset)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            self._asset_list.addItem(item)
            self._asset_checks[asset] = item
            self._custom_asset.clear()

    def _on_demo(self) -> None:
        """Enter the app in demo mode (no API key required)."""
        self.demo_mode = True
        self.api_key = ""
        self.accept()


class SettingsDialog(QDialog):
    """Central settings dialog with tabs for API, General, and Author."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setFixedSize(500, 400)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {T.BG_SURFACE}; border-radius: 12px; }}
            QTabWidget::pane {{ border: 1px solid {T.BORDER}; border-radius: 8px; background: {T.BG_MAIN}; }}
            QTabBar::tab {{ background: {T.BG_PANEL}; color: {T.TEXT_MUTED}; padding: 10px 20px; font-weight: 600; border-top-left-radius: 8px; border-top-right-radius: 8px; border: 1px solid transparent; }}
            QTabBar::tab:selected {{ background: {T.BG_MAIN}; color: {T.PRIMARY}; border: 1px solid {T.BORDER}; border-bottom-color: {T.BG_MAIN}; }}
        """)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)

        tabs = QTabWidget()
        
        # --- TAB 1: API Settings ---
        api_tab = QWidget()
        api_layout = QVBoxLayout(api_tab)
        api_layout.setContentsMargins(24, 24, 24, 24)
        api_layout.setSpacing(16)
        
        lbl0 = QLabel("Saved API Keys")
        lbl0.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-weight: 700;")
        api_layout.addWidget(lbl0)
        
        self.key_combo = QComboBox()
        self.key_combo.setFixedHeight(36)
        self.key_combo.setStyleSheet(f"QComboBox {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 8px; font-size: 13px; }}")
        self.key_combo.currentIndexChanged.connect(self._on_key_selected)
        api_layout.addWidget(self.key_combo)
        
        lbl1 = QLabel("OpenRouter API Key")
        lbl1.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-weight: 700; margin-top: 10px;")
        api_layout.addWidget(lbl1)
        
        self.key_input = QLineEdit()
        self.key_input.setPlaceholderText("sk-or-...")
        self.key_input.setEchoMode(QLineEdit.Password)
        self.key_input.setFixedHeight(40)
        self.key_input.setStyleSheet(f"QLineEdit {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 10px; font-size: 13px; }}")
        api_layout.addWidget(self.key_input)
        
        save_btn = QPushButton("Save Key")
        save_btn.setFixedHeight(40)
        save_btn.setStyleSheet(f"QPushButton {{ background: {T.PRIMARY}; color: white; border: none; border-radius: 8px; font-weight: 600; }} QPushButton:hover {{ background: {T.PRIMARY_HOVER}; }}")
        save_btn.clicked.connect(self._save_api_key)
        api_layout.addWidget(save_btn)
        
        self.api_status = QLabel("")
        self.api_status.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
        api_layout.addWidget(self.api_status)
        api_layout.addStretch()
        tabs.addTab(api_tab, "🔑 API")
        
        self._populate_combo()

        # --- TAB 2: Author ---
        author_tab = QWidget()
        author_layout = QVBoxLayout(author_tab)
        author_layout.setContentsMargins(24, 24, 24, 24)
        author_layout.setSpacing(16)
        author_layout.setAlignment(Qt.AlignCenter)
        
        a_lbl = QLabel(f"<b>{APP_NAME}</b> v{APP_VERSION}")
        a_lbl.setStyleSheet(f"font-size: 18px; color: {T.PRIMARY};")
        a_lbl.setAlignment(Qt.AlignCenter)
        author_layout.addWidget(a_lbl)
        
        c_lbl = QLabel(f"Created by: {APP_CREATOR}")
        c_lbl.setStyleSheet(f"font-size: 14px; color: {T.TEXT_SECONDARY};")
        c_lbl.setAlignment(Qt.AlignCenter)
        author_layout.addWidget(c_lbl)
        
        btn_row = QHBoxLayout()
        btn_row.setSpacing(16)
        
        site_btn = QPushButton("🌐 Visit Website")
        site_btn.setFixedHeight(40)
        site_btn.setCursor(Qt.PointingHandCursor)
        site_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://kowsiganr.github.io")))
        btn_row.addWidget(site_btn)
        
        dm_btn = QPushButton("💬 DM for fun!")
        dm_btn.setFixedHeight(40)
        dm_btn.setCursor(Qt.PointingHandCursor)
        dm_btn.setStyleSheet(f"background: rgba(59,130,246,0.15); color: {T.TEXT_PRIMARY}; border: 1px solid {T.PRIMARY};")
        dm_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://kowsiganr.github.io")))
        btn_row.addWidget(dm_btn)
        
        author_layout.addLayout(btn_row)
        author_layout.addStretch()
        tabs.addTab(author_tab, "👨‍💻 Author")
        
        layout.addWidget(tabs)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.setFixedHeight(36)
        close_btn.setStyleSheet(f"background: {T.BG_PANEL}; color: {T.TEXT_PRIMARY}; border: 1px solid {T.BORDER};")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, 0, Qt.AlignRight)

    def _populate_combo(self):
        store = get_key_store()
        keys = store.get_all_keys()
        
        data = store._load_data()
        active = data.get("active")
        
        self.key_combo.blockSignals(True)
        self.key_combo.clear()
        self.key_combo.addItem("Add New Key...", "")
        for name, key_val in keys.items():
            self.key_combo.addItem(name, key_val)
            
        if active and active in keys:
            idx = self.key_combo.findText(active)
            if idx >= 0:
                self.key_combo.setCurrentIndex(idx)
                self.key_input.setText(keys[active])
        else:
            self.key_combo.setCurrentIndex(0)
            self.key_input.setText("")
        self.key_combo.blockSignals(False)

    def _on_key_selected(self, index):
        name = self.key_combo.itemText(index)
        key_val = self.key_combo.itemData(index)
        if key_val:
            self.key_input.setText(key_val)
            get_key_store().set_active_key(name)
            self.api_status.setText(f"Active key set to: {name}")
            self.api_status.setStyleSheet(f"color: {T.POSITIVE}; font-size: 12px;")
        else:
            self.key_input.setText("")
            self.api_status.setText("")

    def _save_api_key(self):
        k = self.key_input.text().strip()
        if k:
            store = get_key_store()
            keys = store.get_all_keys()
            existing_name = None
            for n, v in keys.items():
                if v == k:
                    existing_name = n
                    break
            
            if existing_name:
                store.set_active_key(existing_name)
                self._populate_combo()
                self.api_status.setText(f"Key '{existing_name}' activated.")
                self.api_status.setStyleSheet(f"color: {T.POSITIVE}; font-size: 12px;")
            else:
                name, ok = QInputDialog.getText(self, "Save API Key", "Enter a name for this API key:", text="My API Key")
                if ok and name.strip():
                    store.save_key(k, name.strip())
                    self._populate_combo()
                    self.api_status.setText(f"API Key '{name.strip()}' saved successfully!")
                    self.api_status.setStyleSheet(f"color: {T.POSITIVE}; font-size: 12px;")
        else:
            self.api_status.setText("Please enter a key.")
            self.api_status.setStyleSheet(f"color: {T.NEGATIVE}; font-size: 12px;")


# ── Sidebar ──────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════

# ═══════════════════════════════════════════════════════════════════
# SECTION 10 — CHATGPT-STYLE UI
# ═══════════════════════════════════════════════════════════════════

class HistorySidebar(QWidget):
    """Left sidebar like ChatGPT: logo, new backtest, history list."""
    new_backtest_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(260)
        self.setStyleSheet(f"background: {T.BG_SIDEBAR}; border-right: 1px solid {T.BORDER};")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(8)

        brand = QLabel(f"⚡  {APP_NAME}")
        brand.setStyleSheet(f"color: {T.PRIMARY}; font-size: 16px; font-weight: 800; padding: 4px;")
        layout.addWidget(brand)
        ver = QLabel(f"v{APP_VERSION}")
        ver.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 10px; padding-left: 4px;")
        layout.addWidget(ver)
        layout.addSpacing(8)

        new_btn = QPushButton("  +   New Backtest")
        new_btn.setFixedHeight(44)
        new_btn.setCursor(Qt.PointingHandCursor)
        new_btn.setStyleSheet(
            f"QPushButton {{ background: {T.PRIMARY}; color: white; border: none;"
            f" border-radius: 10px; font-size: 14px; font-weight: 700; text-align: left; padding: 0 16px; }}"
            f"QPushButton:hover {{ background: {T.PRIMARY_HOVER}; }}"
        )
        new_btn.clicked.connect(self.new_backtest_clicked)
        layout.addWidget(new_btn)

        hist_lbl = QLabel("Recent Backtests")
        hist_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 10px; font-weight: 700; padding: 8px 4px 4px 4px; letter-spacing: 1px;")
        layout.addWidget(hist_lbl)

        self._hist_scroll = QScrollArea()
        self._hist_scroll.setWidgetResizable(True)
        self._hist_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        hist_content = QWidget()
        hist_content.setStyleSheet("background: transparent;")
        self._hist_layout = QVBoxLayout(hist_content)
        self._hist_layout.setContentsMargins(0, 0, 0, 0)
        self._hist_layout.setSpacing(2)
        self._hist_layout.addStretch()
        self._hist_scroll.setWidget(hist_content)
        layout.addWidget(self._hist_scroll, 1)

        layout.addStretch()
        self.settings_btn = QPushButton("⚙   Settings")
        self.settings_btn.setFixedHeight(36)
        self.settings_btn.setCursor(Qt.PointingHandCursor)
        self.settings_btn.setStyleSheet(f"background: transparent; color: {T.TEXT_MUTED}; border: none; font-size: 13px; font-weight: 600; text-align: left; padding: 0 12px;")
        layout.addWidget(self.settings_btn)

        # Author Attribution (Restored for visibility)
        creator_lbl = QLabel(f'<a href="https://kowsiganr.github.io" style="color:{T.PRIMARY}; text-decoration:none; font-weight:600;">{APP_CREATOR}</a>')
        creator_lbl.setOpenExternalLinks(True)
        creator_lbl.setStyleSheet(f"font-size: 11px; padding: 4px 12px;")
        creator_lbl.setWordWrap(True)
        layout.addWidget(creator_lbl)

    def add_history_item(self, title: str, subtitle: str) -> None:
        btn = QPushButton()
        btn.setFixedHeight(48)
        btn.setCursor(Qt.PointingHandCursor)
        btn.setStyleSheet(
            f"QPushButton {{ background: transparent; border: none; border-radius: 8px; padding: 0; text-align: left; }}"
            f"QPushButton:hover {{ background: {T.BG_HOVER}; }}"
        )
        row_w = QWidget()
        row_w.setStyleSheet("background: transparent;")
        wl = QVBoxLayout(row_w)
        wl.setContentsMargins(8, 4, 8, 4)
        wl.setSpacing(2)
        t = QLabel(title)
        t.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-size: 12px; font-weight: 600; background: transparent;")
        wl.addWidget(t)
        s = QLabel(subtitle)
        s.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 10px; background: transparent;")
        wl.addWidget(s)
        count = self._hist_layout.count()
        self._hist_layout.insertWidget(count - 1, row_w)


class _UserBubble(QWidget):
    """Right-aligned user message bubble."""
    def __init__(self, text: str, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(80, 6, 16, 6)
        layout.addStretch()
        bubble = QLabel(text)
        bubble.setWordWrap(True)
        bubble.setStyleSheet(
            f"background: {T.PRIMARY}; color: white; border-radius: 18px 18px 4px 18px;"
            f" padding: 10px 16px; font-size: 13px;"
        )
        bubble.setMaximumWidth(540)
        layout.addWidget(bubble)


class _AIMsgWidget(QWidget):
    """Left-aligned AI message with icon and rich HTML content."""
    def __init__(self, sender: str, html_text: str, icon: str = "🤖", parent=None, bg: str = None, border: str = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 6, 80, 6)
        layout.setAlignment(Qt.AlignTop)

        icon_lbl = QLabel(icon)
        icon_lbl.setFixedSize(32, 32)
        icon_lbl.setAlignment(Qt.AlignCenter)
        icon_lbl.setStyleSheet(f"background: {T.PRIMARY}33; border-radius: 16px; font-size: 16px;")
        layout.addWidget(icon_lbl, 0, Qt.AlignTop)

        right = QVBoxLayout()
        right.setSpacing(4)
        right.setContentsMargins(8, 0, 0, 0)

        sender_lbl = QLabel(sender)
        sender_lbl.setStyleSheet(f"color: {T.TEXT_SECONDARY}; font-size: 11px; font-weight: 700;")
        right.addWidget(sender_lbl)

        self._content = QTextEdit()
        self._content.setReadOnly(True)
        self._content.setHtml(html_text)
        self._content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self._content.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._content.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        c_bg = bg or T.BG_SURFACE
        c_border = border or f"1px solid {T.BORDER}"
        self._content.setStyleSheet(
            f"QTextEdit {{ background: {c_bg}; color: {T.TEXT_PRIMARY};"
            f" border: {c_border}; border-radius: 4px 18px 18px 18px;"
            f" padding: 10px 14px; font-size: 13px; }}"
        )
        self._content.document().contentsChanged.connect(self._adjust_height)
        right.addWidget(self._content)
        layout.addLayout(right, 1)
        QTimer.singleShot(0, self._adjust_height)

    def _adjust_height(self):
        doc_height = int(self._content.document().size().height())
        self._content.setFixedHeight(min(max(doc_height + 24, 48), 700))

    def update_html(self, html_text: str):
        self._content.setHtml(html_text)
        QTimer.singleShot(0, self._adjust_height)


class ChatView(QWidget):
    """Scrollable chat area — the heart of the ChatGPT UI."""

    def __init__(self, parent=None):
        super().__init__(parent)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setStyleSheet(f"QScrollArea {{ border: none; background: {T.BG_MAIN}; }}")
        self._content_w = QWidget()
        self._content_w.setStyleSheet(f"background: {T.BG_MAIN};")
        self._msgs = QVBoxLayout(self._content_w)
        self._msgs.setContentsMargins(0, 20, 0, 20)
        self._msgs.setSpacing(4)
        self._msgs.addStretch()
        self._scroll.setWidget(self._content_w)
        outer.addWidget(self._scroll)

        self._progress_widget: _AIMsgWidget | None = None
        self._progress_lines: list[str] = []

    def _scroll_bottom(self):
        QTimer.singleShot(60, lambda: self._scroll.verticalScrollBar().setValue(
            self._scroll.verticalScrollBar().maximum()
        ))

    def _insert(self, widget: QWidget):
        self._msgs.insertWidget(self._msgs.count() - 1, widget)
        self._scroll_bottom()

    def add_user_msg(self, text: str):
        self._progress_widget = None
        self._progress_lines = []
        self._insert(_UserBubble(text))

    def add_ai_msg(self, text: str, icon: str = "🤖", sender: str = "OpenBacktest.ai", color: str = None):
        self._progress_widget = None
        self._progress_lines = []
        safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
        clr = color or T.TEXT_PRIMARY
        html = f'<span style="color:{clr}; font-size:13px;">{safe}</span>'
        w = _AIMsgWidget(sender, html, icon)
        self._insert(w)
        return w

    def start_progress_msg(self):
        self._progress_lines = []
        html = f'<span style="color:{T.TEXT_MUTED}; font-size:13px;">⏳ Starting backtest...</span>'
        w = _AIMsgWidget("System", html, "⚙")
        self._insert(w)
        self._progress_widget = w

    def append_progress_line(self, msg: str, done: bool = False):
        if not self._progress_widget:
            return
        icon_char = "✅" if done else "⏳"
        color = T.POSITIVE if done else T.TEXT_MUTED
        self._progress_lines.append(f'<span style="color:{color};">{icon_char} {msg}</span>')
        html = "<br/>".join(self._progress_lines)
        self._progress_widget.update_html(f'<span style="font-size:13px; font-family: monospace;">{html}</span>')
        self._scroll_bottom()

    def finish_progress(self):
        self._progress_widget = None

    def add_result_card(self, result: "BacktestResult"):
        self._progress_widget = None
        m = result.metrics
        pc = T.POSITIVE if m.net_pnl >= 0 else T.NEGATIVE
        rc = T.POSITIVE if m.return_pct >= 0 else T.NEGATIVE

        # Sleek horizontal summary block matching Image 1
        html = (
            f'<div style="font-family: {T.FONT_FAMILY}; max-width: 700px;">'
            f'<div style="background: {T.PRIMARY_LIGHT}; color: {T.TEXT_PRIMARY}; padding: 12px 16px; border-radius: 8px; margin-bottom: 12px; font-size: 13px;">'
            f'Backtest completed successfully! Here is the summary. Detailed charts are loading below...'
            f'</div>'
            f'<div style="display: flex; flex-direction: row; align-items: center; justify-content: space-between; '
            f'background: {T.BG_SURFACE}; border: 1px solid {T.BORDER}; border-radius: 12px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">'
            
            f'<div style="flex: 1; border-right: 1px solid {T.BORDER}; padding-right: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Net P&amp;L</div>'
            f'<div style="color: {pc}; font-size: 18px; font-weight: 700;">{"+" if m.net_pnl>=0 else ""}${m.net_pnl:,.2f}</div>'
            f'</div>'
            
            f'<div style="flex: 1; border-right: 1px solid {T.BORDER}; padding-left: 12px; padding-right: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Return</div>'
            f'<div style="color: {rc}; font-size: 18px; font-weight: 700;">{m.return_pct:+.2f}%</div>'
            f'</div>'
            
            f'<div style="flex: 1; border-right: 1px solid {T.BORDER}; padding-left: 12px; padding-right: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Total Trades</div>'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 18px; font-weight: 700;">{m.total_trades}</div>'
            f'</div>'
            
            f'<div style="flex: 1; border-right: 1px solid {T.BORDER}; padding-left: 12px; padding-right: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Win Rate</div>'
            f'<div style="color: {T.POSITIVE}; font-size: 18px; font-weight: 700;">{m.win_rate:.2f}%</div>'
            f'</div>'
            
            f'<div style="flex: 1; border-right: 1px solid {T.BORDER}; padding-left: 12px; padding-right: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Max Drawdown</div>'
            f'<div style="color: {T.NEGATIVE}; font-size: 18px; font-weight: 700;">{m.max_drawdown_pct:.2f}%</div>'
            f'</div>'

            f'<div style="flex: 1; padding-left: 12px;">'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 11px; font-weight: 600; margin-bottom: 4px;">Total Charges</div>'
            f'<div style="color: {T.TEXT_PRIMARY}; font-size: 18px; font-weight: 700;">${m.total_fees + m.total_slippage:,.2f}</div>'
            f'</div>'
            
            f'</div>'
            f'</div>'
        )
        w = _AIMsgWidget("OpenBacktest.ai", html, "🤖", bg="transparent", border="none")
        self._insert(w)


class BacktestInputPanel(QWidget):
    """Bottom input panel — like ChatGPT input area with inline config."""
    run_requested = Signal(dict)
    demo_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"QWidget {{ background: {T.BG_SIDEBAR}; border-top: 1px solid {T.BORDER}; }}")
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 16, 24, 24)
        layout.setSpacing(12)

        # Row 1: assets + timeframe + dates + exchange + demo
        r1 = QHBoxLayout()
        r1.setSpacing(8)
        r1.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        assets_lbl = QLabel("Assets:")
        assets_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        r1.addWidget(assets_lbl)

        self._asset_btns: dict[str, QPushButton] = {}
        for asset, sym in [("BTC/USDT", "₿"), ("ETH/USDT", "Ξ"), ("DOGE/USDT", "Ð")]:
            btn = QPushButton(f"{sym} {asset}")
            btn.setCheckable(True)
            btn.setChecked(True)
            btn.setFixedHeight(28)
            btn.setCursor(Qt.PointingHandCursor)
            btn.setStyleSheet(
                f"QPushButton {{ background: {T.BG_INPUT}; color: {T.TEXT_SECONDARY}; border: 1px solid {T.BORDER};"
                f" border-radius: 14px; padding: 0 12px; font-size: 11px; }}"
                f"QPushButton:checked {{ background: #dcfce7; color: #3b82f6; border-color: #86efac; font-weight: 700; }}"
            )
            r1.addWidget(btn)
            self._asset_btns[asset] = btn

        self._other_asset = QLineEdit()
        self._other_asset.setPlaceholderText("+ Add asset")
        self._other_asset.setFixedWidth(100)
        self._other_asset.setFixedHeight(30)
        self._other_asset.setStyleSheet(
            f"border-radius: 15px; padding: 0 12px; font-size: 11px; font-weight: 600;"
            f" background: {T.BG_INPUT}; border: 1px dashed #94a3b8; color: #475569;"
        )
        r1.addWidget(self._other_asset)
        r1.addSpacing(16)

        tf_lbl = QLabel("TF:")
        tf_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        r1.addWidget(tf_lbl)
        self._tf = QComboBox()
        self._tf.addItems(SUPPORTED_TIMEFRAMES)
        self._tf.setCurrentText(DEFAULT_TIMEFRAME)
        self._tf.setFixedHeight(28)
        self._tf.setFixedWidth(60)
        self._tf.setStyleSheet(f"QComboBox {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 8px; font-size: 11px; }}")
        r1.addWidget(self._tf)
        r1.addSpacing(12)

        from_lbl = QLabel("From:")
        from_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        r1.addWidget(from_lbl)
        self._start_date = QDateEdit()
        self._start_date.setCalendarPopup(True)
        self._start_date.setDate(QDate.currentDate().addYears(-1))
        self._start_date.setFixedHeight(28)
        self._start_date.setFixedWidth(95)
        self._start_date.setStyleSheet(f"QDateEdit {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 8px; font-size: 11px; }}")
        r1.addWidget(self._start_date)
        arr = QLabel("→")
        arr.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 12px;")
        r1.addWidget(arr)
        self._end_date = QDateEdit()
        self._end_date.setCalendarPopup(True)
        self._end_date.setDate(QDate.currentDate())
        self._end_date.setFixedHeight(28)
        self._end_date.setFixedWidth(95)
        self._end_date.setStyleSheet(f"QDateEdit {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 8px; font-size: 11px; }}")
        r1.addWidget(self._end_date)
        r1.addStretch()

        exc_lbl = QLabel("Exchange:")
        exc_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        r1.addWidget(exc_lbl)
        self._exchange = QComboBox()
        self._exchange.addItems([e.capitalize() for e in SUPPORTED_EXCHANGES])
        self._exchange.setFixedHeight(28)
        self._exchange.setFixedWidth(90)
        self._exchange.setStyleSheet(f"QComboBox {{ border-radius: 14px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 10px; font-size: 11px; }}")
        r1.addWidget(self._exchange)

        demo_btn = QPushButton("🎬 Demo")
        demo_btn.setFixedHeight(28)
        demo_btn.setCursor(Qt.PointingHandCursor)
        demo_btn.setStyleSheet(
            "QPushButton { background: #0f766e; color: white; border: none; border-radius: 8px; padding: 0 12px; font-size: 11px; font-weight: 700; }"
            "QPushButton:hover { background: #0d9488; }"
        )
        demo_btn.clicked.connect(self.demo_requested)
        r1.addWidget(demo_btn)
        layout.addLayout(r1)

        # Row 2: capital + pos + strategy input + run button
        r2 = QHBoxLayout()
        r2.setSpacing(12)

        adv_col = QVBoxLayout()
        adv_col.setSpacing(6)
        
        cap_row = QHBoxLayout()
        cap_lbl = QLabel("Capital:")
        cap_lbl.setFixedWidth(40)
        cap_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        cap_row.addWidget(cap_lbl)
        self._capital = QLineEdit()
        self._capital.setText(str(DEFAULT_INITIAL_CAPITAL))
        self._capital.setFixedHeight(30)
        self._capital.setFixedWidth(80)
        self._capital.setStyleSheet(f"QLineEdit {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 10px; font-size: 11px; }}")
        cap_row.addWidget(self._capital)
        adv_col.addLayout(cap_row)

        pos_row = QHBoxLayout()
        pos_lbl = QLabel("Pos Size:")
        pos_lbl.setFixedWidth(45)
        pos_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        pos_row.addWidget(pos_lbl)
        self._pos = QLineEdit()
        self._pos.setText(str(int(DEFAULT_POSITION_SIZE_PCT * 100)))
        self._pos.setFixedHeight(30)
        self._pos.setFixedWidth(60)
        self._pos.setStyleSheet(f"QLineEdit {{ border-radius: 8px; border: 1px solid {T.BORDER}; background: {T.BG_INPUT}; padding: 0 10px; font-size: 11px; }}")
        pos_row.addWidget(self._pos)
        adv_col.addLayout(pos_row)
        
        r2.addLayout(adv_col)

        self._strategy_input = QTextEdit()
        self._strategy_input.setPlaceholderText(
            "Describe your strategy or ask anything... e.g. 'Buy when RSI(14) < 30 and price above EMA(50). Exit when RSI > 70 or stop loss 2%'"
        )
        self._strategy_input.setFixedHeight(76)
        self._strategy_input.setStyleSheet(
            f"QTextEdit {{ background: {T.BG_INPUT}; color: {T.TEXT_PRIMARY}; border: 1px solid {T.BORDER};"
            f" border-radius: 16px; padding: 12px 16px; font-size: 13px; }}"
            f"QTextEdit:focus {{ border-color: {T.PRIMARY}; }}"
        )
        r2.addWidget(self._strategy_input, 1)

        self._run_btn = QPushButton("▶  Run")
        self._run_btn.setFixedHeight(76)
        self._run_btn.setFixedWidth(90)
        self._run_btn.setCursor(Qt.PointingHandCursor)
        self._run_btn.setStyleSheet(
            f"QPushButton {{ background: #3b82f6;"
            f" color: white; border: none; border-radius: 16px; font-size: 15px; font-weight: 800; }}"
            f"QPushButton:hover {{ background: #2563eb; }}"
            f"QPushButton:disabled {{ background: {T.BG_HOVER}; color: {T.TEXT_MUTED}; }}"
        )
        self._run_btn.clicked.connect(self._on_run)
        r2.addWidget(self._run_btn)
        
        layout.addLayout(r2)

    def _on_run(self):
        assets = [a for a, b in self._asset_btns.items() if b.isChecked()]
        extra = self._other_asset.text().strip().upper()
        if extra and "/" in extra and extra not in assets:
            assets.append(extra)
        if not assets:
            return
        self.run_requested.emit({
            "assets": assets,
            "timeframe": self._tf.currentText(),
            "exchange": self._exchange.currentText().lower(),
            "initial_capital": float(self._capital.text()),
            "position_size_pct": float(self._pos.text()) / 100,
            "strategy_text": self._strategy_input.toPlainText().strip(),
            "maker_fee_pct": DEFAULT_MAKER_FEE_PCT,
            "taker_fee_pct": DEFAULT_TAKER_FEE_PCT,
            "slippage_pct": DEFAULT_SLIPPAGE_PCT,
            "start_date": self._start_date.date().toPython(),
            "end_date": self._end_date.date().toPython(),
        })

    def set_running(self, running: bool):
        self._run_btn.setEnabled(not running)
        self._run_btn.setText("⏳" if running else "▶  Run")


# ═══════════════════════════════════════════════════════════════════
# SECTION 11 — MAIN WINDOW (CHATGPT STYLE)
# ═══════════════════════════════════════════════════════════════════

class MainWindow(QMainWindow):
    """Pure ChatGPT-style main window."""

    def __init__(self, orchestrator: OpenBacktestOrchestrator) -> None:
        super().__init__()
        self._orch = orchestrator
        self._worker: BacktestWorker | None = None
        self._backtest_queue: list[BacktestConfig] = []
        self._last_stage: str = ""
        self.setWindowTitle(APP_TITLE)
        
        # Resizable window but with sensible minimums
        self.setMinimumSize(1100, 700)
        self.resize(1280, 860)
        
        self._build_ui()
        self._connect_events()
        self._start_event_polling()

    def _toggle_dark_mode(self):
        self._is_dark = not getattr(self, "_is_dark", False)
        
        app = QApplication.instance()
        old_t = T_Dark if not self._is_dark else T_Light
        new_t = T_Dark if self._is_dark else T_Light
        
        # 1. Update Global QSS
        app.setStyleSheet(GLOBAL_QSS_DARK if self._is_dark else GLOBAL_QSS_LIGHT)
        
        # 2. Update inline styles via simultaneous regex substitution
        import re
        replacements = {}
        for attr in dir(old_t):
            if not attr.startswith("_") and isinstance(getattr(old_t, attr), str):
                old_val = getattr(old_t, attr).lower()
                new_val = getattr(new_t, attr).lower()
                if old_val.startswith("#") and old_val != new_val:
                    replacements[old_val] = new_val

        if replacements:
            # Match any of the keys, escape them, and use word boundaries if necessary (though # doesn't work with \b, so just match literal)
            # Sort keys by length descending to match longest hex codes first
            sorted_keys = sorted(replacements.keys(), key=len, reverse=True)
            pattern = re.compile("|".join(map(re.escape, sorted_keys)), flags=re.IGNORECASE)
            
            def replacer(match):
                return replacements[match.group(0).lower()]

            for widget in app.allWidgets():
                sheet = widget.styleSheet()
                if sheet:
                    new_sheet = pattern.sub(replacer, sheet)
                    if new_sheet != sheet:
                        widget.setStyleSheet(new_sheet)
                    
        # 3. Update active Dashboards
        for widget in app.allWidgets():
            if isinstance(widget, PlotlyWidget) and getattr(widget, "current_result", None):
                html = self._orch.viz_mgr.generate_dashboard_html(widget.current_result, is_dark=self._is_dark)
                widget.load_html(html)

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        self._sidebar = HistorySidebar()
        root.addWidget(self._sidebar)

        right_w = QWidget()
        right_w.setStyleSheet(f"background: {T.BG_MAIN};")
        right_layout = QVBoxLayout(right_w)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        # Top bar
        topbar = QWidget()
        topbar.setFixedHeight(50)
        topbar.setStyleSheet(f"background: {T.BG_SIDEBAR}; border-bottom: 1px solid {T.BORDER};")
        tbl = QHBoxLayout(topbar)
        tbl.setContentsMargins(24, 0, 24, 0)
        title = QLabel("🤖  AI Backtest Assistant")
        title.setStyleSheet(f"color: {T.TEXT_PRIMARY}; font-size: 14px; font-weight: 700;")
        tbl.addWidget(title)
        tbl.addStretch()

        self._dark_toggle_btn = QPushButton("🌙 Toggle Dark Mode")
        self._dark_toggle_btn.setCursor(Qt.PointingHandCursor)
        self._dark_toggle_btn.setStyleSheet(f"background: {T.BG_PANEL}; color: {T.TEXT_PRIMARY}; border: 1px solid {T.BORDER}; border-radius: 6px; padding: 4px 8px;")
        self._dark_toggle_btn.clicked.connect(self._toggle_dark_mode)
        tbl.addWidget(self._dark_toggle_btn)

        self._dm_btn = QPushButton("DM me for fun")
        self._dm_btn.setCursor(Qt.PointingHandCursor)
        self._dm_btn.setStyleSheet(f"color: {T.PRIMARY}; border: none; font-size: 12px; text-decoration: underline; background: transparent;")
        self._dm_btn.clicked.connect(lambda: QDesktopServices.openUrl(QUrl("https://kowsiganr.github.io/")))
        tbl.addWidget(self._dm_btn)

        self._status_lbl = QLabel("Demo Mode — Enter API key in settings to use custom strategies")
        self._status_lbl.setStyleSheet(f"color: {T.TEXT_MUTED}; font-size: 11px;")
        tbl.addWidget(self._status_lbl)
        right_layout.addWidget(topbar)

        # Main content area (Chat only)
        self._chat = ChatView()
        right_layout.addWidget(self._chat, 1)

        # Input panel
        self._input_panel = BacktestInputPanel()
        right_layout.addWidget(self._input_panel)

        root.addWidget(right_w, 1)

        # Welcome
        self._chat.add_ai_msg(
            "Welcome to OpenBacktest.ai!\n\n"
            "Describe your trading strategy below and click Run — or click 🎬 Demo to see an example.\n\n"
            "I will download live market data, run the backtest, and show you a full analytics report right here in chat.",
            icon="🤖",
            sender="OpenBacktest.ai",
        )

    def _connect_events(self) -> None:
        self._sidebar.new_backtest_clicked.connect(self._on_new_backtest)
        self._sidebar.settings_btn.clicked.connect(self._open_settings)
        self._input_panel.run_requested.connect(self._on_run_backtest)
        self._input_panel.demo_requested.connect(self._on_demo)
        self._orch.progress.subscribe(EventType.PROGRESS, self._on_progress)
        self._orch.progress.subscribe(EventType.RUN_COMPLETE, self._on_complete)
        self._orch.progress.subscribe(EventType.RUN_FAILED, self._on_failed)
        self._orch.progress.subscribe(EventType.RUN_CANCELLED, self._on_cancelled)

    def _open_settings(self) -> None:
        SettingsDialog(self).exec()

    def _start_event_polling(self) -> None:
        self._poll_timer = QTimer(self)
        self._poll_timer.timeout.connect(lambda: self._orch.progress.process_pending(20))
        self._poll_timer.start(100)

    def _on_new_backtest(self):
        self._chat.add_ai_msg(
            "Ready for a new backtest! Configure your assets, timeframe and strategy below, then click Run.",
            icon="🤖",
        )

    def _on_demo(self):
        self._chat.add_user_msg("Run the demo strategy and show me the full report.")
        self._chat.start_progress_msg()
        self._input_panel.set_running(True)
        self._last_stage = ""

        # Enqueue demo assets sequentially
        self._backtest_queue = []
        for asset in ["BTC/USDT", "ETH/USDT", "DOGE/USDT"]:
            config = BacktestConfig(
                assets=[asset],
                exchange="binance",
                timeframe="1d",
                initial_capital=10_000.0,
                position_size_pct=0.90,
                fee_config=FeeConfig(),
                slippage_config=SlippageConfig(),
                strategy_text="",
                start_date=datetime(2018, 1, 1).date(),
                end_date=datetime.now(timezone.utc).date(),
            )
            self._backtest_queue.append(config)
            
        self._process_next_in_queue()

    def _on_run_backtest(self, vals: dict) -> None:
        if self._worker and self._worker.is_running:
            return

        strat = vals.get("strategy_text", "").strip()
        preview = strat[:80] + "..." if len(strat) > 80 else strat or "Default RSI/EMA/SMA strategy"
        self._chat.add_user_msg(
            f"Run backtest: {', '.join(vals['assets'])} | {vals['timeframe']} | {preview}"
        )
        self._input_panel.set_running(True)
        
        self._backtest_queue = []
        for asset in vals["assets"]:
            config = BacktestConfig(
                assets=[asset],  # Single asset per run
                exchange=vals.get("exchange", self._orch.config.exchange),
                timeframe=vals["timeframe"],
                initial_capital=vals["initial_capital"],
                position_size_pct=vals["position_size_pct"],
                fee_config=FeeConfig(
                    maker_fee_pct=vals.get("maker_fee_pct", DEFAULT_MAKER_FEE_PCT),
                    taker_fee_pct=vals.get("taker_fee_pct", DEFAULT_TAKER_FEE_PCT),
                ),
                slippage_config=SlippageConfig(slippage_pct=vals.get("slippage_pct", DEFAULT_SLIPPAGE_PCT)),
                strategy_text=strat,
                start_date=vals.get("start_date"),
                end_date=vals.get("end_date"),
            )
            self._backtest_queue.append(config)
            
        self._process_next_in_queue()

    def _process_next_in_queue(self) -> None:
        if not self._backtest_queue:
            self._input_panel.set_running(False)
            return
            
        config = self._backtest_queue.pop(0)
        self._chat.start_progress_msg()
        self._last_stage = ""
        self._worker = BacktestWorker(self._orch, config)
        self._worker.start()

    def _on_progress(self, data: dict) -> None:
        msg = data.get("message", "")
        stage = data.get("stage", "")
        if not msg:
            return
        done = stage in [Stage.VALIDATING_DATA.value, Stage.ANALYZING.value, Stage.SAVING_RESULTS.value]
        if stage != self._last_stage or "candles" in msg or "✓" in msg:
            self._chat.append_progress_line(msg, done=done)
            if stage:
                self._last_stage = stage

    def _on_complete(self, data: dict) -> None:
        result: BacktestResult = data.get("result")
        if not result:
            QTimer.singleShot(500, self._process_next_in_queue)
            return
            
        self._chat.finish_progress()
        self._chat.add_result_card(result)
        # Load dashboard html inline into chat
        try:
            html = self._orch.viz_mgr.generate_dashboard_html(result, is_dark=getattr(self, "_is_dark", False))
            dash_view = PlotlyWidget()
            dash_view.current_result = result
            # Set a very large minimum height to completely eliminate internal scrolling
            dash_view.setMinimumHeight(1600)
            dash_view.load_html(html)
            
            # Wrap in a widget to add margins and insert into chat
            wrapper = QWidget()
            wrapper_layout = QVBoxLayout(wrapper)
            wrapper_layout.setContentsMargins(16, 8, 80, 24)
            wrapper_layout.addWidget(dash_view)
            
            self._chat._insert(wrapper)
        except Exception as e:
            get_logger("ui").error("Failed to generate dashboard HTML: %s", e)

        self._sidebar.add_history_item(
            f"{', '.join(result.config.assets[:2])}{'...' if len(result.config.assets) > 2 else ''}",
            f"{result.config.timeframe} | {'+'if result.metrics.net_pnl>=0 else''}${result.metrics.net_pnl:,.0f}",
        )
        
        # Process next asset in queue
        QTimer.singleShot(500, self._process_next_in_queue)

    def _on_failed(self, data: dict) -> None:
        error = data.get("error", "Unknown error")
        self._chat.finish_progress()
        self._chat.add_ai_msg(f"Backtest failed: {error}", icon="❌", sender="Error", color=T.NEGATIVE)
        QTimer.singleShot(500, self._process_next_in_queue)

    def _on_cancelled(self, data: dict) -> None:
        self._chat.finish_progress()
        self._chat.add_ai_msg("Backtest was cancelled.", icon="⚠", sender="System", color=T.WARNING)
        self._backtest_queue.clear()
        self._input_panel.set_running(False)

    def run_demo(self):
        self._on_demo()

def main() -> None:
    """Launch the OpenBacktest.ai desktop application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(GLOBAL_QSS)

    # Load configuration
    app_config = AppConfig.load()
    demo_mode = False

    # Check for API key — show dialog if missing
    if not app_config.api_key:
        dialog = ApiKeyDialog()
        if dialog.exec() != QDialog.Accepted:
            sys.exit(0)
        if dialog.demo_mode:
            # User chose demo: launch without AI features
            demo_mode = True
        else:
            app_config.api_key = dialog.api_key
            app_config.save()

    # Create orchestrator
    orchestrator = OpenBacktestOrchestrator(app_config)

    # Create and show main window
    window = MainWindow(orchestrator)
    window.show()

    # If demo mode: auto-navigate to Chat page and trigger demo
    if demo_mode:
        # Small delay so the window renders first
        QTimer.singleShot(600, window.run_demo)

    _log.info("OpenBacktest.ai started (demo=%s)", demo_mode)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
