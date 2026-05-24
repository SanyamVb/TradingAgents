"""
KronosWrapper — lazy-loading wrapper around Kronos financial time-series model.

Features:
- Singleton model loading (load once, predict many times)
- yfinance data fetching with NSE ticker normalization
- Multi-sample inference for uncertainty quantification
- Caching of predictions by (ticker, date) to avoid redundant calls
"""

from __future__ import annotations

import sys
import time
import logging
import hashlib
from pathlib import Path
from typing import Optional
from functools import lru_cache

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Path to the cloned Kronos repository
# __file__ = .../TradingAgents/tradingagents/kronos/wrapper.py
# parents[0] = .../tradingagents/kronos
# parents[1] = .../tradingagents
# parents[2] = .../TradingAgents
# parents[3] = .../trading_enhancement
_KRONOS_REPO = Path(__file__).parents[3] / "kronos"  # ~/trading_enhancement/kronos


class KronosWrapper:
    """Lazy-loading wrapper for Kronos financial forecasting model.

    Usage::

        wrapper = KronosWrapper()
        wrapper.load_model()          # 1-time load, ~100s first run
        df_pred = wrapper.predict("RELIANCE.NS", pred_len=5)

    Thread safety: not guaranteed for concurrent predict() calls.
    """

    _instance: Optional[KronosWrapper] = None  # singleton pattern

    def __init__(
        self,
        kronos_repo_path: Optional[str] = None,
        max_context: int = 512,
        sample_count: int = 3,
        temperature: float = 1.0,
        top_p: float = 0.9,
    ):
        self.kronos_repo_path = Path(kronos_repo_path) if kronos_repo_path else _KRONOS_REPO
        self.max_context = max_context
        self.sample_count = sample_count
        self.temperature = temperature
        self.top_p = top_p
        self._model = None
        self._tokenizer = None
        self._predictor = None
        self._load_time: Optional[float] = None
        self._prediction_cache: dict = {}

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def is_loaded(self) -> bool:
        return self._predictor is not None

    def load_model(self) -> None:
        """Load Kronos model weights. No-op if already loaded."""
        if self.is_loaded():
            return

        repo = str(self.kronos_repo_path)
        if repo not in sys.path:
            sys.path.insert(0, repo)

        try:
            from model import Kronos, KronosTokenizer, KronosPredictor
        except ImportError as e:
            raise ImportError(
                f"Cannot import Kronos from {repo}. "
                f"Ensure the repo is cloned at {self.kronos_repo_path}. "
                f"Original error: {e}"
            )

        logger.info("Loading Kronos model (first load ~100s)...")
        t0 = time.time()
        self._tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
        self._model = Kronos.from_pretrained("NeoQuasar/Kronos-base")
        self._predictor = KronosPredictor(self._model, self._tokenizer, max_context=self.max_context)
        self._load_time = time.time() - t0
        logger.info(f"Kronos model loaded in {self._load_time:.1f}s")

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_ticker(ticker: str) -> str:
        """Add .NS suffix for NSE tickers if not already present."""
        if "." not in ticker and ticker.isalpha():
            return ticker + ".NS"
        return ticker

    @staticmethod
    def fetch_ohlcv(ticker: str, lookback_days: int = 300) -> pd.DataFrame:
        """Fetch OHLCV data from yfinance and normalize column names."""
        try:
            import yfinance as yf
        except ImportError:
            raise ImportError("yfinance is required: pip install yfinance")

        ticker_sym = KronosWrapper._normalize_ticker(ticker)
        raw = yf.download(ticker_sym, period=f"{lookback_days}d", progress=False, auto_adjust=True)

        if raw.empty:
            raise ValueError(f"No data returned from yfinance for {ticker_sym}")

        df = raw.reset_index()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0].lower() if c[1] == "" else c[0].lower() for c in df.columns]
        else:
            df.columns = [c.lower() for c in df.columns]

        df = df.rename(columns={"date": "timestamps"})
        df["timestamps"] = pd.to_datetime(df["timestamps"])

        # Ensure 'amount' column (Kronos expects it)
        if "amount" not in df.columns:
            df["amount"] = df["close"] * df["volume"]

        # Drop any rows with NaN close
        df = df.dropna(subset=["close"]).reset_index(drop=True)
        return df

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        ticker: str,
        pred_len: int = 5,
        lookback: int = 200,
        use_cache: bool = True,
        df: Optional[pd.DataFrame] = None,
    ) -> dict:
        """Run Kronos inference for a ticker.

        Args:
            ticker:    Ticker symbol (e.g. "RELIANCE", "RELIANCE.NS", "AAPL")
            pred_len:  Number of future candles to predict (default: 5 trading days)
            lookback:  Historical bars to feed the model (default: 200)
            use_cache: Return cached result if available (default: True)
            df:        Pre-fetched OHLCV DataFrame; if None, fetched from yfinance

        Returns:
            dict with keys: ticker, pred_df, signal, confidence, pred_len,
                            latency_s, samples (list of raw pred DataFrames)
        """
        if not self.is_loaded():
            self.load_model()

        cache_key = f"{ticker}_{pred_len}_{lookback}"
        if use_cache and cache_key in self._prediction_cache:
            logger.debug(f"Kronos cache hit for {ticker}")
            return self._prediction_cache[cache_key]

        if df is None:
            df = self.fetch_ohlcv(ticker, lookback_days=lookback + pred_len + 50)

        # Trim to lookback window
        if len(df) < lookback + pred_len:
            raise ValueError(
                f"Not enough data for {ticker}: got {len(df)} rows, need {lookback + pred_len}"
            )

        x_df = df.iloc[:lookback][["open", "high", "low", "close", "volume", "amount"]]
        x_ts = df.iloc[:lookback]["timestamps"]
        y_ts = df.iloc[lookback:lookback + pred_len]["timestamps"]

        t0 = time.time()
        samples = []
        for _ in range(self.sample_count):
            pred = self._predictor.predict(
                df=x_df,
                x_timestamp=x_ts,
                y_timestamp=y_ts,
                pred_len=pred_len,
                T=self.temperature,
                top_p=self.top_p,
                sample_count=1,
                verbose=False,
            )
            samples.append(pred)

        latency = time.time() - t0

        # Ensemble: median across samples
        close_samples = np.stack([s["close"].values for s in samples])
        median_close = np.median(close_samples, axis=0)
        std_close = np.std(close_samples, axis=0)

        # Build median prediction DataFrame
        pred_df = samples[0].copy()
        pred_df["close"] = median_close
        pred_df["close_std"] = std_close

        last_close = float(df.iloc[lookback - 1]["close"])
        final_close = float(median_close[-1])
        pct_change = (final_close - last_close) / last_close * 100

        # Uncertainty: coefficient of variation of std over mean
        avg_cv = float(np.mean(std_close / (np.abs(median_close) + 1e-9)))

        signal, confidence = self._derive_signal(pct_change, avg_cv)

        result = {
            "ticker": ticker,
            "pred_df": pred_df,
            "signal": signal,            # "BUY", "SELL", "HOLD", "SKIP"
            "confidence": confidence,    # 0.0 – 1.0
            "pct_change_5d": round(pct_change, 2),
            "last_close": round(last_close, 2),
            "pred_close_5d": [round(float(v), 2) for v in median_close],
            "pred_std_5d": [round(float(v), 2) for v in std_close],
            "latency_s": round(latency, 3),
            "samples": samples,
        }

        if use_cache:
            self._prediction_cache[cache_key] = result

        return result

    # ------------------------------------------------------------------
    # Signal derivation
    # ------------------------------------------------------------------

    @staticmethod
    def _derive_signal(pct_change: float, uncertainty: float) -> tuple[str, float]:
        """Derive trading signal from predicted price change and uncertainty.

        Returns:
            (signal, confidence) where signal is BUY/SELL/HOLD/SKIP
        """
        # High uncertainty → skip
        if uncertainty > 0.15:
            return "SKIP", round(max(0.0, 1.0 - uncertainty * 4), 2)

        if pct_change > 2.0:
            signal = "BUY"
        elif pct_change < -2.0:
            signal = "SELL"
        else:
            signal = "HOLD"

        # Confidence: higher change and lower uncertainty = higher confidence
        raw_conf = min(abs(pct_change) / 5.0, 1.0) * (1.0 - uncertainty * 3)
        confidence = round(max(0.05, min(1.0, raw_conf)), 2)
        return signal, confidence

    # ------------------------------------------------------------------
    # Singleton / class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def get_instance(cls, **kwargs) -> KronosWrapper:
        """Return a shared KronosWrapper instance (singleton)."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance
