import json
import os
import re
import time
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup

warnings.filterwarnings("ignore")

try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except Exception:
    pass

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass
class ScreenerConfig:
    period: str = "9mo"
    interval: str = "1d"
    min_price: float = 50.0
    min_avg_volume_20d: float = 200_000
    min_turnover_idr: float = 1_000_000_000
    min_symbols_required: int = 50
    request_timeout: int = 15
    telegram_bot_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")


class IndonesianStockScreener:
    """
    Daily Indonesian stock screener.

    Key behavior:
    - Stock universe is pulled fresh every run from online sources (no manual static stock list).
    - Ranking is based on daily technical trend + momentum + volume quality.
    - Output report is designed for stock-pick decision making.
    """

    def __init__(self, config: Optional[ScreenerConfig] = None):
        self.config = config or ScreenerConfig()
        self.idx_symbols: List[str] = []
        self.screened_stocks: List[Dict] = []
        self.ssl_error_detected = False

    def _clean_symbol(self, symbol: str) -> Optional[str]:
        if not symbol:
            return None
        s = str(symbol).strip().upper().replace(".JK", "")
        # Regular stock symbols in IDX are generally 4 uppercase letters.
        # Reject numeric/invalid codes (e.g., '12') to avoid broken tickers like 12.JK.
        if re.fullmatch(r"[A-Z]{4}", s):
            return s
        return None

    def fetch_all_idx_stocks_from_idx_website(self) -> List[str]:
        """Fetch symbols from IDX endpoint."""
        symbols: List[str] = []
        try:
            url = "https://www.idx.co.id/umbraco/surface/ListedCompany/GetCompany"
            resp = requests.post(url, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return symbols

            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        cleaned = self._clean_symbol(item.get("Kode", ""))
                        if cleaned:
                            symbols.append(cleaned)
        except Exception as e:
            print(f"IDX source failed: {e}")
        return symbols

    def fetch_all_idx_stocks_from_sectors_app(self) -> List[str]:
        """Fetch symbols from Sectors.app."""
        symbols: List[str] = []
        try:
            url = "https://api.sectors.app/v1/companies/"
            resp = requests.get(url, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return symbols

            data = resp.json()
            items = data.get("data", []) if isinstance(data, dict) else data
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        cleaned = self._clean_symbol(item.get("symbol", ""))
                        if cleaned:
                            symbols.append(cleaned)
        except Exception as e:
            print(f"Sectors.app source failed: {e}")
        return symbols

    def fetch_all_idx_stocks_from_stockanalysis(self) -> List[str]:
        """Fetch symbols by scraping stockanalysis IDX page."""
        symbols: List[str] = []
        try:
            url = "https://stockanalysis.com/list/indonesia-stock-exchange/"
            headers = {
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0 Safari/537.36"
                )
            }
            resp = requests.get(url, headers=headers, timeout=self.config.request_timeout)
            if resp.status_code != 200:
                return symbols

            soup = BeautifulSoup(resp.text, "html.parser")
            table = soup.find("table")
            if not table:
                return symbols

            rows = table.find_all("tr")
            for row in rows[1:]:
                tds = row.find_all("td")
                if not tds:
                    continue
                text = tds[0].get_text(strip=True)
                cleaned = self._clean_symbol(text)
                if cleaned:
                    symbols.append(cleaned)
        except Exception as e:
            print(f"StockAnalysis source failed: {e}")
        return symbols

    def load_all_idx_symbols(self) -> List[str]:
        """Always refresh universe from online sources each run."""
        symbols_set = set()
        sources = [
            self.fetch_all_idx_stocks_from_idx_website,
            self.fetch_all_idx_stocks_from_sectors_app,
            self.fetch_all_idx_stocks_from_stockanalysis,
        ]

        for source in sources:
            try:
                fetched = source()
                if fetched:
                    symbols_set.update(fetched)
                    print(f"{source.__name__}: +{len(fetched)} symbols")
                time.sleep(0.8)
            except Exception as e:
                print(f"{source.__name__} error: {e}")

        if len(symbols_set) < self.config.min_symbols_required:
            raise RuntimeError(
                f"Too few symbols fetched ({len(symbols_set)}). "
                "Check internet/source endpoints before screening."
            )

        self.idx_symbols = sorted([f"{s}.JK" for s in symbols_set])
        print(f"Total symbols loaded: {len(self.idx_symbols)}")
        return self.idx_symbols

    @staticmethod
    def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
        delta = close.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        return (100 - (100 / (1 + rs))).fillna(50)

    @staticmethod
    def _macd(close: pd.Series) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        hist = macd - signal
        return macd, signal, hist

    def get_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        if self.ssl_error_detected:
            return None
        try:
            data = yf.download(
                symbol,
                period=self.config.period,
                interval=self.config.interval,
                progress=False,
                auto_adjust=False,
                threads=False,
            )
            if data is None or data.empty:
                return None
            return data.dropna().copy()
        except Exception as e:
            err = str(e)
            if "CERTIFICATE_VERIFY_FAILED" in err or "SSLCertVerificationError" in err:
                self.ssl_error_detected = True
                print(
                    "Yahoo SSL certificate verification failed. "
                    "Stopping the screener early to avoid repeated errors."
                )
            return None

    def preflight_yahoo_connection(self) -> bool:
        """Run one small Yahoo request first, fail fast if SSL/network is broken."""
        if not self.idx_symbols:
            return False
        probe_symbol = self.idx_symbols[0]
        _ = self.get_stock_data(probe_symbol)
        if self.ssl_error_detected:
            print(
                f"Preflight failed on {probe_symbol}. "
                "Please fix CA certificates/network TLS before running full screening."
            )
            return False
        return True

    def _is_symbol_tradeable_hint(self, symbol: str) -> bool:
        """Skip obvious non-stock/invalid ticker formats before calling Yahoo."""
        base = symbol.replace(".JK", "")
        return bool(re.fullmatch(r"[A-Z]{4}", base))

    def screen_stocks(self, max_symbols: Optional[int] = None) -> List[Dict]:
        if not self.idx_symbols:
            self.load_all_idx_symbols()

        symbols = self.idx_symbols[:max_symbols] if max_symbols else self.idx_symbols
        symbols = [s for s in symbols if self._is_symbol_tradeable_hint(s)]
        total = len(symbols)
        print(f"Running daily screener for {total} symbols...")

        if not self.preflight_yahoo_connection():
            self.screened_stocks = []
            return []
    
        results: List[Dict] = []
        for i, symbol in enumerate(symbols, 1):
            if i % 25 == 0 or i == total:
                print(f"Progress: {i}/{total}")

            if self.ssl_error_detected:
                print("Screening stopped due to SSL verification issue.")
                break

            data = self.get_stock_data(symbol)
            analyzed = self.analyze_stock_daily(symbol, data)
            if analyzed:
                results.append(analyzed)

        results.sort(key=lambda x: (x["total_score"], x["vol_ratio"], x["change_pct"]), reverse=True)
        self.screened_stocks = results
        return results

    def analyze_stock_daily(self, symbol: str, data: pd.DataFrame) -> Optional[Dict]:
        if data is None or len(data) < 80:
            return None

        df = data.copy()
        df["SMA20"] = df["Close"].rolling(20).mean()
        df["SMA50"] = df["Close"].rolling(50).mean()
        df["VOL20"] = df["Volume"].rolling(20).mean()
        df["RSI14"] = self._rsi(df["Close"], 14)
        _, _, macd_hist = self._macd(df["Close"])
        df["MACD_HIST"] = macd_hist
        df["HH20"] = df["High"].rolling(20).max()

        last = df.iloc[-1]
        prev = df.iloc[-2]

        if pd.isna(last["SMA20"]) or pd.isna(last["SMA50"]) or pd.isna(last["VOL20"]):
            return None

        close = float(last["Close"])
        avg_vol = float(last["VOL20"])
        vol = float(last["Volume"])
        vol_ratio = vol / avg_vol if avg_vol > 0 else 0
        turnover = close * vol
        day_change_pct = ((close - float(prev["Close"])) / float(prev["Close"])) * 100 if prev["Close"] else 0.0

        above_sma20 = close > float(last["SMA20"])
        sma20_above_sma50 = float(last["SMA20"]) > float(last["SMA50"])
        rsi_ok = 50 <= float(last["RSI14"]) <= 72
        macd_ok = float(last["MACD_HIST"]) > 0
        breakout_20d = close >= float(last["HH20"]) * 0.995

        trend_score = sum([above_sma20, sma20_above_sma50, rsi_ok, macd_ok, breakout_20d])
        volume_score = 0
        if vol_ratio >= 1.2:
            volume_score += 1
        if vol_ratio >= 1.8:
            volume_score += 1
        if turnover >= self.config.min_turnover_idr:
            volume_score += 1

        total_score = trend_score + volume_score

        if close < self.config.min_price:
            return None
        if avg_vol < self.config.min_avg_volume_20d:
            return None

        action = "MONITOR"
        if total_score >= 7 and vol_ratio >= 1.2 and breakout_20d:
            action = "ADD"
        elif total_score >= 5:
            action = "WATCH"

        notes: List[str] = []
        if breakout_20d:
            notes.append("near_breakout")
        if vol_ratio >= 1.8:
            notes.append("strong_volume")
        if rsi_ok and macd_ok:
            notes.append("momentum_confirmed")
        if day_change_pct > 2:
            notes.append("strong_daily_move")

        return {
            "symbol": symbol,
            "close": close,
            "change_pct": day_change_pct,
            "avg_vol_20d": avg_vol,
            "volume": vol,
            "vol_ratio": vol_ratio,
            "turnover_idr": turnover,
            "rsi14": float(last["RSI14"]),
            "trend_up": bool(above_sma20 and sma20_above_sma50),
            "macd_positive": bool(macd_ok),
            "breakout_20d": bool(breakout_20d),
            "trend_score": trend_score,
            "volume_score": volume_score,
            "total_score": total_score,
            "action": action,
            "notes": ",".join(notes) if notes else "-",
        }

    def build_report_dataframe(self) -> pd.DataFrame:
        if not self.screened_stocks:
            return pd.DataFrame()

        df = pd.DataFrame(self.screened_stocks).copy()
        df["turnover_b"] = df["turnover_idr"] / 1_000_000_000
        report_cols = [
            "symbol",
            "action",
            "total_score",
            "trend_score",
            "volume_score",
            "close",
            "change_pct",
            "vol_ratio",
            "avg_vol_20d",
            "turnover_b",
            "rsi14",
            "trend_up",
            "macd_positive",
            "breakout_20d",
            "notes",
        ]
        return df[report_cols]

    def display_report(self, top_n: int = 30) -> None:
        df = self.build_report_dataframe()
        if df.empty:
            print("No screening results available.")
            return

        top = df.head(top_n).copy()
        top["close"] = top["close"].map(lambda x: f"{x:,.0f}")
        top["change_pct"] = top["change_pct"].map(lambda x: f"{x:+.2f}%")
        top["vol_ratio"] = top["vol_ratio"].map(lambda x: f"{x:.2f}x")
        top["avg_vol_20d"] = top["avg_vol_20d"].map(lambda x: f"{x:,.0f}")
        top["turnover_b"] = top["turnover_b"].map(lambda x: f"{x:.2f}")
        top["rsi14"] = top["rsi14"].map(lambda x: f"{x:.1f}")

        print("\n" + "=" * 120)
        print(f"INDONESIAN DAILY TECHNICAL + VOLUME SCREENER ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
        print("=" * 120)
        print(top.to_string(index=False))

        add_count = int((df["action"] == "ADD").sum())
        watch_count = int((df["action"] == "WATCH").sum())
        monitor_count = int((df["action"] == "MONITOR").sum())
        print("\nDecision Summary:")
        print(f"ADD candidates    : {add_count}")
        print(f"WATCH candidates  : {watch_count}")
        print(f"MONITOR candidates: {monitor_count}")

    def export_report(self, csv_path: str = "screener_report.csv", json_path: str = "screener_report.json") -> None:
        df = self.build_report_dataframe()
        if df.empty:
            print("No data to export.")
            return

        df.to_csv(csv_path, index=False)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(self.screened_stocks, f, indent=2)
        print(f"Report exported: {csv_path}, {json_path}")

    def _telegram_request_with_retry(
        self,
        url: str,
        data: Dict,
        files: Optional[Dict] = None,
        max_retries: int = 3,
    ) -> bool:
        for attempt in range(1, max_retries + 1):
            try:
                response = requests.post(
                    url,
                    data=data,
                    files=files,
                    timeout=self.config.request_timeout,
                )
                if response.status_code == 200:
                    return True
                print(
                    f"Telegram API failed ({attempt}/{max_retries}) "
                    f"status={response.status_code}: {response.text[:160]}"
                )
            except requests.exceptions.RequestException as e:
                print(f"Telegram request error ({attempt}/{max_retries}): {e}")
            time.sleep(1.5 * attempt)
        return False

    def _send_telegram_text(self, text: str) -> bool:
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            print("Telegram disabled: missing TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID in environment.")
            return False

        url = f"https://api.telegram.org/bot{token}/sendMessage"
        # Telegram message limit ~4096 chars.
        chunks: List[str] = []
        if len(text) <= 3900:
            chunks = [text]
        else:
            lines = text.splitlines()
            buffer = ""
            for line in lines:
                candidate = f"{buffer}\n{line}" if buffer else line
                if len(candidate) > 3900:
                    if buffer:
                        chunks.append(buffer)
                    buffer = line
                else:
                    buffer = candidate
            if buffer:
                chunks.append(buffer)

        ok = True
        for chunk in chunks:
            payload = {"chat_id": chat_id, "text": chunk}
            ok = self._telegram_request_with_retry(url, payload) and ok
        return ok

    def _send_telegram_file(self, file_path: str, caption: str = "") -> bool:
        token = self.config.telegram_bot_token
        chat_id = self.config.telegram_chat_id
        if not token or not chat_id:
            return False
        if not os.path.exists(file_path):
            print(f"Telegram file skipped (not found): {file_path}")
            return False

        url = f"https://api.telegram.org/bot{token}/sendDocument"
        payload = {"chat_id": chat_id, "caption": caption}
        with open(file_path, "rb") as f:
            files = {"document": f}
            return self._telegram_request_with_retry(url, payload, files=files)

    def build_telegram_summary(self, top_n: int = 10) -> str:
        df = self.build_report_dataframe()
        if df.empty:
            return "Stock Screener: no candidates found."

        add_df = df[df["action"] == "ADD"].head(top_n)
        watch_df = df[df["action"] == "WATCH"].head(top_n)

        lines = [
            f"IDX Daily Screener ({datetime.now().strftime('%Y-%m-%d %H:%M:%S')})",
            f"Universe screened: {len(df)} symbols",
            f"ADD: {(df['action'] == 'ADD').sum()} | WATCH: {(df['action'] == 'WATCH').sum()} | MONITOR: {(df['action'] == 'MONITOR').sum()}",
            "",
            "Top ADD candidates:",
        ]
        if add_df.empty:
            lines.append("- none")
        else:
            for _, row in add_df.iterrows():
                lines.append(
                    f"- {row['symbol']} | score {int(row['total_score'])} | "
                    f"chg {row['change_pct']:+.2f}% | vol {row['vol_ratio']:.2f}x | RSI {row['rsi14']:.1f}"
                )

        lines.append("")
        lines.append("Top WATCH candidates:")
        if watch_df.empty:
            lines.append("- none")
        else:
            for _, row in watch_df.iterrows():
                lines.append(
                    f"- {row['symbol']} | score {int(row['total_score'])} | "
                    f"chg {row['change_pct']:+.2f}% | vol {row['vol_ratio']:.2f}x | RSI {row['rsi14']:.1f}"
                )

        lines.append("")
        lines.append("Full report attached: CSV + JSON")
        return "\n".join(lines)

    def send_telegram_report(self, csv_path: str = "screener_report.csv", json_path: str = "screener_report.json") -> bool:
        summary = self.build_telegram_summary(top_n=10)
        ok_text = self._send_telegram_text(summary)
        ok_csv = self._send_telegram_file(csv_path, caption="Screener CSV report")
        ok_json = self._send_telegram_file(json_path, caption="Screener JSON report")

        ok = ok_text and (ok_csv or ok_json)
        if ok:
            print("Telegram screener report sent successfully.")
        else:
            print("Telegram screener report failed or partially sent.")
        return ok


if __name__ == "__main__":
    screener = IndonesianStockScreener()

    print("Refreshing IDX symbol universe...")
    screener.load_all_idx_symbols()

    print("\nRunning daily technical + volume screening...")
    screener.screen_stocks()

    screener.display_report(top_n=30)
    screener.export_report()
    screener.send_telegram_report()
