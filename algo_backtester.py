#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
algo_backtester.py
==================
Strategy-agnostic backtesting engine for intraday or multi-day OHLCV data.

Design goals
------------
- No assumptions about asset class, instrument, session times, or strategy type.
- Pure Python (pandas + numpy only) — no external dependencies.
- UTC-aware DatetimeIndex throughout.
- Plug-in strategy interface: subclass Strategy and implement three hooks.
- Generates a trade log DataFrame and key performance metrics.
- All instrument / cost parameters are caller-supplied (no hard-coded defaults).

Typical usage
-------------
    from algo_backtester import load_csv, Broker, Backtester, equity_and_stats, Strategy

    class MyStrategy(Strategy):
        def on_bar(self, ctx):
            ...  # return "long", "short", or None

        def on_exit_prices(self, ctx, direction, entry_price):
            ...  # return (stop_price, target_price)

    df     = load_csv("data.csv")
    broker = Broker(usd_per_point=5.0, commission_rt=2.50)
    engine = Backtester(broker)
    trades = engine.run(df, MyStrategy())
    stats  = equity_and_stats(trades)

CSV format (expected, all lowercase)
--------------------------------------
    datetime,open,high,low,close,volume
    20240101 09:30,18000.0,18010.0,17995.0,18005.0,3421
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def load_csv(
    path: str,
    dt_col: str = "datetime",
    fmt: Optional[str] = "%Y%m%d %H%M",
    tz_utc: bool = True,
) -> pd.DataFrame:
    """
    Load an OHLCV CSV into a UTC-indexed DataFrame.

    Parameters
    ----------
    path    : path to the CSV file
    dt_col  : name of the datetime column
    fmt     : strptime format string; pass None to let pandas auto-detect
    tz_utc  : if True, attach UTC timezone to the index

    Returns
    -------
    DataFrame with columns [open, high, low, close, volume] and a UTC DatetimeIndex.
    """
    df = pd.read_csv(path)
    print(f"Loaded CSV with columns: {list(df.columns)}")

    if dt_col not in df.columns:
        raise ValueError(
            f"Column '{dt_col}' not found. Available columns: {list(df.columns)}"
        )

    if fmt is None:
        dt = pd.to_datetime(df[dt_col], utc=tz_utc)
    else:
        try:
            dt = pd.to_datetime(df[dt_col], format=fmt, utc=tz_utc)
        except ValueError:
            print(f"Warning: format '{fmt}' failed — falling back to pandas auto-detection.")
            dt = pd.to_datetime(df[dt_col], utc=tz_utc)

    df = df.drop(columns=[dt_col]).rename(columns=str.lower)
    df["datetime"] = dt
    df = df.set_index("datetime").sort_index()

    required = ["open", "high", "low", "close", "volume"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found: {list(df.columns)}")

    for col in required:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=required)

    print(f"Loaded {len(df):,} rows | {df.index.min()} → {df.index.max()}")
    return df[required]


def between_time_utc(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Slice a UTC-indexed DataFrame to a daily time window (e.g. '09:30', '16:00')."""
    return df.between_time(start, end)


def resample_ohlcv(df: pd.DataFrame, rule: str = "1min") -> pd.DataFrame:
    """Resample OHLCV data to a coarser timeframe."""
    return (
        df.resample(rule)
          .agg({"open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"})
          .dropna()
    )


# ---------------------------------------------------------------------------
# Broker / fills
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    entry_time:  pd.Timestamp
    entry_price: float
    exit_time:   pd.Timestamp
    exit_price:  float
    qty:         int        # positive = long contract(s), negative = short
    direction:   str        # "long" | "short"
    exit_reason: str        # "TP" | "SL" | "EOD" | "OTHER"
    gross:       float      # USD before commission
    net:         float      # USD after commission


class Broker:
    """
    Simulates fills at bar prices with optional slippage and commission.

    Parameters
    ----------
    usd_per_point  : dollar value per point of price movement per contract
    commission_rt  : round-trip commission in USD per contract
    slippage_points: symmetric slippage applied to every fill side (in price points)
    """

    def __init__(
        self,
        usd_per_point: float,
        commission_rt: float,
        slippage_points: float = 0.0,
    ):
        self.usd_per_point   = float(usd_per_point)
        self.commission_rt   = float(commission_rt)
        self.slippage_points = float(slippage_points)

    def apply_slippage(self, price: float, is_buy: bool) -> float:
        """Worsen the fill price by the configured slippage."""
        if self.slippage_points == 0.0:
            return price
        return price + self.slippage_points if is_buy else price - self.slippage_points

    def pnl_dollars(
        self, entry: float, exit_p: float, qty: int
    ) -> Tuple[float, float]:
        """
        Return (gross_pnl, net_pnl) in USD.

        qty > 0 means long; qty < 0 means short.
        """
        direction_sign = 1 if qty > 0 else -1
        points = (exit_p - entry) * direction_sign
        gross  = points * abs(qty) * self.usd_per_point
        net    = gross - self.commission_rt * abs(qty)
        return gross, net


# ---------------------------------------------------------------------------
# Strategy API
# ---------------------------------------------------------------------------

class BarContext:
    """
    Passed to the strategy on every bar.

    Attributes
    ----------
    df         : the full working DataFrame (session slice or entire dataset)
    i          : integer index of the current bar within df
    bar        : the current bar as a Series (open, high, low, close, volume)
    time       : Timestamp of the current bar
    extras     : dict for engine → strategy communication (e.g. grouping info)
    """

    def __init__(
        self,
        df: pd.DataFrame,
        i: int,
        extras: Optional[Dict[str, Any]] = None,
    ):
        self.df     = df
        self.i      = i
        self.bar    = df.iloc[i]
        self.time   = df.index[i]
        self.extras = extras or {}


class Strategy:
    """
    Base class for all user strategies.

    Override
    --------
    on_session_start  : called once at the start of each day/group (optional)
    on_bar            : called on every bar; return "long", "short", or None
    on_exit_prices    : called once when a position is opened; must return (stop, target)
    allow_multiple_trades : return True to allow more than one trade per day/group
    """

    def on_session_start(self, session_df: pd.DataFrame) -> None:
        """Hook called at the beginning of each daily session (or group)."""
        pass

    def on_bar(self, ctx: BarContext) -> Optional[str]:
        """
        Signal hook called for every bar.

        Return "long" or "short" to open a trade, or None to do nothing.
        """
        return None

    def on_exit_prices(
        self, ctx: BarContext, direction: str, entry_price: float
    ) -> Tuple[float, float]:
        """
        Called once when entering a trade.

        Must return (stop_price, target_price).
        """
        raise NotImplementedError("Implement on_exit_prices in your Strategy subclass.")

    def allow_multiple_trades(self) -> bool:
        """
        Return True to allow more than one trade per day.
        Default is False (one trade per calendar day).
        """
        return False


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class Backtester:
    """
    Generic bar-by-bar backtesting engine.

    The engine iterates through all bars (optionally grouped by calendar day),
    calls the strategy hooks, and simulates bar-level exits using H/L touches.

    Parameters
    ----------
    broker           : Broker instance (cost model)
    group_by_day     : if True (default), the strategy's on_session_start hook
                       is called once per calendar day and one-trade-per-day
                       logic is applied based on strategy.allow_multiple_trades()
    session_start_utc: if set, only bars at or after this time (HH:MM) are used
    session_end_utc  : if set, only bars before or at this time (HH:MM) are used
    """

    def __init__(
        self,
        broker: Broker,
        group_by_day: bool = True,
        session_start_utc: Optional[str] = None,
        session_end_utc: Optional[str] = None,
    ):
        self.broker            = broker
        self.group_by_day      = group_by_day
        self.session_start_utc = session_start_utc
        self.session_end_utc   = session_end_utc

    def run(
        self,
        df: pd.DataFrame,
        strategy: Strategy,
        resample_rule: str = "1min",
    ) -> pd.DataFrame:
        """
        Run the strategy over the dataset.

        Parameters
        ----------
        df             : UTC-indexed OHLCV DataFrame (from load_csv)
        strategy       : Strategy instance
        resample_rule  : pandas offset alias; data is resampled before iteration

        Returns
        -------
        DataFrame of Trade records.
        """
        data = resample_ohlcv(df, resample_rule)

        # Optional session filter
        if self.session_start_utc and self.session_end_utc:
            data = between_time_utc(data, self.session_start_utc, self.session_end_utc)

        records: List[Trade] = []

        if self.group_by_day:
            groups = data.groupby(data.index.date)
        else:
            # Treat the entire dataset as a single group
            groups = [(None, data)]

        for day, session in groups:
            if session.empty:
                continue

            strategy.on_session_start(session)

            trades_today = 0
            i = 0
            while i < len(session):
                ctx = BarContext(session, i, extras={"day": day})
                sig = strategy.on_bar(ctx)

                if sig in ("long", "short"):
                    entry_price = float(ctx.bar["close"])
                    is_buy      = sig == "long"
                    entry_price = self.broker.apply_slippage(entry_price, is_buy)
                    stop, target = strategy.on_exit_prices(ctx, sig, entry_price)

                    exit_price, exit_time, exit_reason = self._walk_exit(
                        session, i + 1, sig, stop, target
                    )

                    qty = 1 if sig == "long" else -1
                    gross, net = self.broker.pnl_dollars(entry_price, exit_price, qty)

                    records.append(Trade(
                        entry_time=ctx.time,
                        entry_price=round(entry_price, 6),
                        exit_time=exit_time,
                        exit_price=round(exit_price, 6),
                        qty=qty,
                        direction=sig,
                        exit_reason=exit_reason,
                        gross=round(gross, 2),
                        net=round(net, 2),
                    ))

                    trades_today += 1
                    if not strategy.allow_multiple_trades() and trades_today >= 1:
                        break

                    # Advance past the exit bar
                    exit_loc = session.index.get_indexer([exit_time])[0]
                    i = exit_loc + 1
                    continue

                i += 1

        return pd.DataFrame([asdict(t) for t in records])

    def _walk_exit(
        self,
        session: pd.DataFrame,
        start_idx: int,
        direction: str,
        stop: float,
        target: float,
    ) -> Tuple[float, pd.Timestamp, str]:
        """
        Walk bars forward from start_idx looking for a stop or target touch.

        If both stop and target are hit within the same bar, stop is assumed
        to fill first (conservative assumption).

        Falls back to EOD exit at the last bar's close if neither level is hit.
        """
        for j in range(start_idx, len(session)):
            bar  = session.iloc[j]
            high = float(bar["high"])
            low  = float(bar["low"])
            ts   = session.index[j]

            if direction == "long":
                if low  <= stop:   return stop,   ts, "SL"
                if high >= target: return target,  ts, "TP"
            else:  # short
                if high >= stop:   return stop,   ts, "SL"
                if low  <= target: return target,  ts, "TP"

        last = session.iloc[-1]
        return float(last["close"]), session.index[-1], "EOD"


# ---------------------------------------------------------------------------
# Performance & reporting
# ---------------------------------------------------------------------------

def equity_and_stats(trades: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute an equity curve and comprehensive performance statistics.

    Returns a dict with keys:
        trades, winrate_pct, profit_factor, net_profit, gross_profit, gross_loss,
        max_drawdown, max_drawdown_pct, avg_trade, avg_win, avg_loss,
        largest_win, largest_loss, win_loss_ratio, expectancy,
        sharpe_ratio, sortino_ratio, calmar_ratio,
        consec_wins_max, consec_losses_max,
        long_trades, short_trades, tp_count, sl_count, eod_count,
        recovery_factor
    """
    if trades.empty:
        return {
            "trades": 0, "winrate_pct": 0.0, "profit_factor": 0.0,
            "net_profit": 0.0, "gross_profit": 0.0, "gross_loss": 0.0,
            "max_drawdown": 0.0, "max_drawdown_pct": 0.0,
            "avg_trade": 0.0, "avg_win": 0.0, "avg_loss": 0.0,
            "largest_win": 0.0, "largest_loss": 0.0,
            "win_loss_ratio": 0.0, "expectancy": 0.0,
            "sharpe_ratio": 0.0, "sortino_ratio": 0.0, "calmar_ratio": 0.0,
            "consec_wins_max": 0, "consec_losses_max": 0,
            "long_trades": 0, "short_trades": 0,
            "tp_count": 0, "sl_count": 0, "eod_count": 0,
            "recovery_factor": 0.0,
        }

    t = trades.copy()
    net = t["net"]
    t["cum_equity"] = net.cumsum()

    # ── Basic counts ──
    n         = len(t)
    wins_mask = net > 0
    loss_mask = net <= 0
    wins      = int(wins_mask.sum())
    losses    = int(loss_mask.sum())
    winrate   = wins / max(1, n) * 100.0

    # ── PnL aggregates ──
    gross_profit = float(net[wins_mask].sum())
    gross_loss   = float(-net[loss_mask].sum())
    net_profit   = float(net.sum())
    pf           = (gross_profit / gross_loss) if gross_loss > 0 else math.inf
    avg_trade    = float(net.mean())
    avg_win      = float(net[wins_mask].mean()) if wins  > 0 else 0.0
    avg_loss     = float(net[loss_mask].mean()) if losses > 0 else 0.0
    largest_win  = float(net.max())
    largest_loss = float(net.min())
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else math.inf

    # ── Expectancy: avg $ gained per trade weighted by probability ──
    # E = (WR * avg_win) + ((1-WR) * avg_loss)   [avg_loss is negative]
    wr_dec    = wins / max(1, n)
    expectancy = (wr_dec * avg_win) + ((1 - wr_dec) * avg_loss)

    # ── Drawdown ──
    cum_eq      = t["cum_equity"]
    rolling_max = cum_eq.cummax()
    dd_series   = rolling_max - cum_eq
    max_dd      = float(dd_series.max())
    peak        = float(rolling_max.max())
    max_dd_pct  = (max_dd / peak * 100.0) if peak > 0 else 0.0

    # ── Sharpe ratio (per-trade, risk-free = 0) ──
    std_trade = float(net.std(ddof=1)) if n > 1 else 0.0
    sharpe    = (avg_trade / std_trade * math.sqrt(n)) if std_trade > 0 else 0.0

    # ── Sortino ratio (penalises only downside deviation) ──
    downside   = net[net < 0]
    down_std   = float(downside.std(ddof=1)) if len(downside) > 1 else 0.0
    sortino    = (avg_trade / down_std * math.sqrt(n)) if down_std > 0 else 0.0

    # ── Calmar ratio (net profit / max drawdown) ──
    calmar = (net_profit / max_dd) if max_dd > 0 else math.inf

    # ── Recovery factor (net profit / max drawdown) — alias for calmar here ──
    recovery_factor = calmar

    # ── Consecutive wins / losses ──
    streak_wins = streak_losses = cur_w = cur_l = 0
    for v in net:
        if v > 0:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        streak_wins   = max(streak_wins,   cur_w)
        streak_losses = max(streak_losses, cur_l)

    # ── Direction & exit reason breakdown ──
    long_trades  = int((t["direction"] == "long").sum())
    short_trades = int((t["direction"] == "short").sum())
    tp_count     = int((t["exit_reason"] == "TP").sum())
    sl_count     = int((t["exit_reason"] == "SL").sum())
    eod_count    = int((t["exit_reason"] == "EOD").sum())

    def _r(v):
        return round(float(v), 2) if v not in (math.inf, -math.inf) else v

    return {
        "trades":           n,
        "long_trades":      long_trades,
        "short_trades":     short_trades,
        "winrate_pct":      _r(winrate),
        "profit_factor":    _r(pf),
        "net_profit":       _r(net_profit),
        "gross_profit":     _r(gross_profit),
        "gross_loss":       _r(gross_loss),
        "max_drawdown":     _r(max_dd),
        "max_drawdown_pct": _r(max_dd_pct),
        "avg_trade":        _r(avg_trade),
        "avg_win":          _r(avg_win),
        "avg_loss":         _r(avg_loss),
        "largest_win":      _r(largest_win),
        "largest_loss":     _r(largest_loss),
        "win_loss_ratio":   _r(win_loss_ratio),
        "expectancy":       _r(expectancy),
        "sharpe_ratio":     _r(sharpe),
        "sortino_ratio":    _r(sortino),
        "calmar_ratio":     _r(calmar),
        "recovery_factor":  _r(recovery_factor),
        "consec_wins_max":  streak_wins,
        "consec_losses_max": streak_losses,
        "tp_count":         tp_count,
        "sl_count":         sl_count,
        "eod_count":        eod_count,
    }


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

def run_backtest(
    csv_path: str,
    strategy: Strategy,
    usd_per_point: float,
    commission_rt: float,
    slippage_points: float = 0.0,
    session_start_utc: Optional[str] = None,
    session_end_utc: Optional[str] = None,
    group_by_day: bool = True,
    resample_rule: str = "1min",
    dt_col: str = "datetime",
    dt_fmt: Optional[str] = "%Y%m%d %H%M",
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    One-call convenience wrapper: load CSV → run → return (trades_df, stats_dict).

    All instrument and cost parameters must be supplied explicitly.
    """
    df     = load_csv(csv_path, dt_col=dt_col, fmt=dt_fmt, tz_utc=True)
    broker = Broker(
        usd_per_point=usd_per_point,
        commission_rt=commission_rt,
        slippage_points=slippage_points,
    )
    engine = Backtester(
        broker=broker,
        group_by_day=group_by_day,
        session_start_utc=session_start_utc,
        session_end_utc=session_end_utc,
    )
    trades = engine.run(df, strategy=strategy, resample_rule=resample_rule)
    stats  = equity_and_stats(trades)
    return trades, stats


# ---------------------------------------------------------------------------
# Kept for backwards-compatibility (deprecated — use BarContext instead)
# ---------------------------------------------------------------------------

# StrategyContext is retained so existing code referencing it still imports,
# but new strategies should use BarContext.
StrategyContext = BarContext
