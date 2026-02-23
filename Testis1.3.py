#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Testis.py  —  Streamlit front-end for algo_backtester.py
=========================================================
Run with:
    streamlit run Testis.py

Tabs
----
  Backtester  : Load CSV + strategy script, configure instrument params, run & visualise.
  Real-Time   : Live candlestick chart + L2 order book via NinjaTrader UDP.

Requirements
------------
    pip install streamlit plotly pandas numpy
"""

import sys
import io
import json
import math
import socket
import threading
import time
import traceback
import importlib.util
import tempfile
import os
from collections import deque
from datetime import datetime, timezone

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── page config (must be FIRST st call) ───────────────────────────────────────
st.set_page_config(
    page_title="TESTIS — Algo Backtester",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Barlow:wght@300;400;600;700&family=Barlow+Condensed:wght@400;600;700&display=swap');

  html, body, [class*="css"] {
    font-family: 'Barlow', sans-serif;
    background-color: #080c10;
    color: #c8d6e0;
  }

  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 1rem 2rem 2rem 2rem; max-width: 100%; }

  .app-header {
    display: flex; align-items: center; gap: 1.2rem;
    padding: 0.6rem 0 1.2rem 0;
    border-bottom: 1px solid #1e3040;
    margin-bottom: 1.4rem;
  }
  .logo {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 2rem; font-weight: 700; letter-spacing: 0.15em;
    color: #00d4ff; text-shadow: 0 0 20px rgba(0,212,255,0.4);
  }
  .subtitle {
    font-size: 0.78rem; font-weight: 300; color: #4a7090;
    letter-spacing: 0.12em; text-transform: uppercase; margin-top: 2px;
  }
  .badge {
    margin-left: auto;
    font-family: 'Share Tech Mono', monospace; font-size: 0.7rem;
    color: #2a5a3a; background: #0a1a0f;
    border: 1px solid #1a4a2a; padding: 3px 10px; border-radius: 3px;
  }

  .stTabs [data-baseweb="tab-list"] {
    gap: 0; border-bottom: 1px solid #1a2e40; background: transparent;
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.85rem; letter-spacing: 0.1em; font-weight: 600;
    text-transform: uppercase; padding: 0.5rem 1.6rem;
    color: #3a6070; border-bottom: 2px solid transparent;
    background: transparent; transition: all 0.15s;
  }
  .stTabs [aria-selected="true"] {
    color: #00d4ff !important;
    border-bottom: 2px solid #00d4ff !important;
    background: transparent !important;
  }
  .stTabs [data-baseweb="tab-panel"] { padding-top: 1.2rem; }

  .panel-title {
    font-family: 'Barlow Condensed', sans-serif;
    font-size: 0.7rem; font-weight: 600; letter-spacing: 0.18em;
    text-transform: uppercase; color: #2a6080;
    margin-bottom: 0.8rem; border-bottom: 1px solid #1a2e40;
    padding-bottom: 0.4rem;
  }

  .stat-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.5rem; }
  .stat-grid-wide { display: grid; grid-template-columns: repeat(4,1fr); gap: 0.5rem; margin-top: 0.5rem; }
  .stat-section-title { font-family: "Barlow Condensed", sans-serif; font-size: 0.62rem; font-weight: 600; letter-spacing: 0.16em; text-transform: uppercase; color: #1a5060; margin: 0.9rem 0 0.4rem 0; }
  .stat-card {
    background: #0a1520; border: 1px solid #1a2e40;
    border-radius: 4px; padding: 0.7rem 0.9rem;
  }
  .stat-label {
    font-size: 0.6rem; letter-spacing: 0.14em; text-transform: uppercase;
    color: #3a6070; margin-bottom: 0.2rem;
  }
  .stat-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 1.1rem; color: #c8d6e0;
  }
  .stat-value.positive { color: #00c87a; }
  .stat-value.negative { color: #e05050; }

  .terminal-wrap {
    background: #050a0e; border: 1px solid #0f2030;
    border-radius: 4px; overflow: hidden;
  }
  .terminal-titlebar {
    background: #0a1520; border-bottom: 1px solid #0f2030;
    padding: 5px 12px; display: flex; align-items: center; gap: 6px;
  }
  .terminal-dot { width:8px; height:8px; border-radius:50%; display:inline-block; }
  .dot-red   { background:#e05050; }
  .dot-amber { background:#d4a017; }
  .dot-green { background:#00c87a; }
  .terminal-title {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.65rem; color: #2a5060; margin-left: 6px; letter-spacing: 0.1em;
  }
  .terminal-body {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.72rem; line-height: 1.6; color: #4aaa6a;
    background: #050a0e; padding: 0.8rem 1rem;
    min-height: 180px; max-height: 320px;
    overflow-y: auto; white-space: pre-wrap; word-break: break-all;
  }
  .t-info  { color: #4aaa6a; }
  .t-warn  { color: #d4a017; }
  .t-error { color: #e05050; }
  .t-data  { color: #00d4ff; }
  .t-dim   { color: #2a5060; }

  .stNumberInput input, .stTextInput input {
    background: #0a1520 !important; border: 1px solid #1a3040 !important;
    color: #c8d6e0 !important;
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 0.85rem !important; border-radius: 3px !important;
  }
  .stSelectbox > div > div {
    background: #0a1520 !important; border: 1px solid #1a3040 !important;
    color: #c8d6e0 !important;
  }
  label {
    font-size: 0.68rem !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; color: #3a6070 !important;
    font-weight: 600 !important;
  }

  .stButton > button {
    font-family: 'Barlow Condensed', sans-serif !important;
    font-weight: 600 !important; letter-spacing: 0.1em !important;
    text-transform: uppercase !important; font-size: 0.82rem !important;
    border-radius: 3px !important; transition: all 0.15s !important;
  }
  .stButton > button[kind="primary"] {
    background: #003a52 !important; border: 1px solid #00d4ff !important;
    color: #00d4ff !important;
  }
  .stButton > button[kind="primary"]:hover {
    background: #00d4ff !important; color: #080c10 !important;
    box-shadow: 0 0 16px rgba(0,212,255,0.3) !important;
  }
  .stButton > button[kind="secondary"] {
    background: transparent !important; border: 1px solid #1a3040 !important;
    color: #4a7090 !important;
  }
  .stButton > button[kind="secondary"]:hover {
    border-color: #3a6080 !important; color: #c8d6e0 !important;
  }

  .stFileUploader section {
    background: #0a1520 !important; border: 1px dashed #1a3040 !important;
    border-radius: 4px !important;
  }

  hr { border-color: #1a2e40 !important; margin: 0.8rem 0 !important; }

  ::-webkit-scrollbar { width: 4px; height: 4px; }
  ::-webkit-scrollbar-track { background: #080c10; }
  ::-webkit-scrollbar-thumb { background: #1a3040; border-radius: 2px; }
  ::-webkit-scrollbar-thumb:hover { background: #2a5060; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Session state
# ══════════════════════════════════════════════════════════════════════════════

def _init_state():
    defaults = {
        "log":           [],
        "trades_df":     None,
        "stats":         None,
        "data_df":       None,
        "data_name":     None,
        "strategy_src":  None,
        "strategy_name": None,
        "rt_ticks":      deque(maxlen=30000),
        "rt_bids":       {},
        "rt_asks":       {},
        "rt_connected":  False,
        "rt_tick_count": 0,
        "rt_l2_count":   0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ══════════════════════════════════════════════════════════════════════════════
# Terminal logger
# ══════════════════════════════════════════════════════════════════════════════

def log(msg: str, level: str = "info"):
    ts = datetime.now(timezone.utc).strftime("%H:%M:%S.%f")[:-3]
    st.session_state["log"].append((level, f"[{ts}]  {msg}"))
    if len(st.session_state["log"]) > 500:
        st.session_state["log"] = st.session_state["log"][-500:]


class TerminalCapture(io.StringIO):
    def __init__(self, level="info"):
        super().__init__()
        self._level = level

    def write(self, s):
        s = s.strip()
        if s:
            log(s, self._level)

    def flush(self):
        pass


def render_terminal():
    lines_html = []
    for level, msg in st.session_state["log"]:
        cls  = {"info":"t-info","warn":"t-warn","error":"t-error",
                "data":"t-data","dim":"t-dim"}.get(level, "t-info")
        safe = msg.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")
        lines_html.append(f'<span class="{cls}">{safe}</span>')

    body = "\n".join(lines_html) if lines_html else \
        '<span class="t-dim">  // terminal ready — awaiting operations</span>'

    st.markdown(f"""
    <div class="terminal-wrap">
      <div class="terminal-titlebar">
        <span class="terminal-dot dot-red"></span>
        <span class="terminal-dot dot-amber"></span>
        <span class="terminal-dot dot-green"></span>
        <span class="terminal-title">TESTIS CONSOLE  //  UTC</span>
      </div>
      <div class="terminal-body">{body}</div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Import engine
# ══════════════════════════════════════════════════════════════════════════════

try:
    from algo_backtester import (
        Backtester, Broker, BarContext, load_csv,
        equity_and_stats, resample_ohlcv, Strategy,
    )
    log("algo_backtester imported OK", "info")
except ImportError as e:
    st.error(f"Cannot import algo_backtester: {e}")
    st.stop()


# ══════════════════════════════════════════════════════════════════════════════
# Plotly chart builders
# ══════════════════════════════════════════════════════════════════════════════

_LAYOUT_BASE = dict(
    paper_bgcolor="#080c10",
    plot_bgcolor="#0c1520",
    font=dict(family="Share Tech Mono", color="#6a9ab0", size=11),
    xaxis=dict(gridcolor="#0f2030", linecolor="#1a3040", tickcolor="#1a3040",
               showgrid=True, zeroline=False),
    yaxis=dict(gridcolor="#0f2030", linecolor="#1a3040", tickcolor="#1a3040",
               showgrid=True, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#1a3040", borderwidth=1),
    margin=dict(l=50, r=20, t=36, b=40),
)


def _base_fig():
    fig = go.Figure()
    fig.update_layout(**_LAYOUT_BASE)
    return fig


def build_candle_chart(df, trades_df=None, show_trades=False, title="Price"):
    fig = _base_fig()
    idx = df.index
    if hasattr(idx, "tz") and idx.tz is not None:
        idx = idx.tz_localize(None)

    fig.add_trace(go.Candlestick(
        x=idx, open=df["open"], high=df["high"], low=df["low"], close=df["close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00c87a", width=1), fillcolor="#00c87a"),
        decreasing=dict(line=dict(color="#e05050", width=1), fillcolor="#e05050"),
    ))

    if show_trades and trades_df is not None and not trades_df.empty:
        entry_t = pd.to_datetime(trades_df["entry_time"], utc=True, errors="coerce").dt.tz_localize(None)
        exit_t  = pd.to_datetime(trades_df["exit_time"],  utc=True, errors="coerce").dt.tz_localize(None)
        longs   = trades_df["direction"] == "long"
        shorts  = trades_df["direction"] == "short"
        tps     = trades_df["exit_reason"] == "TP"
        sls     = trades_df["exit_reason"] == "SL"

        if longs.any():
            fig.add_trace(go.Scatter(x=entry_t[longs], y=trades_df.loc[longs,"entry_price"],
                mode="markers", name="Long Entry",
                marker=dict(symbol="triangle-up", size=11, color="#00d4ff",
                            line=dict(width=1,color="#007a9a"))))
        if shorts.any():
            fig.add_trace(go.Scatter(x=entry_t[shorts], y=trades_df.loc[shorts,"entry_price"],
                mode="markers", name="Short Entry",
                marker=dict(symbol="triangle-down", size=11, color="#ff7a3a",
                            line=dict(width=1,color="#aa3a00"))))
        if tps.any():
            fig.add_trace(go.Scatter(x=exit_t[tps], y=trades_df.loc[tps,"exit_price"],
                mode="markers", name="TP Exit",
                marker=dict(symbol="circle", size=8, color="#00c87a",
                            line=dict(width=1,color="#007a40"))))
        if sls.any():
            fig.add_trace(go.Scatter(x=exit_t[sls], y=trades_df.loc[sls,"exit_price"],
                mode="markers", name="SL Exit",
                marker=dict(symbol="x", size=9, color="#e05050",
                            line=dict(width=2,color="#a02020"))))

    fig.update_layout(xaxis_rangeslider_visible=False, height=460,
                      title=dict(text=title, font=dict(size=11, color="#2a6080")))
    return fig


def build_equity_chart(trades_df):
    fig = _base_fig()
    if trades_df is None or trades_df.empty:
        return fig
    eq = trades_df["net"].cumsum().reset_index(drop=True)
    fig.add_trace(go.Scatter(
        x=eq.index, y=eq.values, mode="lines",
        line=dict(color="#00d4ff", width=1.5),
        fill="tozeroy", fillcolor="rgba(0,212,255,0.06)",
        name="Equity",
    ))
    fig.add_hline(y=0, line=dict(color="#1a3040", width=1, dash="dot"))
    fig.update_layout(height=220, showlegend=False,
                      title=dict(text="EQUITY CURVE", font=dict(size=10,color="#2a6080")),
                      yaxis=dict(tickprefix="$"))
    return fig


def build_pnl_dist(trades_df):
    fig = _base_fig()
    if trades_df is None or trades_df.empty:
        return fig
    net    = trades_df["net"]
    wins   = net[net > 0]
    losses = net[net <= 0]
    fig.add_trace(go.Histogram(x=wins,   name="Win",  marker_color="#00c87a", opacity=0.8, nbinsx=20))
    fig.add_trace(go.Histogram(x=losses, name="Loss", marker_color="#e05050", opacity=0.8, nbinsx=20))
    fig.update_layout(barmode="overlay", height=220,
                      title=dict(text="PnL DISTRIBUTION", font=dict(size=10,color="#2a6080")),
                      xaxis=dict(tickprefix="$"))
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# Stats cards
# ══════════════════════════════════════════════════════════════════════════════

def _pf_str(v):
    return f"{v:.2f}" if v not in (math.inf, -math.inf) else "∞"

def _card(lbl, val, cls=""):
    return (f'<div class="stat-card">'
            f'<div class="stat-label">{lbl}</div>'
            f'<div class="stat-value {cls}">{val}</div>'
            f'</div>')

def _section(title, cards):
    inner = "".join(_card(l, v, c) for l, v, c in cards)
    return (f'<div class="stat-section-title">{title}</div>'
            f'<div class="stat-grid">{inner}</div>')

def render_stats(stats: dict):
    if not stats:
        return

    def _g(k, d=0):
        return stats.get(k, d)

    netp  = _g("net_profit")
    wr    = _g("winrate_pct")
    pf    = _g("profit_factor")
    dd    = _g("max_drawdown")
    ddpct = _g("max_drawdown_pct")
    avg   = _g("avg_trade")
    n     = _g("trades")
    longs = _g("long_trades")
    shts  = _g("short_trades")
    gp    = _g("gross_profit")
    gl    = _g("gross_loss")
    aw    = _g("avg_win")
    al    = _g("avg_loss")
    lw    = _g("largest_win")
    ll    = _g("largest_loss")
    wlr   = _g("win_loss_ratio")
    exp   = _g("expectancy")
    sh    = _g("sharpe_ratio")
    so    = _g("sortino_ratio")
    cal   = _g("calmar_ratio")
    rf    = _g("recovery_factor")
    cw    = _g("consec_wins_max")
    cl    = _g("consec_losses_max")
    tp    = _g("tp_count")
    sl    = _g("sl_count")
    eod   = _g("eod_count")

    nc  = "positive" if netp >= 0 else "negative"
    ac  = "positive" if avg  >= 0 else "negative"
    ec  = "positive" if exp  >= 0 else "negative"
    wc  = "positive" if wr   >= 50 else "negative"
    shc = "positive" if sh   >= 1  else ("negative" if sh < 0 else "")
    soc = "positive" if so   >= 1  else ("negative" if so < 0 else "")

    html = ""

    # ── Overview ──
    html += _section("Overview", [
        ("Net Profit",        f"${netp:,.2f}",    nc),
        ("Gross Profit",      f"${gp:,.2f}",      "positive"),
        ("Gross Loss",        f"${gl:,.2f}",      "negative" if gl > 0 else ""),
        ("Total Trades",      f"{n}  ({longs}L / {shts}S)", ""),
    ])

    # ── Win / Loss ──
    html += _section("Win / Loss", [
        ("Win Rate",          f"{wr:.1f}%",        wc),
        ("Profit Factor",     _pf_str(pf),          "positive" if pf >= 1.5 else ("negative" if pf < 1 else "")),
        ("Avg Win",           f"${aw:,.2f}",        "positive"),
        ("Avg Loss",          f"${al:,.2f}",        "negative" if al < 0 else ""),
    ])

    # ── Risk ──
    html += _section("Risk", [
        ("Max Drawdown",      f"${dd:,.2f}  ({ddpct:.1f}%)", "negative" if dd > 0 else ""),
        ("Largest Win",       f"${lw:,.2f}",        "positive"),
        ("Largest Loss",      f"${ll:,.2f}",        "negative" if ll < 0 else ""),
        ("Win / Loss Ratio",  _pf_str(wlr),          "positive" if wlr >= 1.5 else ("negative" if wlr < 1 else "")),
    ])

    # ── Quality ──
    html += _section("Quality", [
        ("Expectancy",        f"${exp:,.2f}",       ec),
        ("Avg Trade",         f"${avg:,.2f}",       ac),
        ("Sharpe Ratio",      f"{sh:.2f}",          shc),
        ("Sortino Ratio",     f"{so:.2f}",          soc),
    ])

    # ── Advanced ──
    html += _section("Advanced", [
        ("Calmar Ratio",      _pf_str(cal),          "positive" if cal != math.inf and cal >= 1 else ("negative" if cal != math.inf and cal < 0.5 else "")),
        ("Recovery Factor",   _pf_str(rf),           "positive" if rf  != math.inf and rf  >= 1 else ""),
        ("Max Consec. Wins",  str(cw),               "positive" if cw > 0 else ""),
        ("Max Consec. Losses",str(cl),               "negative" if cl > 3 else ""),
    ])

    # ── Exit breakdown ──
    total_exits = max(1, tp + sl + eod)
    html += _section("Exit Breakdown", [
        ("Take Profit (TP)",  f"{tp}  ({tp/total_exits*100:.0f}%)",  "positive" if tp > 0 else ""),
        ("Stop Loss (SL)",    f"{sl}  ({sl/total_exits*100:.0f}%)",  "negative" if sl > 0 else ""),
        ("End of Day (EOD)",  f"{eod} ({eod/total_exits*100:.0f}%)", ""),
        ("TP / SL Ratio",     f"{tp/sl:.2f}" if sl > 0 else "∞",    "positive" if sl == 0 or tp/sl >= 1 else "negative"),
    ])

    st.markdown(html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# Strategy loader
# ══════════════════════════════════════════════════════════════════════════════

def load_strategy_from_source(src: str, filename: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, prefix="strat_") as f:
        f.write(src)
        tmp = f.name
    try:
        spec = importlib.util.spec_from_file_location("_user_strategy", tmp)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.unlink(tmp)
    candidates = [obj for _, obj in vars(mod).items()
                  if isinstance(obj, type) and issubclass(obj, Strategy) and obj is not Strategy]
    if not candidates:
        raise ValueError("No Strategy subclass found in the uploaded script.")
    if len(candidates) > 1:
        raise ValueError(f"Multiple Strategy subclasses: {', '.join(c.__name__ for c in candidates)}. Keep one.")
    return candidates[0]


# ══════════════════════════════════════════════════════════════════════════════
# App header
# ══════════════════════════════════════════════════════════════════════════════

now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d  %H:%M  UTC")
st.markdown(f"""
<div class="app-header">
  <div>
    <div class="logo">TESTIS</div>
    <div class="subtitle">Algorithmic Backtesting &amp; Real-Time Monitor</div>
  </div>
  <div class="badge">● {now_str}</div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

tab_bt, tab_rt = st.tabs(["⬛  BACKTESTER", "◉  REAL-TIME"])


# ─────────────────────────────────────────────────────────────────────────────
#  BACKTESTER TAB
# ─────────────────────────────────────────────────────────────────────────────

with tab_bt:
    col_ctrl, col_main = st.columns([1, 3], gap="medium")

    # ── LEFT CONTROL PANEL ──────────────────────────────────────────────────
    with col_ctrl:

        # 01 Data
        st.markdown('<div class="panel-title">01 / DATA SOURCE</div>', unsafe_allow_html=True)
        csv_file = st.file_uploader("CSV file", type=["csv"], key="csv_up",
                                    label_visibility="collapsed")
        if csv_file and csv_file.name != st.session_state["data_name"]:
            try:
                df_raw = pd.read_csv(csv_file)
                dt_col = "datetime" if "datetime" in df_raw.columns else df_raw.columns[0]
                if dt_col != "datetime":
                    log(f"No 'datetime' column — using '{dt_col}'", "warn")
                try:
                    dt = pd.to_datetime(df_raw[dt_col], format="%Y%m%d %H%M", utc=True)
                except Exception:
                    dt = pd.to_datetime(df_raw[dt_col], utc=True)
                df_raw = df_raw.drop(columns=[dt_col]).rename(columns=str.lower)
                df_raw["datetime"] = dt
                df_raw = df_raw.set_index("datetime").sort_index()
                for c in ["open","high","low","close","volume"]:
                    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")
                df_raw = df_raw.dropna(subset=["open","high","low","close","volume"])
                st.session_state["data_df"]   = df_raw[["open","high","low","close","volume"]]
                st.session_state["data_name"] = csv_file.name
                log(f"CSV loaded: {csv_file.name}  ({len(df_raw):,} rows)", "info")
                log(f"Range: {df_raw.index.min()} → {df_raw.index.max()}", "data")
            except Exception as e:
                log(f"CSV load error: {e}", "error")

        if st.session_state["data_df"] is not None:
            d = st.session_state["data_df"]
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;'
                f'color:#2a8050;padding:4px 0 8px 0;">'
                f'✔ {st.session_state["data_name"]}<br>'
                f'<span style="color:#2a5060">{len(d):,} bars &nbsp;·&nbsp; '
                f'{d.index.min().strftime("%Y-%m-%d")} → {d.index.max().strftime("%Y-%m-%d")}'
                f'</span></div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # 02 Strategy
        st.markdown('<div class="panel-title">02 / STRATEGY SCRIPT</div>', unsafe_allow_html=True)
        strat_file = st.file_uploader("Python file", type=["py"], key="strat_up",
                                      label_visibility="collapsed")
        if strat_file and strat_file.name != st.session_state["strategy_name"]:
            src = strat_file.read().decode("utf-8")
            st.session_state["strategy_src"]  = src
            st.session_state["strategy_name"] = strat_file.name
            log(f"Strategy loaded: {strat_file.name}", "info")
            try:
                cls = load_strategy_from_source(src, strat_file.name)
                log(f"Class found: {cls.__name__}  ✔", "info")
            except Exception as e:
                log(f"Strategy validation: {e}", "error")

        if st.session_state["strategy_name"]:
            st.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.68rem;'
                f'color:#2a8050;padding:4px 0 8px 0;">✔ {st.session_state["strategy_name"]}</div>',
                unsafe_allow_html=True,
            )

        st.markdown("<hr>", unsafe_allow_html=True)

        # 03 Instrument
        st.markdown('<div class="panel-title">03 / INSTRUMENT</div>', unsafe_allow_html=True)
        usd_per_point   = st.number_input("USD / Point",       min_value=0.01, value=2.0,  step=0.5,  format="%.2f")
        commission_rt   = st.number_input("Commission RT ($)", min_value=0.0,  value=1.22, step=0.25, format="%.2f")
        slippage_points = st.number_input("Slippage (pts)",    min_value=0.0,  value=0.0,  step=0.25, format="%.2f")

        st.markdown("<hr>", unsafe_allow_html=True)

        # 04 Session
        st.markdown('<div class="panel-title">04 / SESSION FILTER  (UTC)</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            session_start = st.text_input("Start", value="14:30", placeholder="HH:MM")
        with c2:
            session_end   = st.text_input("End",   value="21:00", placeholder="HH:MM")

        st.markdown("<hr>", unsafe_allow_html=True)

        # 05 Chart
        st.markdown('<div class="panel-title">05 / CHART OPTIONS</div>', unsafe_allow_html=True)
        tf_map   = {"1 min":"1min","5 min":"5min","15 min":"15min","30 min":"30min",
                    "1 hour":"60min","4 hours":"4H","1 day":"1D"}
        chart_tf = st.selectbox("Timeframe", list(tf_map.keys()), index=0)
        show_trades = st.checkbox("Overlay trade markers", value=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        run_btn   = st.button("▶  RUN BACKTEST", type="primary",   use_container_width=True)
        clear_btn = st.button("✕  CLEAR",        type="secondary", use_container_width=True)

        if clear_btn:
            st.session_state["trades_df"] = None
            st.session_state["stats"]     = None
            log("Results cleared.", "dim")
            st.rerun()

    # ── RIGHT MAIN PANEL ────────────────────────────────────────────────────
    with col_main:

        # Run backtest
        if run_btn:
            errors = []
            if st.session_state["data_df"] is None:
                errors.append("No CSV loaded.")
            if st.session_state["strategy_src"] is None:
                errors.append("No strategy script loaded.")
            for e in errors:
                log(f"ERROR: {e}", "error")

            if not errors:
                log("─" * 55, "dim")
                log("BACKTEST STARTED", "info")
                log(f"Strategy : {st.session_state['strategy_name']}", "data")
                log(f"Params   : ${usd_per_point}/pt  commission=${commission_rt}  slip={slippage_points}pts", "data")
                log(f"Session  : {session_start or 'all'} → {session_end or 'all'} UTC", "data")

                with st.spinner("Running…"):
                    try:
                        StratClass = load_strategy_from_source(
                            st.session_state["strategy_src"],
                            st.session_state["strategy_name"],
                        )
                        log(f"Instantiating {StratClass.__name__}", "info")
                        strategy = StratClass()

                        broker = Broker(
                            usd_per_point=usd_per_point,
                            commission_rt=commission_rt,
                            slippage_points=slippage_points,
                        )
                        engine = Backtester(
                            broker=broker,
                            group_by_day=True,
                            session_start_utc=session_start.strip() or None,
                            session_end_utc=session_end.strip()   or None,
                        )

                        old_out, old_err = sys.stdout, sys.stderr
                        sys.stdout = TerminalCapture("data")
                        sys.stderr = TerminalCapture("error")
                        try:
                            df_1min = resample_ohlcv(st.session_state["data_df"], "1min")
                            trades  = engine.run(df_1min, strategy=strategy, resample_rule="1min")
                        finally:
                            sys.stdout = old_out
                            sys.stderr = old_err

                        st.session_state["trades_df"] = trades
                        st.session_state["stats"]     = equity_and_stats(trades)
                        net = st.session_state["stats"].get("net_profit", 0)
                        log(f"Complete — {len(trades)} trades  net={net:+.2f}", "info")
                        log("─" * 55, "dim")

                    except Exception as e:
                        for line in traceback.format_exc().splitlines():
                            log(line, "error")
                        st.error(f"Backtest error: {e}")

                st.rerun()

        # Price chart
        if st.session_state["data_df"] is not None:
            df_vis    = st.session_state["data_df"]
            rule      = tf_map[chart_tf]
            trades_vis = st.session_state["trades_df"]

            # When trade markers are on, restrict candle window to trade date range
            # so that candles and markers always share the same time axis.
            if show_trades and trades_vis is not None and not trades_vis.empty:
                try:
                    entry_t = pd.to_datetime(trades_vis["entry_time"], utc=True, errors="coerce")
                    exit_t  = pd.to_datetime(trades_vis["exit_time"],  utc=True, errors="coerce")
                    all_t   = pd.concat([entry_t, exit_t]).dropna()
                    if not all_t.empty:
                        pad    = pd.Timedelta(hours=2)
                        df_vis = df_vis.loc[all_t.min() - pad : all_t.max() + pad]
                except Exception:
                    pass
            else:
                if len(df_vis) > 5000:
                    df_vis = df_vis.iloc[-5000:]

            if rule != "1min":
                try:
                    df_vis = resample_ohlcv(df_vis, rule)
                except Exception:
                    pass

            title = f"{st.session_state['data_name'] or ''}  ·  {chart_tf}"
            st.plotly_chart(
                build_candle_chart(df_vis, trades_vis, show_trades, title),
                use_container_width=True, config={"displayModeBar": True},
            )
        else:
            st.markdown(
                '<div style="height:460px;display:flex;align-items:center;justify-content:center;'
                'background:#0c1520;border:1px solid #1a2e40;border-radius:6px;'
                'color:#1a3040;font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;">'
                '// NO DATA LOADED — UPLOAD A CSV TO BEGIN</div>',
                unsafe_allow_html=True,
            )

        # Stats
        if st.session_state["stats"]:
            st.markdown('<div class="panel-title" style="margin-top:1.2rem;">PERFORMANCE METRICS</div>',
                        unsafe_allow_html=True)
            render_stats(st.session_state["stats"])

            eq_col, dist_col = st.columns(2)
            with eq_col:
                st.plotly_chart(build_equity_chart(st.session_state["trades_df"]),
                                use_container_width=True, config={"displayModeBar": False})
            with dist_col:
                st.plotly_chart(build_pnl_dist(st.session_state["trades_df"]),
                                use_container_width=True, config={"displayModeBar": False})

        # Trade log
        if st.session_state["trades_df"] is not None and not st.session_state["trades_df"].empty:
            with st.expander("▼  TRADE LOG", expanded=False):
                df_show = st.session_state["trades_df"].copy()
                def _color_net(val):
                    return "color: #00c87a" if val > 0 else "color: #e05050"
                styled = (
                    df_show.style
                    .applymap(_color_net, subset=["net","gross"])
                    .format({"entry_price":"{:.2f}","exit_price":"{:.2f}",
                             "gross":"${:,.2f}","net":"${:,.2f}"})
                )
                st.dataframe(styled, use_container_width=True, height=300)

        # Terminal
        st.markdown('<div class="panel-title" style="margin-top:1.2rem;">CONSOLE OUTPUT</div>',
                    unsafe_allow_html=True)
        render_terminal()
        if st.button("Clear console", type="secondary", key="clr_log_bt"):
            st.session_state["log"] = []
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  REAL-TIME TAB
# ─────────────────────────────────────────────────────────────────────────────

with tab_rt:
    col_rt_ctrl, col_rt_main = st.columns([1, 3], gap="medium")

    with col_rt_ctrl:
        st.markdown('<div class="panel-title">CONNECTION</div>', unsafe_allow_html=True)
        rt_symbol = st.text_input("Symbol", value="MNQ", key="rt_sym")
        rt_port   = st.number_input("UDP Port", value=19999, step=1, format="%d", key="rt_port")
        rt_tf     = st.selectbox("Chart Timeframe", ["1s","5s","15s","30s","1min","5min"], key="rt_tf")
        rt_depth  = st.number_input("Order book depth", min_value=5, max_value=30, value=10, key="rt_depth")
        rt_sim    = st.checkbox("Simulate L1 (no NinjaTrader)", value=False, key="rt_sim")

        st.markdown("<hr>", unsafe_allow_html=True)

        con_btn  = st.button("▶  CONNECT",    type="primary",   use_container_width=True, key="rt_con")
        dis_btn  = st.button("◼  DISCONNECT", type="secondary", use_container_width=True, key="rt_dis")
        ref_btn  = st.button("↻  REFRESH",    type="secondary", use_container_width=True, key="rt_ref")

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<div class="panel-title">STATUS</div>', unsafe_allow_html=True)
        connected  = st.session_state["rt_connected"]
        sc = "#00c87a" if connected else "#e05050"
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.8rem;color:{sc};">'
            f'● {"CONNECTED" if connected else "DISCONNECTED"}</div>'
            f'<div style="font-family:\'Share Tech Mono\',monospace;font-size:0.65rem;color:#2a5060;margin-top:4px;">'
            f'Ticks: {st.session_state["rt_tick_count"]:,}<br>'
            f'L2 events: {st.session_state["rt_l2_count"]:,}</div>',
            unsafe_allow_html=True,
        )

    with col_rt_main:

        if con_btn and not st.session_state["rt_connected"]:
            if rt_sim:
                def _sim():
                    rng = np.random.default_rng()
                    px  = 20000.0
                    while st.session_state.get("rt_connected", False):
                        time.sleep(0.05)
                        px = max(0.25, px + rng.normal(0, 0.6))
                        now = datetime.now(timezone.utc)
                        st.session_state["rt_ticks"].append({
                            "ts": now, "last": round(px,2),
                            "bid": round(px-0.25,2), "ask": round(px+0.25,2),
                            "size": int(rng.integers(1,5)),
                        })
                        st.session_state["rt_tick_count"] += 1
                st.session_state["rt_connected"] = True
                threading.Thread(target=_sim, daemon=True).start()
                log(f"Simulation started ({rt_symbol})", "info")
            else:
                def _udp(port):
                    try:
                        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                        sock.bind(("127.0.0.1", port))
                        sock.settimeout(0.5)
                    except Exception as e:
                        log(f"UDP bind failed: {e}", "error")
                        st.session_state["rt_connected"] = False
                        return
                    while st.session_state.get("rt_connected", False):
                        try:
                            data, _ = sock.recvfrom(16384)
                            msg = json.loads(data.decode("utf-8").strip())
                            typ = msg.get("type")
                            ts  = datetime.now(timezone.utc)
                            if typ == "trade":
                                st.session_state["rt_ticks"].append({
                                    "ts": ts, "last": float(msg["price"]),
                                    "size": int(msg.get("size",1)), "bid": None, "ask": None,
                                })
                                st.session_state["rt_tick_count"] += 1
                            elif typ == "quote":
                                st.session_state["rt_ticks"].append({
                                    "ts": ts, "last": None,
                                    "bid": float(msg["bid"]), "ask": float(msg["ask"]), "size": None,
                                })
                                st.session_state["rt_tick_count"] += 1
                            elif typ == "depth":
                                side  = msg.get("side"); op = msg.get("op")
                                price = float(msg["price"]); size = int(msg.get("size",0))
                                book  = st.session_state["rt_bids"] if side=="B" \
                                        else st.session_state["rt_asks"]
                                if op in ("insert","update"):
                                    if size <= 0: book.pop(price, None)
                                    else: book[price] = size
                                elif op == "delete":
                                    book.pop(price, None)
                                st.session_state["rt_l2_count"] += 1
                        except socket.timeout:
                            continue
                        except Exception as e:
                            log(f"UDP error: {e}", "error")
                            time.sleep(0.05)
                    sock.close()
                st.session_state["rt_connected"] = True
                threading.Thread(target=_udp, args=(int(rt_port),), daemon=True).start()
                log(f"UDP listener started on port {rt_port}", "info")
            st.rerun()

        if dis_btn and st.session_state["rt_connected"]:
            st.session_state["rt_connected"] = False
            log("Real-time feed stopped.", "warn")
            st.rerun()

        # Live chart
        ticks = list(st.session_state["rt_ticks"])
        if ticks:
            df_rt = pd.DataFrame(ticks)
            for c in ["last","bid","ask","size"]:
                if c in df_rt.columns:
                    df_rt[c] = pd.to_numeric(df_rt[c], errors="coerce")
            df_rt.index = pd.to_datetime([t["ts"] for t in ticks], utc=True)
            df_rt = df_rt.sort_index()
            if "bid" in df_rt and "ask" in df_rt:
                mid = (df_rt["bid"] + df_rt["ask"]) / 2
            else:
                mid = None
            df_rt["px"] = df_rt.get("last", pd.Series(dtype=float)).fillna(mid)

            rule_map = {"1s":"1S","5s":"5S","15s":"15S","30s":"30S","1min":"1min","5min":"5min"}
            ohlc = df_rt["px"].resample(rule_map.get(rt_tf,"1S")).ohlc().dropna()

            fig_rt = _base_fig()
            if not ohlc.empty:
                disp = ohlc.iloc[-300:]
                ix   = disp.index.tz_localize(None) if disp.index.tz else disp.index
                fig_rt.add_trace(go.Candlestick(
                    x=ix, open=disp["open"], high=disp["high"],
                    low=disp["low"], close=disp["close"],
                    increasing=dict(line=dict(color="#00c87a"), fillcolor="#00c87a"),
                    decreasing=dict(line=dict(color="#e05050"), fillcolor="#e05050"),
                ))
            fig_rt.update_layout(
                xaxis_rangeslider_visible=False, height=400,
                title=dict(text=f"LIVE  ·  {rt_symbol}  ·  {rt_tf}", font=dict(size=11,color="#2a6080")),
            )
            st.plotly_chart(fig_rt, use_container_width=True, config={"displayModeBar": True})
        else:
            st.markdown(
                '<div style="height:400px;display:flex;align-items:center;justify-content:center;'
                'background:#0c1520;border:1px solid #1a2e40;border-radius:6px;'
                'color:#1a3040;font-family:\'Share Tech Mono\',monospace;font-size:0.85rem;">'
                '// WAITING FOR DATA — CONNECT TO BEGIN</div>',
                unsafe_allow_html=True,
            )

        # Order book
        st.markdown('<div class="panel-title" style="margin-top:1rem;">LIVE ORDER BOOK</div>',
                    unsafe_allow_html=True)
        depth_n    = int(rt_depth)
        bid_levels = sorted(st.session_state["rt_bids"].items(), key=lambda x: x[0], reverse=True)[:depth_n]
        ask_levels = sorted(st.session_state["rt_asks"].items(), key=lambda x: x[0])[:depth_n]

        if bid_levels or ask_levels:
            mr = max(len(bid_levels), len(ask_levels))
            bid_levels += [(None,None)] * (mr - len(bid_levels))
            ask_levels += [(None,None)] * (mr - len(ask_levels))
            ob_df = pd.DataFrame({
                "Bid Size":  [str(s) if s is not None else "" for _,s in bid_levels],
                "Bid Price": [str(p) if p is not None else "" for p,_ in bid_levels],
                "Ask Price": [str(p) if p is not None else "" for p,_ in ask_levels],
                "Ask Size":  [str(s) if s is not None else "" for _,s in ask_levels],
            })
            st.dataframe(ob_df, use_container_width=True, height=min(35*depth_n+40, 400))
        else:
            st.markdown(
                '<div style="padding:1rem;background:#0a1520;border:1px solid #1a2e40;border-radius:4px;'
                'font-family:\'Share Tech Mono\',monospace;font-size:0.72rem;color:#1a3040;">'
                '// No L2 data yet</div>',
                unsafe_allow_html=True,
            )

        # Terminal in RT tab
        st.markdown('<div class="panel-title" style="margin-top:1rem;">CONSOLE OUTPUT</div>',
                    unsafe_allow_html=True)
        render_terminal()
        if ref_btn:
            st.rerun()
