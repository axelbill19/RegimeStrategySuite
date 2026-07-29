"""
Current Regime + Signal Dashboard
===================================
Responde dos preguntas concretas:
  1. En que regimen de mercado estamos HOY?
  2. Los candidatos value (ADBE, PYPL, MSFT) tienen senal tecnica activa?

Fuentes:
  - Datos/extended_current.parquet  → precios de cierre hasta junio 2026 (regimen)
  - yfinance                        → OHLCV reciente para SAR + EMA200 + RSI
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import yfinance as yf
from ta.trend import EMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator

# ── Config ─────────────────────────────────────────────────────────────────────
PRICES_PATH     = Path("../Datos/extended_current.parquet")
CANDIDATES      = ["ADBE", "PYPL", "MSFT"]
TREND_WINDOW    = 60
FEAR_PERCENTILE = 0.65
PSAR_STEP       = 0.02
PSAR_MAX        = 0.20
EMA_WIN         = 200
RSI_WIN         = 14
RSI_LO          = 40
RSI_HI          = 75
LOOKBACK_MONTHS = 18    # cuanto historico mostrar en graficas

REGIME_COLORS = {
    "MARKUP":       "#2980b9",
    "DISTRIBUCION": "#e67e22",
    "ACUMULACION":  "#27ae60",
    "MARKDOWN":     "#c0392b",
    "UNKNOWN":      "#95a5a6",
}
REGIME_ES = {
    "MARKUP":       "MARKUP (tendencia+ / estres bajo)",
    "DISTRIBUCION": "DISTRIBUCION (tendencia+ / estres alto)",
    "ACUMULACION":  "ACUMULACION (tendencia- / estres bajo)",
    "MARKDOWN":     "MARKDOWN (tendencia- / estres alto)",
    "UNKNOWN":      "UNKNOWN",
}
REGIME_ACTION = {
    "MARKUP":       "Momentum domina. Invertir en senales tecnicas activas.",
    "DISTRIBUCION": "Precaucion. Reducir exposicion a la mitad.",
    "ACUMULACION":  "Zona value. Mejor momento para entrar con fundamentales.",
    "MARKDOWN":     "Defensivo. Cash. No atrapar el cuchillo.",
    "UNKNOWN":      "Sin clasificar.",
}


# ── 1. Regimen ─────────────────────────────────────────────────────────────────

def detect_regime(prices: pd.DataFrame) -> pd.Series:
    mkt_ret  = prices.mean(axis=1).pct_change()
    vol_20   = mkt_ret.rolling(20).std() * np.sqrt(252)
    vol_60   = mkt_ret.rolling(60).std() * np.sqrt(252)
    fear_lv  = vol_20 / vol_60.replace(0, np.nan)
    fear_pct = fear_lv.expanding().rank(pct=True)
    mom_60   = mkt_ret.rolling(TREND_WINDOW).mean()

    pos_trend   = mom_60 > 0
    high_stress = fear_pct >= FEAR_PERCENTILE

    r = pd.Series("UNKNOWN", index=prices.index)
    r[ pos_trend & ~high_stress] = "MARKUP"
    r[ pos_trend &  high_stress] = "DISTRIBUCION"
    r[~pos_trend & ~high_stress] = "ACUMULACION"
    r[~pos_trend &  high_stress] = "MARKDOWN"
    return r


# ── 2. Senales tecnicas para candidatos ────────────────────────────────────────

def get_candidate_signals(tickers: list) -> dict:
    """
    Descarga 2 anos de OHLCV via yfinance y calcula SAR + EMA200 + RSI.
    Retorna dict con DataFrame por ticker.
    """
    results = {}
    print(f"\nDescargando OHLCV para: {', '.join(tickers)}...")
    raw = yf.download(tickers, period="2y", auto_adjust=True, progress=False)

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                tk = raw.copy()
            else:
                tk = raw.xs(ticker, axis=1, level=1).copy()

            tk = tk.dropna()
            tk.index = pd.to_datetime(tk.index).tz_localize(None)

            close = tk["Close"]
            high  = tk["High"]
            low   = tk["Low"]

            psar   = PSARIndicator(high=high, low=low, close=close,
                                   step=PSAR_STEP, max_step=PSAR_MAX).psar()
            ema200 = EMAIndicator(close=close, window=EMA_WIN).ema_indicator()
            rsi    = RSIIndicator(close=close, window=RSI_WIN).rsi()

            sar_bull = close > psar
            above_ema = close > ema200
            rsi_ok = (rsi >= RSI_LO) & (rsi <= RSI_HI)
            signal = (sar_bull & above_ema & rsi_ok).astype(int)

            tk["psar"]    = psar
            tk["ema200"]  = ema200
            tk["rsi"]     = rsi
            tk["signal"]  = signal
            results[ticker] = tk
            print(f"  {ticker}: ok  ({tk.index[0].date()} a {tk.index[-1].date()})")
        except Exception as e:
            print(f"  {ticker}: error — {e}")

    return results


# ── 3. Graficas ────────────────────────────────────────────────────────────────

def _regime_bg(ax, regime_series):
    prev, t0 = None, regime_series.index[0]
    for t, r in regime_series.items():
        if r != prev and prev is not None:
            ax.axvspan(t0, t, alpha=0.12,
                       color=REGIME_COLORS.get(prev, "#888"), lw=0)
            t0 = t
        prev = r
    ax.axvspan(t0, regime_series.index[-1], alpha=0.12,
               color=REGIME_COLORS.get(prev, "#888"), lw=0)


def plot_dashboard(regime: pd.Series, candidates: dict):
    n = len(candidates)
    fig = plt.figure(figsize=(16, 5 + 4 * n))
    fig.suptitle("DASHBOARD: Regimen Actual + Senales Candidatos Value",
                 fontsize=13, fontweight="bold", y=0.99)

    # Layout: fila 0 = regimen | filas 1..n = candidatos (2 paneles cada uno)
    gs = gridspec.GridSpec(1 + n, 2, figure=fig, hspace=0.55, wspace=0.30)

    # ── Panel regimen (fila 0, span columnas) ──────────────────────────────────
    ax_r = fig.add_subplot(gs[0, :])
    cutoff = regime.index[-1] - pd.DateOffset(months=LOOKBACK_MONTHS)
    reg_plot = regime[regime.index >= cutoff]

    # Precio de mercado (EW) como fondo
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    mkt = prices.mean(axis=1)
    mkt_plot = mkt[mkt.index >= cutoff]
    mkt_norm  = mkt_plot / mkt_plot.iloc[0]

    ax_r2 = ax_r.twinx()
    ax_r2.plot(mkt_norm.index, mkt_norm.values, color="gray",
               lw=1.2, alpha=0.5, label="Mercado EW (norm.)")
    ax_r2.set_ylabel("Mercado (normalizado)", color="gray", fontsize=8)
    ax_r2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    _regime_bg(ax_r, reg_plot)
    # Linea de regimen como numero
    rmap = {"MARKUP": 4, "DISTRIBUCION": 3, "ACUMULACION": 2, "MARKDOWN": 1, "UNKNOWN": 0}
    reg_num = reg_plot.map(rmap)
    ax_r.step(reg_plot.index, reg_num.values, where="post",
              color="#2c3e50", lw=1.5)
    ax_r.set_yticks([1, 2, 3, 4])
    ax_r.set_yticklabels(["MARKDOWN", "ACUMULACION", "DISTRIBUCION", "MARKUP"],
                          fontsize=8)
    ax_r.set_title(f"Regimen de Mercado — ultimos {LOOKBACK_MONTHS} meses",
                   fontweight="bold")
    ax_r.grid(axis="x", alpha=0.2)
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.6, label=r)
               for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]]
    ax_r.legend(handles=patches, loc="lower left", fontsize=7.5, ncol=4)

    # ── Paneles por candidato ──────────────────────────────────────────────────
    for i, (ticker, tk) in enumerate(candidates.items(), 1):
        cutoff_tk = tk.index[-1] - pd.DateOffset(months=LOOKBACK_MONTHS)
        tk_plot   = tk[tk.index >= cutoff_tk]
        reg_tk    = regime.reindex(tk_plot.index, method="ffill")

        # Panel izquierdo: precio + SAR + EMA200
        ax_p = fig.add_subplot(gs[i, 0])
        _regime_bg(ax_p, reg_tk)
        ax_p.plot(tk_plot.index, tk_plot["Close"],
                  color="#2c3e50", lw=1.5, label="Precio")
        ax_p.plot(tk_plot.index, tk_plot["psar"],
                  color="#e74c3c", lw=0.8, ls=":", alpha=0.8, label="SAR")
        ax_p.plot(tk_plot.index, tk_plot["ema200"],
                  color="#2980b9", lw=1.2, ls="--", alpha=0.8, label="EMA 200")

        # Marcar dias con senal activa
        sig_on = tk_plot[tk_plot["signal"] == 1]
        if not sig_on.empty:
            ax_p.scatter(sig_on.index, sig_on["Close"],
                         color="#27ae60", s=12, zorder=5, alpha=0.6)

        ax_p.set_title(f"{ticker} — Precio / SAR / EMA200", fontweight="bold")
        ax_p.set_ylabel("USD")
        ax_p.legend(fontsize=7.5, loc="upper left")
        ax_p.grid(alpha=0.2)

        # Panel derecho: RSI
        ax_q = fig.add_subplot(gs[i, 1])
        _regime_bg(ax_q, reg_tk)
        ax_q.plot(tk_plot.index, tk_plot["rsi"],
                  color="#8e44ad", lw=1.4, label="RSI 14")
        ax_q.axhline(RSI_HI, color="#e74c3c", lw=0.9, ls="--",
                     label=f"Salida RSI {RSI_HI}")
        ax_q.axhline(RSI_LO, color="#27ae60", lw=0.9, ls="--",
                     label=f"Entrada min RSI {RSI_LO}")
        ax_q.axhline(50, color="gray", lw=0.7, ls=":", alpha=0.6)
        ax_q.fill_between(tk_plot.index, RSI_LO, RSI_HI,
                           alpha=0.08, color="#27ae60", label="Zona activa")
        ax_q.set_ylim(0, 100)
        ax_q.set_title(f"{ticker} — RSI 14", fontweight="bold")
        ax_q.set_ylabel("RSI")
        ax_q.legend(fontsize=7.5, loc="upper left")
        ax_q.grid(alpha=0.2)

    Path("outputs").mkdir(exist_ok=True)
    out = Path("outputs/current_regime_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nDashboard guardado: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  CURRENT REGIME + SIGNAL DASHBOARD")
    print("=" * 58)

    # Regimen
    print("\nCargando precios de mercado...")
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    print(f"  {prices.shape[1]} tickers | hasta {prices.index[-1].date()}")

    regime = detect_regime(prices)
    today_regime = regime.iloc[-1]

    # Ultimos 10 dias de regimen
    print("\nRegimen reciente (ultimos 10 dias):")
    recent = regime.tail(10)
    for date, r in recent.items():
        marker = " <<< HOY" if date == regime.index[-1] else ""
        print(f"  {date.date()}  {r}{marker}")

    print(f"\n{'='*58}")
    print(f"  REGIMEN ACTUAL: {today_regime}")
    print(f"  {REGIME_ES[today_regime]}")
    print(f"\n  Accion sugerida:")
    print(f"  {REGIME_ACTION[today_regime]}")
    print(f"{'='*58}")

    # Distribucion ultimos 6 meses
    last_6m = regime[regime.index >= regime.index[-1] - pd.DateOffset(months=6)]
    print(f"\nDistribucion de regimen (ultimos 6 meses):")
    dist = last_6m.value_counts()
    for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]:
        n = dist.get(r, 0)
        bar = "#" * int(n / max(dist.values) * 20)
        print(f"  {r:<14} {n:>4} dias  {bar}")

    # Senales candidatos
    candidates = get_candidate_signals(CANDIDATES)
    print(f"\n{'='*58}")
    print("  SENALES TECNICAS — CANDIDATOS VALUE")
    print(f"{'='*58}")
    for ticker, tk in candidates.items():
        last  = tk.iloc[-1]
        sig   = "ACTIVA" if last["signal"] == 1 else "INACTIVA"
        sar_s = "alcista" if last["Close"] > last["psar"] else "BAJISTA"
        ema_s = "arriba" if last["Close"] > last["ema200"] else "ABAJO"
        print(f"\n  {ticker}  [{sig}]")
        print(f"    Precio:  ${last['Close']:.2f}    SAR: ${last['psar']:.2f} ({sar_s})")
        print(f"    EMA200:  ${last['ema200']:.2f} ({ema_s} de EMA)")
        print(f"    RSI:     {last['rsi']:.1f}  (zona activa: {RSI_LO}-{RSI_HI})")
        # Cuantos dias consecutivos con senal activa
        streak = 0
        for v in reversed(tk["signal"].values):
            if v == 1:
                streak += 1
            else:
                break
        if streak > 0:
            print(f"    Senal activa hace {streak} dias consecutivos")

    plot_dashboard(regime, candidates)
