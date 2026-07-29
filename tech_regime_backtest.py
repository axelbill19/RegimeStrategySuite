"""
Tech + Regime Combined Backtest
================================
Combina dos señales independientes:

  1. SEÑAL TÉCNICA  : SAR alcista + precio > EMA200 + RSI [40-75]
  2. FILTRO RÉGIMEN : tendencia_60d × estrés_vol (4 cuadrantes)

Lógica de combinación (régimen lagged 1 día):
  MARKUP       → invertir en stocks con señal técnica = 1
  ACUMULACIÓN  → invertir en stocks con señal técnica = 1
  DISTRIBUCIÓN → 50% señal técnica + 50% cash
  MARKDOWN     → 100% cash (sin importar señal técnica)

Estrategias comparadas:
  S0  Benchmark EW       — todos los stocks, igual peso, siempre
  S1  Tech solo          — SAR+EMA200+RSI sin filtro de régimen
  S2  Régimen solo       — EW filtrado por régimen (sin señal técnica)
  S3  Tech + Régimen     — combinación completa

Datos: Datos/stock_details_5_years.csv (491 S&P 500, 2018-2023)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from ta.trend import EMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator
from scipy import stats

# ── Config ─────────────────────────────────────────────────────────────────────
DATA_PATH  = Path("../Datos/stock_details_5_years.csv")
OUT_DIR    = Path("outputs")

# Técnicos
PSAR_STEP = 0.02
PSAR_MAX  = 0.20
EMA_WIN   = 200
RSI_WIN   = 14
RSI_LO    = 40
RSI_HI    = 75
MIN_ROWS  = 260

# Régimen
TREND_WINDOW    = 60
FEAR_PERCENTILE = 0.65

RF_ANNUAL = 0.05
RF_DAILY  = RF_ANNUAL / 252

REGIME_COLORS = {
    "MARKUP":       "#2980b9",
    "DISTRIBUCION": "#e67e22",
    "ACUMULACION":  "#27ae60",
    "MARKDOWN":     "#c0392b",
    "UNKNOWN":      "#95a5a6",
}

STRAT_META = {
    "s0_bench":  {"label": "S0  Benchmark EW",    "color": "#95a5a6", "lw": 1.5, "ls": ":"},
    "s1_tech":   {"label": "S1  Tech solo",        "color": "#e74c3c", "lw": 1.5, "ls": "--"},
    "s2_regime": {"label": "S2  Regimen solo",     "color": "#2980b9", "lw": 2.0, "ls": "--"},
    "s3_combo":  {"label": "S3  Tech + Regimen",   "color": "#27ae60", "lw": 2.5, "ls": "-"},
}


# ── 1. Datos ───────────────────────────────────────────────────────────────────

def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)
    return df


# ── 2. Señales técnicas (por ticker) ──────────────────────────────────────────

def compute_signals_ticker(tk: pd.DataFrame) -> pd.DataFrame:
    close = tk["Close"]
    high  = tk["High"]
    low   = tk["Low"]

    psar   = PSARIndicator(high=high, low=low, close=close,
                           step=PSAR_STEP, max_step=PSAR_MAX).psar()
    ema200 = EMAIndicator(close=close, window=EMA_WIN).ema_indicator()
    rsi    = RSIIndicator(close=close, window=RSI_WIN).rsi()

    signal = ((close > psar) & (close > ema200) &
              (rsi >= RSI_LO) & (rsi <= RSI_HI)).astype(int)

    out = tk[["Date", "Close"]].copy()
    out["ret"]    = close.pct_change()
    out["signal"] = signal.values
    return out


def compute_all_signals(df: pd.DataFrame) -> pd.DataFrame:
    parts = []
    tickers = df["Company"].unique()
    print(f"Calculando senales tecnicas para {len(tickers)} tickers...")
    for i, ticker in enumerate(tickers, 1):
        tk = df[df["Company"] == ticker].copy().reset_index(drop=True)
        if len(tk) < MIN_ROWS:
            continue
        try:
            sig = compute_signals_ticker(tk)
            sig["ticker"] = ticker
            parts.append(sig)
        except Exception:
            continue
        if i % 100 == 0:
            print(f"  {i}/{len(tickers)}")
    print(f"  Completado: {len(parts)} tickers procesados")
    return pd.concat(parts, ignore_index=True)


# ── 3. Régimen de mercado ──────────────────────────────────────────────────────

def detect_regimes(df: pd.DataFrame) -> pd.Series:
    """
    Régimen diario sin lookahead.
    Usa el precio promedio de todos los tickers como proxy de mercado.
    """
    prices = (df.pivot(index="Date", columns="Company", values="Close")
               .ffill())
    mkt_ret = prices.mean(axis=1).pct_change()

    vol_20   = mkt_ret.rolling(20).std() * np.sqrt(252)
    vol_60   = mkt_ret.rolling(60).std() * np.sqrt(252)
    fear_lv  = vol_20 / vol_60.replace(0, np.nan)
    fear_pct = fear_lv.expanding().rank(pct=True)
    mom_60   = mkt_ret.rolling(TREND_WINDOW).mean()

    pos_trend   = mom_60 > 0
    high_stress = fear_pct >= FEAR_PERCENTILE

    regime = pd.Series("UNKNOWN", index=prices.index)
    regime[ pos_trend & ~high_stress] = "MARKUP"
    regime[ pos_trend &  high_stress] = "DISTRIBUCION"
    regime[~pos_trend & ~high_stress] = "ACUMULACION"
    regime[~pos_trend &  high_stress] = "MARKDOWN"

    dist = regime.value_counts()
    print("\nDistribucion de regimen:")
    for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN", "UNKNOWN"]:
        n = dist.get(r, 0)
        print(f"  {r:<14} {n:>5} dias  ({n/len(regime):>5.1%})")
    return regime


# ── 4. Backtest combinado ──────────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame, regime: pd.Series) -> pd.DataFrame:
    """
    Construye retornos diarios de las 4 estrategias.
    Señal técnica lagged 1 día. Régimen lagged 1 día.
    """
    signals = signals.sort_values(["ticker", "Date"])
    signals["sig_lag"]    = signals.groupby("ticker")["signal"].shift(1)
    signals["regime_lag"] = signals["Date"].map(regime.shift(1))

    # ── Portfolio diario por estrategia ───────────────────────────────────────
    # Para cada día: retorno de los stocks activos según lógica de cada estrategia

    # S0: benchmark — todos los stocks, igual peso
    bench_daily = (signals.groupby("Date")
                   .agg(s0=("ret", "mean"))
                   .reset_index())

    # S1: tech solo — sig_lag == 1
    tech_in = signals[signals["sig_lag"] == 1]
    tech_daily = (tech_in.groupby("Date")
                  .agg(s1_full=("ret", "mean"), n_tech=("ticker", "count"))
                  .reset_index())

    # S2: régimen solo — EW filtrado por régimen (sin señal técnica)
    # MARKUP/ACUMULACION: benchmark EW; DISTRIBUCION: 50% EW + 50% cash; MARKDOWN: cash
    # Implementamos como multiplicador sobre benchmark EW
    def regime_mult(r):
        return {"MARKUP": 1.0, "ACUMULACION": 1.0,
                "DISTRIBUCION": 0.5, "MARKDOWN": 0.0, "UNKNOWN": 0.75}.get(r, 0.75)

    bench_daily["regime_lag"] = bench_daily["Date"].map(regime.shift(1))
    bench_daily["s2_mult"]    = bench_daily["regime_lag"].map(regime_mult)
    bench_daily["s2"]         = bench_daily["s0"] * bench_daily["s2_mult"]

    # S3: tech + régimen — señal técnica filtrada por régimen
    # En MARKDOWN: cash. En DISTRIBUCION: 50% señal + 50% cash.
    # En MARKUP/ACUMULACION: 100% señal técnica.
    def build_s3(row_date, sig_ret, regime_val):
        mult = {"MARKUP": 1.0, "ACUMULACION": 1.0,
                "DISTRIBUCION": 0.5, "MARKDOWN": 0.0, "UNKNOWN": 0.75}.get(regime_val, 0.75)
        return sig_ret * mult

    merged = bench_daily.merge(tech_daily, on="Date", how="left")
    merged["s1_full"]  = merged["s1_full"].fillna(0)
    merged["n_tech"]   = merged["n_tech"].fillna(0).astype(int)
    merged["s3"]       = merged.apply(
        lambda row: build_s3(row["Date"], row["s1_full"], row["regime_lag"]), axis=1
    )

    # Recortar warmup
    merged = merged[merged["Date"] >= "2019-06-01"].reset_index(drop=True)
    merged["regime_lag"] = merged["regime_lag"].fillna("UNKNOWN")

    # Equity curves
    for col in ["s0", "s1_full", "s2", "s3"]:
        merged[f"eq_{col}"] = (1 + merged[col]).cumprod()

    return merged


# ── 5. Métricas ────────────────────────────────────────────────────────────────

def calc_metrics(returns: pd.Series, label: str) -> dict:
    r = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    n_years  = len(r) / 252
    total    = (1 + r).prod() - 1
    cagr     = (1 + total) ** (1 / n_years) - 1
    vol      = r.std() * np.sqrt(252)
    downside = r[r < 0].std() * np.sqrt(252)
    sharpe   = (cagr - RF_ANNUAL) / vol      if vol      > 0 else 0
    sortino  = (cagr - RF_ANNUAL) / downside if downside > 0 else 0
    eq       = (1 + r).cumprod()
    max_dd   = ((eq - eq.cummax()) / eq.cummax()).min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0
    pct_inv  = (r != 0).mean()
    return dict(label=label, total=total, cagr=cagr, vol=vol,
                sharpe=sharpe, sortino=sortino, max_dd=max_dd,
                calmar=calmar, pct_inv=pct_inv)


def print_metrics(m: dict):
    print(f"\n  {m['label']}")
    print(f"  {'Total':>10} {'CAGR':>7} {'Vol':>7} {'Sharpe':>8} "
          f"{'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'%Inv':>6}")
    print(f"  {m['total']*100:>+9.1f}%"
          f" {m['cagr']*100:>6.1f}%"
          f" {m['vol']*100:>6.1f}%"
          f" {m['sharpe']:>8.2f}"
          f" {m['sortino']:>8.2f}"
          f" {m['max_dd']*100:>7.1f}%"
          f" {m['calmar']:>8.2f}"
          f" {m['pct_inv']*100:>5.0f}%")


# ── 6. Visualizaciones ─────────────────────────────────────────────────────────

def _add_regime_bg(ax, regime_series: pd.Series):
    prev, t0 = None, regime_series.index[0]
    for t, r in regime_series.items():
        if r != prev and prev is not None:
            ax.axvspan(t0, t, alpha=0.09,
                       color=REGIME_COLORS.get(prev, "#888"), linewidth=0)
            t0 = t
        prev = r
    ax.axvspan(t0, regime_series.index[-1], alpha=0.09,
               color=REGIME_COLORS.get(prev, "#888"), linewidth=0)


def plot_results(result: pd.DataFrame, metrics_list: list):
    dates  = result["Date"]
    regime = result.set_index("Date")["regime_lag"]

    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "TECH + REGIME COMBINED BACKTEST\n"
        "SAR + EMA200 + RSI  x  Regimen de Mercado  |  S&P 500 2019-2023",
        fontsize=13, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)

    # ── 1. Curvas de capital ───────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    _add_regime_bg(ax1, regime)
    col_map = {"s0": "s0_bench", "s1_full": "s1_tech",
               "s2": "s2_regime", "s3": "s3_combo"}
    for ret_col, strat_key in col_map.items():
        m = STRAT_META[strat_key]
        eq = (1 + result[ret_col]).cumprod()
        ax1.plot(dates, eq, color=m["color"], lw=m["lw"], ls=m["ls"], label=m["label"])
    ax1.set_title("Curvas de Capital con Overlay de Regimen", fontweight="bold")
    ax1.set_ylabel("Capital ($1 inicial)")
    ax1.axhline(1, color="gray", lw=0.8, ls=":")
    ax1.grid(alpha=0.2)

    lines = [plt.Line2D([0],[0], color=v["color"], lw=v["lw"], ls=v["ls"],
                         label=v["label"]) for v in STRAT_META.values()]
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.5, label=r)
               for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]]
    leg1 = ax1.legend(handles=lines,   loc="upper left",  fontsize=8.5)
    ax1.add_artist(leg1)
    ax1.legend(handles=patches, loc="lower right", fontsize=8, title="Regimen")

    # ── 2. Drawdown ────────────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    _add_regime_bg(ax2, regime)
    for ret_col, strat_key in col_map.items():
        m  = STRAT_META[strat_key]
        eq = (1 + result[ret_col]).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        ax2.fill_between(dates, dd, 0, alpha=0.25, color=m["color"])
        ax2.plot(dates, dd, color=m["color"], lw=1.0, alpha=0.9,
                 label=f"{m['label'].split()[1]}  {dd.min():.1f}%")
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.2)

    # ── 3. Stocks en posición (S1 vs S3) ──────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.fill_between(dates, result["n_tech"], color="#e74c3c", alpha=0.4, label="S1 Tech solo")
    ax3.plot(dates, result["n_tech"], color="#e74c3c", lw=0.8)
    # S3 stocks = S1 * mult (aproximado: n_tech × mult)
    mult_map = {"MARKUP":1.0,"ACUMULACION":1.0,"DISTRIBUCION":0.5,"MARKDOWN":0.0,"UNKNOWN":0.75}
    n_s3 = result["n_tech"] * result["regime_lag"].map(mult_map).fillna(0.75)
    ax3.fill_between(dates, n_s3, color="#27ae60", alpha=0.4, label="S3 Tech+Regimen (efectivo)")
    ax3.plot(dates, n_s3, color="#27ae60", lw=0.8)
    ax3.set_title("Stocks activos por dia", fontweight="bold")
    ax3.set_ylabel("N stocks")
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.2)

    # ── 4. Tabla de métricas ───────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")

    rows_data = []
    for m in metrics_list:
        rows_data.append([
            m["label"],
            f"{m['total']*100:+.1f}%",
            f"{m['cagr']*100:.1f}%",
            f"{m['vol']*100:.1f}%",
            f"{m['sharpe']:.2f}",
            f"{m['sortino']:.2f}",
            f"{m['max_dd']*100:.1f}%",
            f"{m['calmar']:.2f}",
            f"{m['pct_inv']*100:.0f}%",
        ])

    col_labels = ["Estrategia", "Total", "CAGR", "Vol",
                  "Sharpe", "Sortino", "Max DD", "Calmar", "% Inv"]
    tbl = ax4.table(cellText=rows_data, colLabels=col_labels,
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.2, 1.9)

    strat_colors = [v["color"] for v in STRAT_META.values()]
    for j in range(len(col_labels)):
        tbl[0, j].set_facecolor("#1F497D")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for i, color in enumerate(strat_colors, 1):
        tbl[i, 0].set_facecolor(color + "40")
        tbl[i, 0].set_text_props(fontweight="bold")

    # Highlight best value per column (Sharpe=4, Sortino=5, Max DD=6, Calmar=7)
    for col_idx, key, higher_better in [
        (4, "sharpe", True), (5, "sortino", True),
        (6, "max_dd", False), (7, "calmar", True)
    ]:
        vals = [m[key] for m in metrics_list]
        best = max(vals) if higher_better else min(vals)
        for row_i, v in enumerate(vals, 1):
            if v == best:
                tbl[row_i, col_idx].set_facecolor("#C6EFCE")
                tbl[row_i, col_idx].set_text_props(fontweight="bold")

    ax4.set_title("Comparacion de Estrategias — verde = mejor en esa metrica",
                  fontweight="bold", pad=14)

    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / "tech_regime_results.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafica guardada: {path}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 58)
    print("  TECH + REGIME COMBINED BACKTEST")
    print("  SAR + EMA200 + RSI  x  Regimen de Mercado")
    print("=" * 58)

    df = load_data()
    print(f"\nDatos: {df['Company'].nunique()} tickers  "
          f"{df['Date'].min().date()} a {df['Date'].max().date()}")

    # Señales técnicas
    signals = compute_all_signals(df)

    # Régimen
    print("\nDetectando regimen de mercado...")
    regime = detect_regimes(df)

    # Backtest
    print("\nEjecutando backtest combinado...")
    result = run_backtest(signals, regime)

    print(f"\nPeriodo analizado: {result['Date'].iloc[0].date()} "
          f"a {result['Date'].iloc[-1].date()}")

    # Métricas
    strat_cols = [
        ("s0",      "S0  Benchmark EW"),
        ("s1_full", "S1  Tech solo"),
        ("s2",      "S2  Regimen solo"),
        ("s3",      "S3  Tech + Regimen"),
    ]
    print(f"\n{'='*58}")
    print("  RESULTADOS")
    print(f"{'='*58}")
    metrics_list = []
    for col, label in strat_cols:
        m = calc_metrics(result[col], label)
        metrics_list.append(m)
        print_metrics(m)

    # Conclusión
    m0, m1, m2, m3 = metrics_list
    print(f"\n{'='*58}")
    print("  CONCLUSION")
    print(f"{'='*58}")
    print(f"\n  Sharpe:   S0={m0['sharpe']:.2f}  S1={m1['sharpe']:.2f}  "
          f"S2={m2['sharpe']:.2f}  S3={m3['sharpe']:.2f}")
    print(f"  Max DD:   S0={m0['max_dd']*100:.1f}%  S1={m1['max_dd']*100:.1f}%  "
          f"S2={m2['max_dd']*100:.1f}%  S3={m3['max_dd']*100:.1f}%")
    print(f"\n  Mejora de Sharpe (S3 vs S1):  {(m3['sharpe']-m1['sharpe']):+.2f}")
    print(f"  Mejora de Max DD (S3 vs S1):  {(m3['max_dd']-m1['max_dd'])*100:+.1f}pp")
    print(f"  Mejora de Sharpe (S3 vs S0):  {(m3['sharpe']-m0['sharpe']):+.2f}")

    plot_results(result, metrics_list)
