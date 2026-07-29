"""
Sector Rotation por Regimen + Estado Conductual
=================================================
Pregunta: dado el regimen de mercado (4 cuadrantes) y el estado conductual
(PANIC/BUBBLE/HERD/NORMAL) de HOY, ¿que sector ha tendido a liderar en los
siguientes 20 dias habiles?

A diferencia de los proyectos anteriores (todos bottom-up, stock-picking),
esta es una vista top-down: el regimen y el estado conductual son conceptos
de mercado, no de accion individual — aplicarlos a los 11 SPDR sector ETFs
responde una pregunta que el resto del trabajo no cubre.

Universo: XLK XLF XLE XLV XLY XLP XLI XLB XLU XLRE XLC  (SPDR Select Sector)
Benchmark: SPY

Metodologia:
  1. Regimen (MARKUP/DISTRIBUCION/ACUMULACION/MARKDOWN) y estado conductual
     (PANIC/BUBBLE/HERD/NORMAL) se calculan SOLO sobre SPY — son conceptos
     de mercado agregado, no deben calcularse por sector.
  2. Para cada sector, retorno relativo forward de 20 dias habiles vs SPY,
     alineado a la fecha t (usa precios t..t+20, no hay look-ahead porque
     el regimen/estado de t ya esta lageado 1 dia como en el resto del repo).
  3. Se agrupa por regimen y por estado conductual (por separado, no cruzado,
     para no fragmentar la muestra) y se reporta la media + el numero de
     EPISODIOS independientes (bloques contiguos), no de dias — las ventanas
     de 20 dias se traslapan mucho y contar dias infla la significancia.

Limitacion explicita: PANIC ocurrio una sola vez en la muestra (COVID
2020-03). Cualquier conclusion sobre PANIC es anecdota de un evento, no
un patron estadistico. Se reporta igual, marcado como tal.
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from regime_classifier import classify_market_regime

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

BENCHMARK = "SPY"
SECTORS = {
    "XLK": "Technology", "XLF": "Financials", "XLE": "Energy",
    "XLV": "Health Care", "XLY": "Cons. Discretionary", "XLP": "Cons. Staples",
    "XLI": "Industrials", "XLB": "Materials", "XLU": "Utilities",
    "XLRE": "Real Estate", "XLC": "Comm. Services",
}

TREND_WINDOW    = 60
FEAR_PERCENTILE = 0.65
FORWARD_WINDOW  = 20   # dias habiles (~1 mes)

REGIME_COLORS = {
    "MARKUP": "#2980b9", "DISTRIBUCION": "#e67e22",
    "ACUMULACION": "#27ae60", "MARKDOWN": "#c0392b", "UNKNOWN": "#95a5a6",
}
BEHAVIORAL_COLORS = {
    "PANIC": "#f85149", "BUBBLE": "#d29922", "HERD": "#e3b341", "NORMAL": "#8b949e",
}


def download_prices() -> pd.DataFrame:
    tickers = [BENCHMARK] + list(SECTORS.keys())
    print(f"Descargando {len(tickers)} tickers (10y, yfinance)...")
    raw = yf.download(tickers, period="10y", auto_adjust=True, progress=False)["Close"]
    raw.index = pd.to_datetime(raw.index).tz_localize(None)
    raw = raw.dropna(how="all").ffill()
    print(f"  Rango: {raw.index[0].date()} a {raw.index[-1].date()}  |  {raw.shape[0]} dias")
    for t in tickers:
        n_valid = raw[t].dropna().shape[0]
        print(f"    {t:<6} {n_valid} dias validos")
    return raw


def detect_regime(spy_close: pd.Series) -> pd.Series:
    """Mismo esquema de 4 cuadrantes que tech_regime_backtest.py / live_dashboard.py,
    aplicado sobre SPY en vez del promedio EW de 491 stocks."""
    ret = spy_close.pct_change()
    vol_20 = ret.rolling(20).std() * np.sqrt(252)
    vol_60 = ret.rolling(60).std() * np.sqrt(252)
    fear_lv  = vol_20 / vol_60.replace(0, np.nan)
    fear_pct = fear_lv.expanding().rank(pct=True)
    mom_60   = ret.rolling(TREND_WINDOW).mean()

    pos_trend   = mom_60 > 0
    high_stress = fear_pct >= FEAR_PERCENTILE

    regime = pd.Series("UNKNOWN", index=spy_close.index)
    regime[ pos_trend & ~high_stress] = "MARKUP"
    regime[ pos_trend &  high_stress] = "DISTRIBUCION"
    regime[~pos_trend & ~high_stress] = "ACUMULACION"
    regime[~pos_trend &  high_stress] = "MARKDOWN"
    return regime


def compute_forward_excess(prices: pd.DataFrame) -> pd.DataFrame:
    """Retorno relativo forward de FORWARD_WINDOW dias de cada sector vs SPY."""
    fwd = prices.shift(-FORWARD_WINDOW) / prices - 1
    excess = fwd.drop(columns=[BENCHMARK]).sub(fwd[BENCHMARK], axis=0)
    return excess


def count_episodes(labels: pd.Series, mask: pd.Series) -> int:
    """Cuenta bloques contiguos (episodios), no dias — para no inflar la
    muestra con ventanas de 20d traslapadas dentro del mismo episodio."""
    active = mask.reindex(labels.index).fillna(False)
    return int(((active) & (~active.shift(1, fill_value=False))).sum())


def build_heatmap_data(excess: pd.DataFrame, labels_lag: pd.Series,
                       categories: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Media de retorno excedente forward por sector x categoria, y conteo
    de episodios independientes por categoria."""
    means = pd.DataFrame(index=excess.columns, columns=categories, dtype=float)
    episodes = pd.Series(index=categories, dtype=int)
    for cat in categories:
        mask = (labels_lag == cat)
        episodes[cat] = count_episodes(labels_lag, mask)
        sub = excess[mask.reindex(excess.index).fillna(False)]
        means[cat] = sub.mean()
    return means, episodes


def plot_heatmap(ax, data: pd.DataFrame, episodes: pd.Series, title: str,
                 sector_labels: dict):
    arr = data.values * 100  # a %
    vmax = np.nanmax(np.abs(arr))
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(data.columns)))
    ax.set_xticklabels([f"{c}\n(n={int(episodes[c])} ep.)" for c in data.columns], fontsize=8)
    ax.set_yticks(range(len(data.index)))
    ax.set_yticklabels([f"{t} {sector_labels.get(t,'')}" for t in data.index], fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:+.1f}%", ha="center", va="center", fontsize=7.5,
                        color="black")
    ax.set_title(title, fontweight="bold", fontsize=10)
    return im


def main():
    prices = download_prices()
    spy = prices[BENCHMARK]

    print("\nCalculando regimen de mercado (SPY)...")
    regime = detect_regime(spy)
    print(regime.value_counts())

    print("\nClasificando estado conductual (SPY)...")
    feats = classify_market_regime(spy.pct_change())
    behavioral = feats["state"]
    print(behavioral.value_counts())

    print(f"\nCalculando retorno relativo forward {FORWARD_WINDOW}d vs SPY...")
    excess = compute_forward_excess(prices)

    # Lag 1 dia — consistente con el resto del repo (sin look-ahead)
    regime_lag     = regime.shift(1)
    behavioral_lag = behavioral.shift(1)

    regime_cats     = ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]
    behavioral_cats = ["PANIC", "BUBBLE", "HERD", "NORMAL"]

    means_regime, ep_regime = build_heatmap_data(excess, regime_lag, regime_cats)
    means_behav,  ep_behav  = build_heatmap_data(excess, behavioral_lag, behavioral_cats)

    # Un sector puede liderar "siempre" (ventaja secular, p.ej. Tech en la
    # ultima decada) sin que eso sea informacion de regimen. Se resta el
    # promedio incondicional (todo el periodo) para aislar el efecto que
    # SI depende del regimen/estado — el "lift" condicional real.
    unconditional_mean = excess.mean()
    means_regime_adj = means_regime.sub(unconditional_mean, axis=0)
    means_behav_adj  = means_behav.sub(unconditional_mean, axis=0)

    print(f"\n{'='*66}\n  BASELINE INCONDICIONAL (todo el periodo, sin condicionar a regimen)\n{'='*66}")
    for t in unconditional_mean.sort_values(ascending=False).index:
        print(f"    {t:<6} {SECTORS.get(t,''):<22} {unconditional_mean[t]*100:+.2f}%")
    print("\n  Las tablas 'ajustadas' de abajo restan esta baseline — muestran solo")
    print("  el efecto que SI depende del regimen/estado, no la ventaja secular.")

    print(f"\n{'='*66}\n  LIFT CONDICIONAL (ajustado por baseline) — POR REGIMEN\n{'='*66}")
    print(means_regime_adj.mul(100).round(2).to_string())
    print(f"\n{'='*66}\n  LIFT CONDICIONAL (ajustado por baseline) — POR ESTADO CONDUCTUAL\n{'='*66}")
    print(means_behav_adj.mul(100).round(2).to_string())

    print(f"\n{'='*66}\n  EPISODIOS INDEPENDIENTES (bloques contiguos, no dias)\n{'='*66}")
    print("  Regimen:    " + "  ".join(f"{c}={ep_regime[c]}" for c in regime_cats))
    print("  Conductual: " + "  ".join(f"{c}={ep_behav[c]}" for c in behavioral_cats))
    if ep_behav.get("PANIC", 0) <= 1:
        print("\n  [!] PANIC tiene <=1 episodio independiente en la muestra — "
              "cualquier numero para esa columna es anecdota de un solo evento, no patron.")

    # ── Snapshot actual ──────────────────────────────────────────────────
    regime_now     = regime.iloc[-1]
    behavioral_now = behavioral.iloc[-1]
    print(f"\n{'='*66}\n  SNAPSHOT ACTUAL\n{'='*66}")
    print(f"  Regimen:    {regime_now}")
    print(f"  Conductual: {behavioral_now}")

    if regime_now in means_regime_adj.columns:
        top_regime = means_regime_adj[regime_now].dropna().sort_values(ascending=False).head(3)
        print(f"\n  Top 3 sectores (lift condicional) en regimen {regime_now}  "
              f"[n={ep_regime[regime_now]:.0f} episodios]:")
        for t, v in top_regime.items():
            print(f"    {t:<6} {SECTORS.get(t,''):<22} {v*100:+.2f}pp lift vs baseline propio")
    if behavioral_now in means_behav_adj.columns:
        top_behav = means_behav_adj[behavioral_now].dropna().sort_values(ascending=False).head(3)
        print(f"\n  Top 3 sectores (lift condicional) en estado conductual {behavioral_now}  "
              f"[n={ep_behav[behavioral_now]:.0f} episodios]:")
        for t, v in top_behav.items():
            print(f"    {t:<6} {SECTORS.get(t,''):<22} {v*100:+.2f}pp lift vs baseline propio")

    # ── Plot ─────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle(
        f"SECTOR ROTATION — Lift Condicional vs Baseline Propio (forward {FORWARD_WINDOW}d vs SPY)\n"
        f"por Regimen de Mercado y Estado Conductual  |  Snapshot: {regime_now} / {behavioral_now}",
        fontsize=12, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.55)

    ax1 = fig.add_subplot(gs[0, 0])
    plot_heatmap(ax1, means_regime_adj, ep_regime, "Por Regimen (4 cuadrantes)", SECTORS)

    ax2 = fig.add_subplot(gs[0, 1])
    plot_heatmap(ax2, means_behav_adj, ep_behav, "Por Estado Conductual", SECTORS)

    fig.text(0.5, 0.01,
             "Valores = retorno excedente vs SPY MENOS el promedio incondicional propio del sector "
             "(aisla el efecto de regimen, no la ventaja secular tipo 'Tech siempre gana'). "
             "n = episodios independientes, no dias. PANIC con 4 episodios es indicativo, no concluyente.",
             ha="center", fontsize=8, style="italic", color="#555")

    path = OUT_DIR / "sector_rotation.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafica guardada: {path}")


if __name__ == "__main__":
    main()
