"""
Unified Backtester v2 — Regime-Switching Value + Tech (MEJORADO)
================================================================
Mejoras sobre v1:
  - Filtro de calidad FMP: ROIC, margenes, deuda — elimina value traps
  - P/E historico percentil (ratios_annual.parquet, 2022+) — senial mas precisa
  - Value signal = stock de calidad en valuacion historicamente baja
  - Fallback a 52w-proximity solo en stocks que pasan calidad

Estrategias comparadas:
  S0  Benchmark EW          — buy & hold 491 S&P 500
  S1  Tech puro             — SAR + EMA200 + RSI sin regimen
  S2  Value v1 (proxy)      — 52w-low + momentum inverso (version anterior)
  S3  Value v2 (calidad)    — quality screen + P/E percentil historico [NUEVO]
  S4  UNIFIED v2            — tech en MARKUP, value-calidad en ACUMULACION

Dataset: Datos/stock_details_5_years.csv  (491 S&P 500, OHLCV 2018-2023)
FMP:     Datos/fundamentals/              (quality screen + PE historico)
"""

import warnings
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")   # no GUI — guardar en archivo

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from ta.trend import EMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator

# ── Config ─────────────────────────────────────────────────────────────────────

DATA_PATH  = Path("../Datos/stock_details_5_years.csv")
FMP_DIR    = Path("../Datos/fundamentals")
OUT_DIR    = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

PSAR_STEP  = 0.02
PSAR_MAX   = 0.20
EMA_WIN    = 200
RSI_WIN    = 14
RSI_LO, RSI_HI = 40, 75
MIN_ROWS   = 260

TREND_WIN      = 60
FEAR_PCT_THOLD = 0.65

VALUE_LOOKBACK_52W = 252
VALUE_MOM_WINDOW   = 252
TOP_VALUE_PCT      = 0.20

# Filtros de calidad FMP
QUALITY_ROIC_MIN   = 0.05   # 5% ROIC minimo
QUALITY_MARGIN_MIN = 0.02   # 2% margen neto minimo
QUALITY_DE_MAX     = 3.0    # deuda/equity maximo
QUALITY_CR_MIN     = 1.0    # current ratio minimo

# P/E percentil: cuanto peso darle cuando hay historia anual
PE_HIST_WEIGHT     = 0.60   # 60% PE percentil, 40% 52w-proximity
PE_CHEAP_PCT       = 0.40   # stocks en los 40% mas baratos de su historia

RF_ANNUAL = 0.05


# ── Helpers ────────────────────────────────────────────────────────────────────

def fval(d, key, default=0.0):
    v = d.get(key, default) if isinstance(d, dict) else getattr(d, key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


# ── 1. Cargar datos ────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_convert(None)
    df = df.sort_values(["Company", "Date"]).reset_index(drop=True)
    return df


# ── 2. Filtro de calidad FMP (quality screen) ──────────────────────────────────

def build_quality_set() -> set:
    """
    Retorna set de tickers que pasan el filtro de calidad FMP TTM.
    ROIC > 5%, net margin > 2%, D/E < 3x, current ratio > 1.
    Si no hay datos FMP, retorna set vacio (no se aplica filtro).
    """
    km_path = FMP_DIR / "key_metrics_ttm.parquet"
    rt_path = FMP_DIR / "ratios_ttm.parquet"

    if not km_path.exists() or not rt_path.exists():
        print("  [quality] No se encontraron datos FMP TTM — sin filtro de calidad")
        return set()

    km = pd.read_parquet(km_path)
    rt = pd.read_parquet(rt_path)

    km_idx = km.set_index("symbol")
    rt_idx = rt.set_index("symbol")

    quality = set()
    all_syms = set(km_idx.index) | set(rt_idx.index)

    for sym in all_syms:
        try:
            k = km_idx.loc[sym] if sym in km_idx.index else {}
            r = rt_idx.loc[sym] if sym in rt_idx.index else {}

            roic = fval(k, "returnOnInvestedCapitalTTM") if hasattr(k, "get") else (
                float(k["returnOnInvestedCapitalTTM"]) if "returnOnInvestedCapitalTTM" in k else 0.0)
            curr = fval(k, "currentRatioTTM") if hasattr(k, "get") else (
                float(k["currentRatioTTM"]) if "currentRatioTTM" in k else 0.0)
            nm   = fval(r, "netProfitMarginTTM") if hasattr(r, "get") else (
                float(r["netProfitMarginTTM"]) if "netProfitMarginTTM" in r else 0.0)
            deq  = fval(r, "debtToEquityRatioTTM") if hasattr(r, "get") else (
                float(r["debtToEquityRatioTTM"]) if "debtToEquityRatioTTM" in r else 99.0)

            if (roic >= QUALITY_ROIC_MIN and
                nm   >= QUALITY_MARGIN_MIN and
                deq  <= QUALITY_DE_MAX and
                curr >= QUALITY_CR_MIN):
                quality.add(sym)
        except Exception:
            continue

    return quality


def build_quality_set_safe(km: pd.DataFrame, rt: pd.DataFrame) -> set:
    """Version rapida usando pandas directamente."""
    quality = set()

    def to_float(series):
        return pd.to_numeric(series, errors="coerce").fillna(0.0)

    km2 = km.copy()
    rt2 = rt.copy()

    for col in ["returnOnInvestedCapitalTTM", "currentRatioTTM"]:
        if col in km2.columns:
            km2[col] = to_float(km2[col])
    for col in ["netProfitMarginTTM", "debtToEquityRatioTTM"]:
        if col in rt2.columns:
            rt2[col] = to_float(rt2[col])

    km_filt = km2.copy()
    if "returnOnInvestedCapitalTTM" in km2.columns:
        km_filt = km_filt[km_filt["returnOnInvestedCapitalTTM"] >= QUALITY_ROIC_MIN]
    if "currentRatioTTM" in km2.columns:
        km_filt = km_filt[km_filt["currentRatioTTM"] >= QUALITY_CR_MIN]
    quality_km = set(km_filt["symbol"].tolist())

    rt_filt = rt2.copy()
    if "netProfitMarginTTM" in rt2.columns:
        rt_filt = rt_filt[rt_filt["netProfitMarginTTM"] >= QUALITY_MARGIN_MIN]
    if "debtToEquityRatioTTM" in rt2.columns:
        rt_filt = rt_filt[rt_filt["debtToEquityRatioTTM"] <= QUALITY_DE_MAX]
    quality_rt = set(rt_filt["symbol"].tolist())

    # Intersection: must pass BOTH km and rt filters
    quality = quality_km & quality_rt
    return quality


# ── 3. P/E historico percentil (de ratios_annual) ─────────────────────────────

def compute_pe_percentile_series(tickers: list,
                                 dates: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Para cada (ticker, fecha): calcula percentil de P/E vs su propia historia anual.
    Percentil = fraccion de anos anteriores con P/E MENOR al actual.
    0 = mas barato que siempre, 1 = mas caro que siempre.
    Retorna DataFrame wide (filas=fechas, cols=tickers). NaN = sin historia.
    """
    ann_path = FMP_DIR / "ratios_annual.parquet"
    if not ann_path.exists():
        return pd.DataFrame(np.nan, index=dates, columns=tickers)

    ann = pd.read_parquet(ann_path)
    ann["date"] = pd.to_datetime(ann["date"], errors="coerce")
    ann = ann.dropna(subset=["date"])

    def safe_pe(v):
        try:
            f = float(v)
            return f if f > 0 else np.nan
        except:
            return np.nan

    ann["pe"] = ann["priceToEarningsRatio"].apply(safe_pe)
    ann = ann.dropna(subset=["pe"])

    # Lag: datos disponibles 90 dias despues de cierre fiscal
    ann["avail_date"] = ann["date"] + pd.DateOffset(days=90)

    tickers_set = set(tickers)
    ann = ann[ann["symbol"].isin(tickers_set)]

    result_dict = {}

    for sym, grp in ann.groupby("symbol"):
        grp = grp.sort_values("avail_date").reset_index(drop=True)
        if len(grp) < 2:
            continue

        # Construir serie de percentiles en cada fecha de reporte disponible
        checkpoints = {}
        for i in range(1, len(grp)):
            avail_date = grp.iloc[i]["avail_date"]
            prior_pes  = grp.iloc[:i]["pe"].values    # solo los anteriores
            curr_pe    = grp.iloc[i]["pe"]
            # Percentil: cuantos anos anteriores tenian P/E menor que el actual
            pct        = (prior_pes < curr_pe).mean()
            checkpoints[avail_date] = pct

        # Forward-fill sobre el indice de fechas de backtest
        sparse = pd.Series(checkpoints, dtype=float)
        full   = sparse.reindex(dates.union(sparse.index)).sort_index()
        full   = full.ffill().reindex(dates)
        result_dict[sym] = full

    df_pe = pd.DataFrame(result_dict, index=dates)

    # Solo tickers que estan en backtest
    common = [t for t in tickers if t in df_pe.columns]
    df_pe  = df_pe.reindex(columns=tickers)
    return df_pe


# ── 4. Senales tecnicas ────────────────────────────────────────────────────────

def compute_tech_signals(df: pd.DataFrame) -> pd.DataFrame:
    tickers = df["Company"].unique()
    parts   = []
    skip    = 0
    print(f"Calculando senales tecnicas ({len(tickers)} tickers)...")

    for ticker in tickers:
        tk = df[df["Company"] == ticker].copy().reset_index(drop=True)
        if len(tk) < MIN_ROWS:
            skip += 1
            continue
        try:
            close  = tk["Close"]
            high   = tk["High"]
            low    = tk["Low"]
            psar   = PSARIndicator(high=high, low=low, close=close,
                                   step=PSAR_STEP, max_step=PSAR_MAX).psar()
            ema200 = EMAIndicator(close=close, window=EMA_WIN).ema_indicator()
            rsi    = RSIIndicator(close=close, window=RSI_WIN).rsi()

            sig = ((close > psar) & (close > ema200) &
                   (rsi >= RSI_LO) & (rsi <= RSI_HI)).astype(int)

            out = tk[["Date", "Close"]].copy()
            out["ret"]      = close.pct_change()
            out["tech_sig"] = sig.values
            out["ticker"]   = ticker
            parts.append(out)
        except Exception:
            skip += 1

    print(f"  Completado. Omitidos: {skip}")
    return pd.concat(parts, ignore_index=True)


# ── 5. Value score v1 (proxy — para comparacion) ──────────────────────────────

def compute_value_scores_v1(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Score value ORIGINAL (proxy):
    60% cercania a minimo 52 semanas + 40% momentum inverso.
    Sin filtro de calidad — puede elegir value traps.
    """
    print("Calculando scores value v1 (proxy, sin calidad)...")
    wide   = signals.pivot(index="Date", columns="ticker", values="Close").sort_index()
    low52  = wide.rolling(VALUE_LOOKBACK_52W, min_periods=126).min()
    high52 = wide.rolling(VALUE_LOOKBACK_52W, min_periods=126).max()
    prox   = (high52 - wide) / (high52 - low52 + 1e-9)

    mom12  = wide.pct_change(VALUE_MOM_WINDOW)
    raw    = 0.60 * prox + 0.40 * (-mom12)

    def rank_row(row):
        valid = row.dropna()
        if len(valid) < 10:
            return row * np.nan
        return row.rank(pct=True)

    value_rank = raw.apply(rank_row, axis=1)
    value_sig  = (value_rank >= (1 - TOP_VALUE_PCT)).astype(float)
    value_sig[value_rank.isna()] = np.nan

    vs = value_sig.stack().reset_index()
    vs.columns = ["Date", "ticker", "value_sig"]
    print("  v1 completado.")
    return vs


# ── 6. Value score v2 (calidad + P/E historico) — NUEVO ───────────────────────

def compute_value_scores_v2(signals: pd.DataFrame) -> pd.DataFrame:
    """
    Score value MEJORADO:
    1. Filtro de calidad: solo stocks con ROIC > 5%, margen > 2%, D/E < 3x
    2. P/E historico percentil (donde disponible): stocks en el 40% mas barato
    3. Fallback: 52w-low proximity (solo en stocks que pasan calidad)
    Elimina value traps por definicion — un stock de calidad barato es oportunidad.
    """
    print("Calculando scores value v2 (calidad + P/E historico)...")

    # 1. Quality screen
    km_path = FMP_DIR / "key_metrics_ttm.parquet"
    rt_path = FMP_DIR / "ratios_ttm.parquet"
    quality = set()

    if km_path.exists() and rt_path.exists():
        km = pd.read_parquet(km_path)
        rt = pd.read_parquet(rt_path)
        quality = build_quality_set_safe(km, rt)
        print(f"  Quality screen: {len(quality)} / {len(km)} tickers pasan filtro FMP")
    else:
        print("  Sin datos FMP — aplicando score sin filtro de calidad")

    wide   = signals.pivot(index="Date", columns="ticker", values="Close").sort_index()
    tickers = list(wide.columns)
    dates   = wide.index

    # 2. P/E historico percentil (vectorizado)
    print("  Cargando P/E historico anual...")
    pe_pct_df = compute_pe_percentile_series(tickers, dates)
    n_with_pe = pe_pct_df.notna().any(axis=0).sum()
    print(f"  P/E historico disponible para {n_with_pe} tickers")

    # 3. 52-week proximity (base, solo para quality-filtered stocks)
    low52  = wide.rolling(VALUE_LOOKBACK_52W, min_periods=126).min()
    high52 = wide.rolling(VALUE_LOOKBACK_52W, min_periods=126).max()
    prox   = (high52 - wide) / (high52 - low52 + 1e-9)

    # 4. Combinar: score = f(calidad, P/E percentil, prox)
    # Para stocks SIN calidad: score = 0 (excluidos)
    # Para stocks CON calidad, SIN P/E hist: score = prox (mismo que v1 pero filtrado)
    # Para stocks CON calidad, CON P/E hist: score = 0.6 * (1 - pe_pct) + 0.4 * prox

    # Mascara de calidad
    quality_mask = pd.DataFrame(False, index=dates, columns=tickers)
    for t in tickers:
        if t in quality:
            quality_mask[t] = True

    # Score combinado
    pe_component  = 1.0 - pe_pct_df                # 1=mas barato historicamente
    has_pe        = pe_pct_df.notna()

    score_with_pe   = PE_HIST_WEIGHT * pe_component + (1 - PE_HIST_WEIGHT) * prox
    score_no_pe     = prox                          # solo 52w proximity

    score = score_with_pe.where(has_pe, other=score_no_pe)
    score = score.where(quality_mask, other=0.0)

    # 5. Rank cross-seccional: top TOP_VALUE_PCT del score (de stocks con calidad)
    def rank_row(row):
        valid = row[row > 0].dropna()
        if len(valid) < 5:
            return row * np.nan
        ranked = row.rank(pct=True)
        ranked[row <= 0] = np.nan
        return ranked

    value_rank = score.apply(rank_row, axis=1)
    value_sig  = (value_rank >= (1 - TOP_VALUE_PCT)).astype(float)
    value_sig[value_rank.isna()] = np.nan
    value_sig[quality_mask == False] = 0.0

    vs = value_sig.stack().reset_index()
    vs.columns = ["Date", "ticker", "value_sig"]
    print("  v2 completado.")
    return vs


# ── 7. Regimen de mercado ──────────────────────────────────────────────────────

def compute_regime(signals: pd.DataFrame) -> pd.Series:
    print("Calculando regimen de mercado...")
    wide     = signals.pivot(index="Date", columns="ticker", values="Close")
    mkt_ret  = wide.mean(axis=1).pct_change()
    vol_20   = mkt_ret.rolling(20).std() * np.sqrt(252)
    vol_60   = mkt_ret.rolling(60).std() * np.sqrt(252)
    fear_lv  = vol_20 / vol_60.replace(0, np.nan)
    fear_pct = fear_lv.expanding().rank(pct=True)
    mom_60   = mkt_ret.rolling(TREND_WIN).mean()

    pos_trend   = mom_60 > 0
    high_stress = fear_pct >= FEAR_PCT_THOLD

    r = pd.Series("UNKNOWN", index=wide.index)
    r[ pos_trend & ~high_stress] = "MARKUP"
    r[ pos_trend &  high_stress] = "DISTRIBUCION"
    r[~pos_trend & ~high_stress] = "ACUMULACION"
    r[~pos_trend &  high_stress] = "MARKDOWN"
    print(f"  Regimen calculado. Distribucion:")
    for reg in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]:
        n = (r == reg).sum()
        print(f"    {reg:<14} {n:>4} dias")
    return r


# ── 8. Backtest engine ─────────────────────────────────────────────────────────

def run_backtest(signals: pd.DataFrame,
                 values_v1: pd.DataFrame,
                 values_v2: pd.DataFrame,
                 regime: pd.Series) -> pd.DataFrame:
    print("\nEjecutando backtest...")

    merged = signals.merge(values_v1.rename(columns={"value_sig": "v1_sig"}),
                           on=["Date", "ticker"], how="left")
    merged = merged.merge(values_v2.rename(columns={"value_sig": "v2_sig"}),
                          on=["Date", "ticker"], how="left")
    merged = merged.sort_values(["ticker", "Date"])

    merged["tech_lag"] = merged.groupby("ticker")["tech_sig"].shift(1)
    merged["v1_lag"]   = merged.groupby("ticker")["v1_sig"].shift(1)
    merged["v2_lag"]   = merged.groupby("ticker")["v2_sig"].shift(1)

    regime_lag = regime.shift(1).rename("regime")
    merged = merged.merge(
        regime_lag.reset_index().rename(columns={"index": "Date"}),
        on="Date", how="left"
    )

    merged = merged[merged["Date"] >= "2019-06-01"]

    # S0: Benchmark EW
    bench = merged.groupby("Date")["ret"].mean().rename("bench_ret")

    # S1: Tech puro
    in_tech = merged[merged["tech_lag"] == 1]
    s1 = in_tech.groupby("Date")["ret"].mean().rename("s1_ret")

    # S2: Value v1 (proxy)
    in_v1 = merged[merged["v1_lag"] == 1]
    s2 = in_v1.groupby("Date")["ret"].mean().rename("s2_ret")

    # S3: Value v2 (calidad + PE historico)
    in_v2 = merged[merged["v2_lag"] == 1]
    s3 = in_v2.groupby("Date")["ret"].mean().rename("s3_ret")

    # S4: UNIFIED v2 — tech en MARKUP, value-calidad en ACUMULACION
    def unified_in_v2(row):
        r = row["regime"]
        if r == "MARKUP":
            return row["tech_lag"] == 1
        elif r == "ACUMULACION":
            return row["v2_lag"] == 1
        elif r == "DISTRIBUCION":
            return (row["tech_lag"] == 1) or (row["v2_lag"] == 1)
        else:
            return False

    merged["unified_in"] = merged.apply(unified_in_v2, axis=1)
    merged["unified_wt"] = 1.0
    merged.loc[merged["regime"] == "DISTRIBUCION", "unified_wt"] = 0.5

    in_uni = merged[merged["unified_in"]]
    s4 = (in_uni.groupby("Date")
          .apply(lambda g: np.average(g["ret"], weights=g["unified_wt"]))
          .rename("s4_ret"))

    result = (bench.to_frame()
              .join(s1, how="left")
              .join(s2, how="left")
              .join(s3, how="left")
              .join(s4, how="left"))
    result = result.fillna(0)
    result.index = pd.to_datetime(result.index)

    for col, eq_col in [("bench_ret","eq_bench"),("s1_ret","eq_s1"),("s2_ret","eq_s2"),
                        ("s3_ret","eq_s3"),("s4_ret","eq_s4")]:
        result[eq_col] = (1 + result[col]).cumprod()

    result = result.join(regime.rename("regime"), how="left")
    print(f"  Backtest completo. {len(result)} dias "
          f"({result.index[0].date()} a {result.index[-1].date()})")
    return result


# ── 9. Metricas ────────────────────────────────────────────────────────────────

def calc_metrics(returns: pd.Series, label: str) -> dict:
    r = returns.dropna().replace([np.inf, -np.inf], np.nan).dropna()
    r = r[r != 0] if (r == 0).mean() > 0.5 else r
    if len(r) == 0:
        return dict(label=label, cagr=0, sharpe=0, max_dd=0, sortino=0, calmar=0,
                    vol=0, total=0)

    n_years  = len(returns) / 252
    r_all    = returns.fillna(0)
    total    = (1 + r_all).prod() - 1
    cagr     = (1 + total) ** (1 / n_years) - 1
    vol      = r_all.std() * np.sqrt(252)
    downside = r_all[r_all < 0].std() * np.sqrt(252)
    sharpe   = (cagr - RF_ANNUAL) / vol      if vol      > 0 else 0
    sortino  = (cagr - RF_ANNUAL) / downside if downside > 0 else 0
    eq       = (1 + r_all).cumprod()
    max_dd   = ((eq - eq.cummax()) / eq.cummax()).min()
    calmar   = cagr / abs(max_dd) if max_dd != 0 else 0

    return dict(label=label, total=total, cagr=cagr, vol=vol,
                sharpe=sharpe, sortino=sortino, max_dd=max_dd, calmar=calmar)


def print_metrics_table(metrics: list):
    print(f"\n{'='*75}")
    print(f"  {'Estrategia':<26} {'CAGR':>7} {'Vol':>7} {'Sharpe':>7} "
          f"{'MaxDD':>8} {'Sortino':>8} {'Calmar':>7}")
    print(f"  {'-'*26} {'-'*7} {'-'*7} {'-'*7} {'-'*8} {'-'*8} {'-'*7}")
    for m in metrics:
        marker = " <--" if "v2" in m["label"] or "UNIFIED" in m["label"] else ""
        print(f"  {m['label']:<26} "
              f"{m['cagr']*100:>6.1f}% "
              f"{m.get('vol',0)*100:>6.1f}% "
              f"{m['sharpe']:>7.2f} "
              f"{m['max_dd']*100:>7.1f}% "
              f"{m['sortino']:>8.2f} "
              f"{m['calmar']:>7.2f}{marker}")
    print(f"{'='*75}")


# ── 10. Visualizacion ─────────────────────────────────────────────────────────

COLORS = {
    "bench": "#95a5a6",
    "s1":    "#2980b9",
    "s2":    "#bdc3c7",   # v1 proxy (gris — old)
    "s3":    "#27ae60",   # v2 calidad (verde — nuevo)
    "s4":    "#8e44ad",   # unified v2 (violeta)
}
REGIME_COLORS = {
    "MARKUP":       "#2980b9",
    "DISTRIBUCION": "#e67e22",
    "ACUMULACION":  "#27ae60",
    "MARKDOWN":     "#c0392b",
}


def _regime_bg(ax, regime_series):
    prev, t0 = None, regime_series.index[0]
    for t, r in regime_series.items():
        if r != prev and prev is not None:
            ax.axvspan(t0, t, alpha=0.10,
                       color=REGIME_COLORS.get(prev, "#888"), lw=0)
            t0 = t
        prev = r
    ax.axvspan(t0, regime_series.index[-1], alpha=0.10,
               color=REGIME_COLORS.get(prev, "#888"), lw=0)


def plot_results(result: pd.DataFrame, metrics: list):
    fig = plt.figure(figsize=(16, 13))
    fig.suptitle(
        "Unified Backtester v2: Regime + Value con Calidad FMP\n"
        "S&P 500 - 491 tickers - 2019-2023  |  v2: quality screen + P/E historico",
        fontsize=12, fontweight="bold", y=0.99
    )
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.50, wspace=0.35)
    reg = result["regime"].fillna("UNKNOWN")

    labels = {
        "eq_bench": ("S0 Benchmark EW",          COLORS["bench"], "--",  1.2),
        "eq_s1":    ("S1 Tech puro",              COLORS["s1"],    "-.",  1.3),
        "eq_s2":    ("S2 Value v1 (proxy)",       COLORS["s2"],    ":",   1.1),
        "eq_s3":    ("S3 Value v2 (calidad+PE)",  COLORS["s3"],    "-",   1.5),
        "eq_s4":    ("S4 UNIFIED v2",             COLORS["s4"],    "-",   2.2),
    }

    # Panel 1: Curvas de capital
    ax1 = fig.add_subplot(gs[0, :])
    _regime_bg(ax1, reg)
    for col, (lbl, clr, ls, lw) in labels.items():
        ax1.plot(result.index, result[col], label=lbl, color=clr, ls=ls, lw=lw, alpha=0.9)
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.4, label=r)
               for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]]
    handles, lbls = ax1.get_legend_handles_labels()
    ax1.legend(handles=handles + patches, fontsize=7.5, ncol=3, loc="upper left")
    ax1.set_title("Curvas de Capital  (fondo = regimen)", fontweight="bold")
    ax1.set_ylabel("Capital ($1 inicial)")
    ax1.grid(alpha=0.2)
    ax1.axhline(1, color="gray", lw=0.7, ls=":")

    # Panel 2: Drawdown
    ax2 = fig.add_subplot(gs[1, 0])
    _regime_bg(ax2, reg)
    ret_cols = {"eq_bench": "bench_ret", "eq_s1": "s1_ret",
                "eq_s2": "s2_ret", "eq_s3": "s3_ret", "eq_s4": "s4_ret"}
    for col, (lbl, clr, ls, lw) in labels.items():
        r_col = ret_cols[col]
        eq = (1 + result[r_col].fillna(0)).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax() * 100
        if col in ("eq_s3", "eq_s4"):
            ax2.fill_between(result.index, dd, 0, alpha=0.18, color=clr)
        ax2.plot(result.index, dd, color=clr, ls=ls, lw=lw * 0.7, alpha=0.85, label=lbl)
    ax2.set_title("Drawdown (%)", fontweight="bold")
    ax2.set_ylabel("%")
    ax2.legend(fontsize=7, loc="lower left")
    ax2.grid(alpha=0.2)

    # Panel 3: Sharpe comparativo
    ax3 = fig.add_subplot(gs[1, 1])
    names  = [m["label"] for m in metrics]
    sharpe = [m["sharpe"] for m in metrics]
    clrs   = [COLORS["bench"], COLORS["s1"], COLORS["s2"], COLORS["s3"], COLORS["s4"]]
    bars   = ax3.barh(names, sharpe, color=clrs, alpha=0.8)
    for bar, v in zip(bars, sharpe):
        ax3.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                 f"{v:.2f}", va="center", fontsize=9)
    ax3.axvline(0, color="black", lw=0.8)
    ax3.set_title("Sharpe Ratio comparativo", fontweight="bold")
    ax3.grid(axis="x", alpha=0.25)

    # Panel 4: CAGR vs MaxDD
    ax4 = fig.add_subplot(gs[2, 0])
    for m, clr in zip(metrics, clrs):
        ax4.scatter(abs(m["max_dd"]) * 100, m["cagr"] * 100, color=clr, s=120, zorder=5)
        ax4.annotate(m["label"].split(" ")[0],
                     (abs(m["max_dd"]) * 100, m["cagr"] * 100),
                     textcoords="offset points", xytext=(5, 3), fontsize=8)
    ax4.set_xlabel("Max Drawdown (%) — menor es mejor -->")
    ax4.set_ylabel("CAGR (%) — mayor es mejor ^")
    ax4.set_title("CAGR vs Riesgo (esquina sup-izq = ideal)", fontweight="bold")
    ax4.grid(alpha=0.25)
    ax4.invert_xaxis()

    # Panel 5: Tabla resumen
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.axis("off")
    HEADER = "#1F497D"
    rows_data = [["Estrategia", "CAGR", "Sharpe", "MaxDD", "Sortino"]]
    for m in metrics:
        rows_data.append([
            m["label"],
            f"{m['cagr']*100:.1f}%",
            f"{m['sharpe']:.2f}",
            f"{m['max_dd']*100:.1f}%",
            f"{m['sortino']:.2f}",
        ])
    tbl = ax5.table(cellText=rows_data[1:], colLabels=rows_data[0],
                    loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.2, 1.8)
    for j in range(5):
        tbl[0, j].set_facecolor(HEADER)
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    for j in range(5):
        tbl[3, j].set_facecolor("#D5F5E3")  # verde suave — S3 value v2
    for j in range(5):
        tbl[4, j].set_facecolor("#E8D5FF")  # violeta — S4 unified v2

    out = OUT_DIR / "unified_backtest_v2_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nGrafica guardada: {out}")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 65)
    print("  UNIFIED BACKTESTER v2")
    print("  Quality screen FMP + P/E historico percentil")
    print("=" * 65)

    df       = load_data()
    signals  = compute_tech_signals(df)
    values1  = compute_value_scores_v1(signals)    # proxy (v1, para comparar)
    values2  = compute_value_scores_v2(signals)    # calidad + PE (v2, mejorado)
    regime   = compute_regime(signals)

    result   = run_backtest(signals, values1, values2, regime)

    strats = [
        ("bench_ret", "S0 Benchmark EW"),
        ("s1_ret",    "S1 Tech puro"),
        ("s2_ret",    "S2 Value v1 proxy"),
        ("s3_ret",    "S3 Value v2 calidad"),
        ("s4_ret",    "S4 UNIFIED v2"),
    ]

    all_metrics = []
    for col, label in strats:
        m = calc_metrics(result[col], label)
        all_metrics.append(m)

    print_metrics_table(all_metrics)

    reg_dist = result["regime"].value_counts()
    print("\nDistribucion de regimen (periodo backtest):")
    for r in ["MARKUP", "DISTRIBUCION", "ACUMULACION", "MARKDOWN"]:
        n   = reg_dist.get(r, 0)
        pct = n / len(result) * 100
        bar = "#" * int(pct / 2)
        print(f"  {r:<14} {n:>4} dias  {pct:>5.1f}%  {bar}")

    plot_results(result, all_metrics)
