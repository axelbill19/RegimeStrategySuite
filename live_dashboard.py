"""
Live Investment Dashboard
=========================
Responde UNA pregunta: dado el regimen actual, que hago hoy?

  MARKUP      → especular con momentum (SAR+EMA200+RSI)
                solo en empresas con fundamentales razonables (ROIC/PE/deuda OK)
  ACUMULACION → acumular empresas baratas por Graham (precio < Graham Number)
  DISTRIBUCION→ reducir exposicion, vigilar watchlist
  MARKDOWN    → cash. Que hay que ver para que cambie el regimen?

Fuentes:
  Datos/extended_current.parquet     → regimen (596 tickers, hasta jun 2026)
  Datos/fundamentals/ratios_ttm.parquet  + key_metrics_ttm.parquet
  yfinance (2 anos)                  → señales tecnicas por ticker
"""

import warnings
warnings.filterwarnings("ignore")

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from ta.trend import EMAIndicator, PSARIndicator
from ta.momentum import RSIIndicator

from regime_classifier import classify_market_regime

# ── Config ────────────────────────────────────────────────────────────────────

PRICES_PATH  = Path("../Datos/extended_current.parquet")
FMP_DIR      = Path("../Datos/fundamentals")
OUT_DIR      = Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# Regimen
TREND_WIN      = 60
FEAR_PCT_THOLD = 0.65

# Tecnico
PSAR_STEP, PSAR_MAX = 0.02, 0.20
EMA_WIN   = 200
RSI_WIN   = 14
RSI_LO, RSI_HI = 40, 75

# Filtros "fundamentales razonables" para MARKUP
MARKUP_MAX_PE   = 30    # un poco mas laxo — growth con momentum
MARKUP_MIN_ROIC = 0.08
MARKUP_MAX_DEBT = 3.0   # deuda/equity

# Filtros Graham para ACUMULACION
ACUM_GRAHAM_MAX_MULT = 1.5  # precio <= 1.5x Graham Number
ACUM_MIN_ROIC        = 0.08

# En capitulacion (estado conductual PANIC) se relaja el filtro Graham: el
# ABM y la validacion contra el crash de COVID (BehavioralMarket/README.md)
# muestran que ese estado precede rebotes — ser demasiado estricto justo en
# el peor momento hace perder la entrada.
ACUM_GRAHAM_MAX_MULT_PANIC = 2.0

BEHAVIORAL_LABELS = {
    "PANIC":  "PANICO / CAPITULACION",
    "BUBBLE": "BURBUJA / MELT-UP",
    "HERD":   "HERDING / FOMO",
    "NORMAL": "NORMAL",
}
BEHAVIORAL_COLORS = {
    "PANIC": "#f85149", "BUBBLE": "#d29922", "HERD": "#e3b341", "NORMAL": "#8b949e",
}

# Cuantos candidatos descargar de yfinance (los mejor rankeados por FMP)
TOP_MARKUP_CANDS = 40   # solo los top 40 del screener FMP para bajar de yfinance
TOP_ACUM_CANDS   = 30

REGIME_COLORS = {
    "MARKUP":       "#2980b9",
    "DISTRIBUCION": "#e67e22",
    "ACUMULACION":  "#27ae60",
    "MARKDOWN":     "#c0392b",
    "UNKNOWN":      "#95a5a6",
}

# ── 1. Regimen ────────────────────────────────────────────────────────────────

def detect_regime(prices: pd.DataFrame) -> pd.Series:
    mkt_ret  = prices.mean(axis=1).pct_change()
    vol_20   = mkt_ret.rolling(20).std() * np.sqrt(252)
    vol_60   = mkt_ret.rolling(60).std() * np.sqrt(252)
    fear_lv  = vol_20 / vol_60.replace(0, np.nan)
    fear_pct = fear_lv.expanding().rank(pct=True)
    mom_60   = mkt_ret.rolling(TREND_WIN).mean()
    pos_trend   = mom_60 > 0
    high_stress = fear_pct >= FEAR_PCT_THOLD
    r = pd.Series("UNKNOWN", index=prices.index)
    r[ pos_trend & ~high_stress] = "MARKUP"
    r[ pos_trend &  high_stress] = "DISTRIBUCION"
    r[~pos_trend & ~high_stress] = "ACUMULACION"
    r[~pos_trend &  high_stress] = "MARKDOWN"
    return r


# ── 2. Cargar candidatos FMP ──────────────────────────────────────────────────

def load_fmp_candidates() -> pd.DataFrame:
    for f in ["key_metrics_ttm.parquet", "ratios_ttm.parquet", "universe.parquet"]:
        if not (FMP_DIR / f).exists():
            print(f"[ERROR] Falta {f} — corre FMP/fmp_downloader.py primero")
            sys.exit(1)

    km  = pd.read_parquet(FMP_DIR / "key_metrics_ttm.parquet")
    rt  = pd.read_parquet(FMP_DIR / "ratios_ttm.parquet")
    uni = pd.read_parquet(FMP_DIR / "universe.parquet")

    df = uni.merge(
        km[["symbol","grahamNumberTTM","returnOnInvestedCapitalTTM",
            "returnOnAssetsTTM","currentRatioTTM","evToEBITDATTM",
            "incomeQualityTTM","freeCashFlowYieldTTM"]],
        on="symbol", how="inner"
    ).merge(
        rt[["symbol","priceToEarningsRatioTTM","priceToBookRatioTTM",
            "netProfitMarginTTM","grossProfitMarginTTM",
            "debtToEquityRatioTTM","currentRatioTTM",
            "dividendYieldTTM","bookValuePerShareTTM"]],
        on="symbol", how="inner", suffixes=("_km","_rt")
    )

    df["pe"]           = df["priceToEarningsRatioTTM"].fillna(0)
    df["pb"]           = df["priceToBookRatioTTM"].fillna(0)
    df["roic"]         = df["returnOnInvestedCapitalTTM"].fillna(0)
    df["current_r"]    = df["currentRatioTTM_rt"].fillna(df["currentRatioTTM_km"])
    df["debt_eq"]      = df["debtToEquityRatioTTM"].fillna(999)
    df["net_margin"]   = df["netProfitMarginTTM"].fillna(0)
    df["graham"]       = df["grahamNumberTTM"].fillna(0)
    df["graham_mult"]  = np.where(
        (df["graham"] > 0) & (df["price"] > 0),
        df["price"] / df["graham"], np.nan
    )
    df["fcf_yield"]    = df["freeCashFlowYieldTTM"].fillna(0)
    df["ev_ebitda"]    = df["evToEBITDATTM"].fillna(999)
    return df


# ── 3. Candidatos para MARKUP (fundamentales razonables) ─────────────────────

def get_markup_candidates(fmp: pd.DataFrame) -> pd.DataFrame:
    mask = (
        (fmp["pe"] > 0) & (fmp["pe"] <= MARKUP_MAX_PE) &
        (fmp["roic"] >= MARKUP_MIN_ROIC) &
        (fmp["debt_eq"] <= MARKUP_MAX_DEBT) &
        (fmp["net_margin"] > 0.02) &
        (fmp["marketCap"] >= 1e9)   # mid-cap en adelante
    )
    cands = fmp[mask].copy()
    # Rankear por calidad: ROIC + margen + FCF yield
    cands["quality"] = (
        cands["roic"].clip(0, 0.5) / 0.5 * 0.50 +
        cands["net_margin"].clip(0, 0.3) / 0.3 * 0.30 +
        cands["fcf_yield"].clip(0, 0.15) / 0.15 * 0.20
    )
    return cands.sort_values("quality", ascending=False).head(TOP_MARKUP_CANDS)


# ── 4. Candidatos para ACUMULACION (baratos por Graham) ──────────────────────

def get_acum_candidates(fmp: pd.DataFrame, graham_max_mult: float = ACUM_GRAHAM_MAX_MULT) -> pd.DataFrame:
    mask = (
        (fmp["graham"] > 0) &
        (fmp["graham_mult"] <= graham_max_mult) &
        (fmp["roic"] >= ACUM_MIN_ROIC) &
        (fmp["pe"] > 0) &
        (fmp["net_margin"] > 0.02) &
        (fmp["marketCap"] >= 500e6)
    )
    cands = fmp[mask].copy()
    # Rankear por descuento Graham + ROIC
    cands["graham_disc"] = 1 - cands["graham_mult"].clip(0, 2)  # mayor = mas barato
    cands["value_rank"] = (
        cands["graham_disc"].clip(0, 1) * 0.60 +
        cands["roic"].clip(0, 0.5) / 0.5 * 0.40
    )
    return cands.sort_values("value_rank", ascending=False).head(TOP_ACUM_CANDS)


# ── 5. Senales tecnicas via yfinance ─────────────────────────────────────────

def get_tech_signals(tickers: list[str]) -> dict[str, dict]:
    print(f"\nDescargando señales tecnicas para {len(tickers)} candidatos...")
    try:
        raw = yf.download(tickers, period="2y", auto_adjust=True,
                          progress=False, group_by="ticker")
    except Exception as e:
        print(f"  [warn] yfinance error: {e}")
        return {}

    results = {}
    for sym in tickers:
        try:
            if len(tickers) == 1:
                tk = raw.copy()
            else:
                tk = raw[sym].copy() if sym in raw.columns.get_level_values(0) else pd.DataFrame()

            tk = tk.dropna()
            if len(tk) < 220:
                continue

            tk.index = pd.to_datetime(tk.index).tz_localize(None)
            close = tk["Close"]
            high  = tk["High"]
            low   = tk["Low"]

            psar   = PSARIndicator(high=high, low=low, close=close,
                                   step=PSAR_STEP, max_step=PSAR_MAX).psar()
            ema200 = EMAIndicator(close=close, window=EMA_WIN).ema_indicator()
            rsi    = RSIIndicator(close=close, window=RSI_WIN).rsi()

            last = tk.iloc[-1]
            last_psar   = psar.iloc[-1]
            last_ema    = ema200.iloc[-1]
            last_rsi    = rsi.iloc[-1]
            last_close  = float(last["Close"])

            sar_bull    = last_close > last_psar
            above_ema   = last_close > last_ema
            rsi_ok      = RSI_LO <= last_rsi <= RSI_HI
            signal_full = sar_bull and above_ema and rsi_ok

            # Cuantos requisitos cumple (0-3) para "watching"
            score_3 = int(sar_bull) + int(above_ema) + int(rsi_ok)

            results[sym] = {
                "price":      last_close,
                "psar":       float(last_psar),
                "ema200":     float(last_ema),
                "rsi":        float(last_rsi),
                "sar_bull":   sar_bull,
                "above_ema":  above_ema,
                "rsi_ok":     rsi_ok,
                "signal":     signal_full,
                "score_3":    score_3,
                "close_ser":  close,
                "ema_ser":    ema200,
            }
        except Exception:
            continue

    activos = sum(1 for v in results.values() if v["signal"])
    print(f"  Señales obtenidas: {len(results)} tickers  |  activos: {activos}")
    return results


# ── 6. Imprimir dashboard texto ───────────────────────────────────────────────

def _pct(val):
    return f"{val*100:.1f}%"

def print_dashboard(regime_now: str, regime_hist: pd.Series,
                    markup_sigs: dict, markup_cands: pd.DataFrame,
                    acum_cands: pd.DataFrame,
                    behavioral_now: str, behavioral_feats: pd.Series,
                    graham_max_mult_used: float):

    W = 62
    print("\n" + "=" * W)
    print(f"  LIVE INVESTMENT DASHBOARD")
    print("=" * W)

    # Regimen actual
    reg_color = {"MARKUP":"[BULL]","DISTRIBUCION":"[CAUTION]",
                 "ACUMULACION":"[VALUE]","MARKDOWN":"[CASH]","UNKNOWN":"[?]"}
    print(f"\n  REGIMEN ACTUAL: {regime_now}  {reg_color.get(regime_now,'')}")

    behav_label = BEHAVIORAL_LABELS.get(behavioral_now, behavioral_now)
    print(f"  ESTADO CONDUCTUAL: {behav_label}")
    print(f"    kurtosis 60d={behavioral_feats['kurt_60']:+.2f}  "
          f"skew 60d={behavioral_feats['skew_60']:+.2f}  "
          f"drawdown 120d={behavioral_feats['dd_120']*100:.1f}%")

    actions = {
        "MARKUP":       "Especular con momentum. Solo empresas con ROIC/PE razonables.",
        "ACUMULACION":  "Acumular valor. Buscar empresas bajo Graham Number.",
        "DISTRIBUCION": "Reducir exposicion 50%. Preservar ganancias.",
        "MARKDOWN":     "Cash 100%. Esperar confirmacion de giro.",
        "UNKNOWN":      "Sin clasificacion — datos insuficientes.",
    }
    print(f"  Accion: {actions.get(regime_now,'')}")

    overlay_note = None
    if regime_now in ("ACUMULACION", "MARKDOWN") and behavioral_now == "PANIC":
        overlay_note = (
            f"  >> CAPITULACION: filtro Graham relajado a {graham_max_mult_used:.1f}x "
            f"(normal {ACUM_GRAHAM_MAX_MULT}x). El ABM y el crash de COVID (2020-03) "
            f"muestran que este estado suele preceder rebotes — no perder la entrada "
            f"por ser demasiado estricto."
        )
    elif regime_now == "MARKUP" and behavioral_now == "BUBBLE":
        overlay_note = (
            "  >> MELT-UP sin ancla fundamental (analogo a 'No Anchor' del ABM): "
            "considerar reducir el tamano de posicion ~50% aunque la señal tecnica "
            "este activa."
        )
    elif behavioral_now == "HERD":
        overlay_note = (
            "  >> Momentum con alta persistencia (herding/FOMO) — funciona, pero "
            "el ABM lo marca como el regimen mas propenso a reversion brusca."
        )
    if overlay_note:
        print(overlay_note)

    # Ultimos 10 dias de regimen
    print(f"\n  Regimen reciente (ultimos 15 dias):")
    for date, r in regime_hist.tail(15).items():
        mark = " <<< HOY" if date == regime_hist.index[-1] else ""
        sym  = {"MARKUP":"^","ACUMULACION":"v","DISTRIBUCION":"~","MARKDOWN":"X"}.get(r,"?")
        print(f"    {date.date()}  {sym} {r}{mark}")

    # Distribucion ultimos 3 meses
    last3m = regime_hist[regime_hist.index >= regime_hist.index[-1] - pd.DateOffset(months=3)]
    dist = last3m.value_counts()
    total = len(last3m)
    print(f"\n  Distribucion ultimos 3 meses:")
    for r in ["MARKUP","DISTRIBUCION","ACUMULACION","MARKDOWN"]:
        n = dist.get(r, 0)
        bar = "#" * int(n / total * 25)
        print(f"    {r:<14} {n:>3}d  {n/total*100:>4.0f}%  {bar}")

    print("\n" + "-" * W)

    if regime_now == "MARKUP":
        # Mostrar candidatos con señal tecnica activa
        activos = [(sym, v) for sym, v in markup_sigs.items() if v["signal"]]
        watching = [(sym, v) for sym, v in markup_sigs.items()
                    if not v["signal"] and v["score_3"] >= 2]

        activos.sort(key=lambda x: x[1]["rsi"], reverse=False)

        print(f"\n  CANDIDATOS MARKUP — señal ACTIVA ({len(activos)} de {len(markup_sigs)})")
        print(f"  Filtro: ROIC>={MARKUP_MIN_ROIC*100:.0f}% | P/E<={MARKUP_MAX_PE} | Deuda/Eq<={MARKUP_MAX_DEBT}")
        print(f"  {'Ticker':<8}{'Precio':>8}{'EMA200':>8}{'SAR':>8}{'RSI':>7}  {'Empresa'}")
        print(f"  {'-'*8}{'-'*8}{'-'*8}{'-'*8}{'-'*7}  {'-'*25}")

        for sym, v in activos:
            row = markup_cands[markup_cands["symbol"] == sym]
            nombre = row.iloc[0]["companyName"][:25] if not row.empty else ""
            pe_str = f"PE={row.iloc[0]['pe']:.1f}" if not row.empty else ""
            roic_s = f"ROIC={row.iloc[0]['roic']*100:.0f}%" if not row.empty else ""
            print(f"  {sym:<8}${v['price']:>7.2f}  ${v['ema200']:>7.2f}  ${v['psar']:>7.2f}"
                  f"  {v['rsi']:>5.1f}  {nombre}")
            print(f"  {'':8}  {pe_str}  {roic_s}")

        if watching:
            print(f"\n  WATCHING (2/3 condiciones activas) — {len(watching)} candidatos:")
            print(f"  {'Ticker':<8}{'Precio':>8}  SAR  EMA  RSI  Condicion faltante")
            for sym, v in sorted(watching, key=lambda x: -x[1]["score_3"])[:15]:
                falta = []
                if not v["sar_bull"]:  falta.append("SAR")
                if not v["above_ema"]: falta.append("EMA200")
                if not v["rsi_ok"]:    falta.append("RSI")
                s = " ".join(["[X]" if b else "[ ]" for b in
                              [v["sar_bull"], v["above_ema"], v["rsi_ok"]]])
                pct_to_ema = (v["ema200"]/v["price"]-1)*100
                info = f"+{pct_to_ema:.1f}% a EMA200" if not v["above_ema"] else ""
                print(f"  {sym:<8}${v['price']:>7.2f}  {s}  falta:{','.join(falta)}  {info}")

    elif regime_now == "ACUMULACION":
        print(f"\n  CANDIDATOS ACUMULACION — bajo/cerca Graham Number")
        print(f"  Filtro: precio <= {graham_max_mult_used:.1f}x Graham Number | ROIC>={ACUM_MIN_ROIC*100:.0f}%")
        print(f"\n  {'Ticker':<8}{'Precio':>8}{'Graham':>8}{'Mult':>7}{'ROIC':>7}{'PE':>6}  {'Empresa'}")
        print(f"  {'-'*8}{'-'*8}{'-'*8}{'-'*7}{'-'*7}{'-'*6}  {'-'*25}")
        for _, r in acum_cands.head(20).iterrows():
            mult_s = f"{r['graham_mult']:.2f}x" if pd.notna(r['graham_mult']) else "N/A"
            print(f"  {r['symbol']:<8}${r['price']:>7.2f}  ${r['graham']:>7.2f}"
                  f"  {mult_s:>6}  {r['roic']*100:>5.1f}%  {r['pe']:>5.1f}  "
                  f"{r['companyName'][:25]}")

    elif regime_now == "DISTRIBUCION":
        print(f"\n  DISTRIBUCION — mercado tenso pero tendencia positiva")
        print(f"  Accion: reducir posiciones actuales al 50%")
        print(f"  Mantener solo las convicciones mas altas.")
        print(f"\n  Watchlist Graham para cuando baje el estres:")
        for _, r in acum_cands.head(10).iterrows():
            mult_s = f"{r['graham_mult']:.2f}x" if pd.notna(r['graham_mult']) else "N/A"
            print(f"    {r['symbol']:<8} Graham={mult_s}  ROIC={r['roic']*100:.0f}%  "
                  f"PE={r['pe']:.1f}  {r['companyName'][:30]}")

    else:  # MARKDOWN
        print(f"\n  MARKDOWN — cash. No atrapar el cuchillo.")
        print(f"\n  Para detectar el giro a ACUMULACION:")
        print(f"    1. Vol 20d < Vol 60d  (estres bajando)")
        print(f"    2. Mercado EW con retorno 60d > 0  (tendencia virando)")
        print(f"    3. Cuando ambas → ACUMULACION; entrar con lista value abajo")
        print(f"\n  Watchlist para comprar en ACUMULACION:")
        for _, r in acum_cands.head(10).iterrows():
            mult_s = f"{r['graham_mult']:.2f}x" if pd.notna(r['graham_mult']) else "N/A"
            print(f"    {r['symbol']:<8} Graham={mult_s}  ROIC={r['roic']*100:.0f}%  "
                  f"PE={r['pe']:.1f}")

    print("\n" + "=" * W)


# ── 7. Grafica ────────────────────────────────────────────────────────────────

def plot_dashboard(regime: pd.Series, regime_now: str,
                   markup_sigs: dict, markup_cands: pd.DataFrame,
                   acum_cands: pd.DataFrame, prices_wide: pd.DataFrame,
                   behavioral: pd.Series, behavioral_now: str):

    LOOKBACK_M = 18
    cutoff = regime.index[-1] - pd.DateOffset(months=LOOKBACK_M)
    reg_plot = regime[regime.index >= cutoff]

    activos  = {s: v for s, v in markup_sigs.items() if v["signal"]}
    watching = {s: v for s, v in markup_sigs.items()
                if not v["signal"] and v["score_3"] >= 2}

    # Seleccion de candidatos a graficar
    to_plot_tech  = list(activos.keys())[:4]
    to_plot_value = acum_cands["symbol"].head(4).tolist()
    n_plots = len(to_plot_tech) + len(to_plot_value)

    rows = 2 + max(len(to_plot_tech), 1)
    fig  = plt.figure(figsize=(16, 5 + 3.5 * rows))
    fig.suptitle(f"LIVE DASHBOARD  |  Regimen: {regime_now}",
                 fontsize=13, fontweight="bold", y=0.99)

    gs = gridspec.GridSpec(1 + len(to_plot_tech) + len(to_plot_value), 2,
                           figure=fig, hspace=0.55, wspace=0.30)

    def _regime_bg(ax, rs):
        prev, t0 = None, rs.index[0]
        for t, r in rs.items():
            if r != prev and prev is not None:
                ax.axvspan(t0, t, alpha=0.10,
                           color=REGIME_COLORS.get(prev, "#888"), lw=0)
                t0 = t
            prev = r
        ax.axvspan(t0, rs.index[-1], alpha=0.10,
                   color=REGIME_COLORS.get(prev, "#888"), lw=0)

    # ── Panel regimen ──────────────────────────────────────────────────────────
    ax_r = fig.add_subplot(gs[0, :])
    mkt  = prices_wide.mean(axis=1)
    mkt_plot = mkt[mkt.index >= cutoff]
    mkt_norm  = mkt_plot / mkt_plot.iloc[0]
    ax_r2 = ax_r.twinx()
    ax_r2.plot(mkt_norm.index, mkt_norm.values, color="gray",
               lw=1.2, alpha=0.45, label="Mercado EW norm.")
    ax_r2.set_ylabel("Mercado norm.", color="gray", fontsize=8)
    ax_r2.tick_params(axis="y", labelcolor="gray", labelsize=7)

    _regime_bg(ax_r, reg_plot)
    rmap = {"MARKUP":4,"DISTRIBUCION":3,"ACUMULACION":2,"MARKDOWN":1,"UNKNOWN":0}
    reg_num = reg_plot.map(rmap)
    ax_r.step(reg_plot.index, reg_num.values, where="post",
              color="#2c3e50", lw=1.5)
    ax_r.set_yticks([1, 2, 3, 4])
    ax_r.set_yticklabels(["MARKDOWN","ACUMULACION","DISTRIBUCION","MARKUP"], fontsize=8)
    ax_r.set_title(f"Regimen de Mercado — ultimos {LOOKBACK_M} meses", fontweight="bold")
    ax_r.grid(axis="x", alpha=0.2)
    patches = [mpatches.Patch(color=REGIME_COLORS[r], alpha=0.6, label=r)
               for r in ["MARKUP","DISTRIBUCION","ACUMULACION","MARKDOWN"]]
    ax_r.legend(handles=patches, loc="lower left", fontsize=7.5, ncol=4)

    # Franja superior con estado conductual (solo PANIC/BUBBLE/HERD — NORMAL
    # se deja transparente para no saturar el panel)
    behav_plot = behavioral[behavioral.index >= cutoff]
    if len(behav_plot) > 0:
        prev, t0 = None, behav_plot.index[0]
        for t, s in behav_plot.items():
            if s != prev and prev is not None:
                if prev in BEHAVIORAL_COLORS and prev != "NORMAL":
                    ax_r.axvspan(t0, t, alpha=0.40, color=BEHAVIORAL_COLORS[prev],
                                lw=0, ymin=0.93, ymax=1.0)
                t0 = t
            prev = s
        if prev in BEHAVIORAL_COLORS and prev != "NORMAL":
            ax_r.axvspan(t0, behav_plot.index[-1], alpha=0.40,
                        color=BEHAVIORAL_COLORS[prev], lw=0, ymin=0.93, ymax=1.0)

    behav_now_label = BEHAVIORAL_LABELS.get(behavioral_now, behavioral_now)
    ax_r.text(0.99, 0.03, f"Estado conductual actual: {behav_now_label}",
             transform=ax_r.transAxes, ha="right", va="bottom", fontsize=8,
             fontweight="bold", color=BEHAVIORAL_COLORS.get(behavioral_now, "#333"))

    row_idx = 1

    # ── Paneles tech (activos en MARKUP) ──────────────────────────────────────
    for sym in to_plot_tech:
        v = markup_sigs[sym]
        cutoff_tk = v["close_ser"].index[-1] - pd.DateOffset(months=LOOKBACK_M)
        close_p   = v["close_ser"][v["close_ser"].index >= cutoff_tk]
        ema_p     = v["ema_ser"][v["ema_ser"].index >= cutoff_tk]
        reg_tk    = regime.reindex(close_p.index, method="ffill")

        fmp_row  = markup_cands[markup_cands["symbol"] == sym]
        pe_str   = f"PE={fmp_row.iloc[0]['pe']:.1f}  ROIC={fmp_row.iloc[0]['roic']*100:.0f}%" \
                   if not fmp_row.empty else ""

        ax_p = fig.add_subplot(gs[row_idx, 0])
        _regime_bg(ax_p, reg_tk)
        ax_p.plot(close_p.index, close_p.values, color="#2c3e50", lw=1.5, label="Precio")
        ax_p.plot(ema_p.index,   ema_p.values,   color="#2980b9", lw=1.2,
                  ls="--", alpha=0.8, label="EMA200")
        ax_p.set_title(f"{sym} — ACTIVO  |  {pe_str}", fontweight="bold", fontsize=9)
        ax_p.set_ylabel("USD"); ax_p.legend(fontsize=7.5); ax_p.grid(alpha=0.2)

        ax_q = fig.add_subplot(gs[row_idx, 1])
        _regime_bg(ax_q, reg_tk)
        rsi_ser = RSIIndicator(close=v["close_ser"], window=RSI_WIN).rsi()
        rsi_p   = rsi_ser[rsi_ser.index >= cutoff_tk]
        ax_q.plot(rsi_p.index, rsi_p.values, color="#8e44ad", lw=1.4, label="RSI 14")
        ax_q.axhline(RSI_HI, color="#e74c3c", lw=0.9, ls="--")
        ax_q.axhline(RSI_LO, color="#27ae60", lw=0.9, ls="--")
        ax_q.fill_between(rsi_p.index, RSI_LO, RSI_HI, alpha=0.08, color="#27ae60")
        ax_q.set_ylim(0, 100)
        ax_q.set_title(f"{sym} — RSI 14", fontsize=9)
        ax_q.set_ylabel("RSI"); ax_q.legend(fontsize=7.5); ax_q.grid(alpha=0.2)
        row_idx += 1

    # ── Paneles value (candidatos Graham) ─────────────────────────────────────
    for sym in to_plot_value:
        fmp_row = acum_cands[acum_cands["symbol"] == sym]
        if fmp_row.empty:
            continue
        fr = fmp_row.iloc[0]
        try:
            raw = yf.download(sym, period="2y", auto_adjust=True, progress=False)
            if raw.empty or len(raw) < 50:
                continue
            raw.index = pd.to_datetime(raw.index).tz_localize(None)
            close  = raw["Close"]
            cutoff_v = close.index[-1] - pd.DateOffset(months=LOOKBACK_M)
            close_p  = close[close.index >= cutoff_v]
            reg_tk   = regime.reindex(close_p.index, method="ffill")

            ax_v = fig.add_subplot(gs[row_idx, 0])
            _regime_bg(ax_v, reg_tk)
            ax_v.plot(close_p.index, close_p.values, color="#27ae60", lw=1.5, label="Precio")
            ax_v.axhline(fr["graham"], color="#e74c3c", lw=1.2, ls="--",
                         label=f"Graham ${fr['graham']:.2f}")
            mult_str = f"{fr['graham_mult']:.2f}x" if pd.notna(fr["graham_mult"]) else ""
            ax_v.set_title(
                f"{sym} — VALUE  |  {mult_str} Graham  "
                f"ROIC={fr['roic']*100:.0f}%  PE={fr['pe']:.1f}",
                fontweight="bold", fontsize=9
            )
            ax_v.set_ylabel("USD"); ax_v.legend(fontsize=7.5); ax_v.grid(alpha=0.2)

            # Panel derecho: tabla de metricas clave
            ax_t = fig.add_subplot(gs[row_idx, 1])
            ax_t.axis("off")
            data = [
                ["Metrica",         "Valor"],
                ["Precio",          f"${fr['price']:.2f}"],
                ["Graham Number",   f"${fr['graham']:.2f}"],
                ["Precio/Graham",   f"{fr['graham_mult']:.2f}x" if pd.notna(fr['graham_mult']) else "N/A"],
                ["P/E",             f"{fr['pe']:.1f}"],
                ["P/B",             f"{fr['pb']:.2f}"],
                ["ROIC",            f"{fr['roic']*100:.1f}%"],
                ["Margen neto",     f"{fr['net_margin']*100:.1f}%"],
                ["Deuda/Equity",    f"{fr['debt_eq']:.2f}x"],
                ["Sector",          fr["sector"][:20] if pd.notna(fr["sector"]) else "N/A"],
            ]
            tbl = ax_t.table(cellText=[r for r in data[1:]], colLabels=data[0],
                             loc="center", cellLoc="center")
            tbl.auto_set_font_size(False)
            tbl.set_fontsize(8.5)
            tbl.scale(1.1, 1.5)
            for j in range(2):
                tbl[0, j].set_facecolor("#1F497D")
                tbl[0, j].set_text_props(color="white", fontweight="bold")
            ax_t.set_title(f"{sym} — metricas clave", fontsize=9)
            row_idx += 1
        except Exception:
            continue

    out = OUT_DIR / "live_dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\nDashboard guardado: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  LIVE INVESTMENT DASHBOARD")
    print("=" * 62)

    # Regimen
    print("\nCargando precios de mercado...")
    prices = pd.read_parquet(PRICES_PATH)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    print(f"  {prices.shape[1]} tickers | hasta {prices.index[-1].date()}")

    regime    = detect_regime(prices)
    regime_now = regime.iloc[-1]

    # Estado conductual (kurtosis/skew/drawdown/persistencia sobre el
    # mercado EW real — ver BehavioralMarket/regime_classifier.py)
    print("\nClasificando estado conductual...")
    mkt_ret = prices.mean(axis=1).pct_change()
    behavioral_feats_df = classify_market_regime(mkt_ret)
    behavioral    = behavioral_feats_df["state"]
    behavioral_now = behavioral.iloc[-1]
    behavioral_feats_now = behavioral_feats_df.iloc[-1]
    print(f"  Estado actual: {behavioral_now}")

    graham_max_mult_used = (
        ACUM_GRAHAM_MAX_MULT_PANIC if behavioral_now == "PANIC" else ACUM_GRAHAM_MAX_MULT
    )

    # Candidatos FMP
    print("\nCargando datos FMP...")
    fmp = load_fmp_candidates()
    print(f"  {len(fmp)} tickers con fundamentales")

    markup_cands = get_markup_candidates(fmp)
    acum_cands   = get_acum_candidates(fmp, graham_max_mult=graham_max_mult_used)
    print(f"  Candidatos MARKUP (fundamentales razonables): {len(markup_cands)}")
    print(f"  Candidatos ACUMULACION (Graham <= {graham_max_mult_used:.1f}x):  {len(acum_cands)}")

    # Señales tecnicas para candidatos MARKUP
    markup_syms  = markup_cands["symbol"].tolist()
    markup_sigs  = get_tech_signals(markup_syms)

    # Imprimir dashboard
    print_dashboard(regime_now, regime, markup_sigs, markup_cands, acum_cands,
                    behavioral_now, behavioral_feats_now, graham_max_mult_used)

    # Grafica
    plot_dashboard(regime, regime_now, markup_sigs, markup_cands, acum_cands, prices,
                   behavioral, behavioral_now)
