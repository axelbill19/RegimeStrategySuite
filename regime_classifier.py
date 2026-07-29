"""
Regime Classifier — el ABM traducido a una señal sobre datos reales
=====================================================================
El simulador ABM (market.py, agents.py) probó que los sesgos conductuales
(loss aversion, herding, ausencia de ancla fundamental) generan patrones
estadísticos reconocibles: colas gruesas + skew negativo en pánico, baja
kurtosis + tendencia sostenida en burbujas sin value investors, momentum
persistente en mercados dominados por herding (ver BehavioralMarket/README.md,
secciones "Key Findings" y "Sensitivity Analysis").

Este módulo NO vuelve a correr el ABM. Calcula los mismos momentos
estadísticos (kurtosis, skewness, drawdown, persistencia de tendencia) sobre
retornos reales de mercado, con percentiles walk-forward (expanding, sin
look-ahead) — el mismo patrón que ya usa `fear_pct` en tech_regime_backtest.py.
Los umbrales no intentan igualar la magnitud simulada (un tick de Mesa no es
un día de trading); solo usan los 4 estados como vocabulario compartido con
el ABM para razonar sobre el régimen actual.

Estados:
  PANIC  — cola izquierda gruesa + kurtosis alta + drawdown relevante.
           Análogo a "Panic Mkt" (dominado por loss-aversion). Históricamente
           precede rebotes — señal contrarian, no de salida.
  BUBBLE — tendencia sostenida + kurtosis inusualmente baja ("demasiado
           calmo") + retorno acumulado en el percentil alto. Análogo a
           "No Anchor" (sin value investors) — el ABM mostró +249% sin techo
           hasta el colapso. Señal de cautela, no de salida.
  HERD   — tendencia positiva con alta persistencia (autocorrelación de
           signo). Análogo a "Herd Mkt" (momentum 2x, FOMO). El momentum
           funciona pero el ABM lo marca como el régimen más propenso a
           reversión brusca.
  NORMAL — ninguna condición anterior. Comportamiento por defecto.

Prioridad si varias condiciones disparan a la vez: PANIC > BUBBLE > HERD > NORMAL.
"""

import numpy as np
import pandas as pd
from scipy.stats import kurtosis as sci_kurt, skew as sci_skew

# ── Ventanas ────────────────────────────────────────────────────────────────
KURT_WINDOW  = 60
SKEW_WINDOW  = 60
DD_WINDOW    = 120
TREND_WINDOW = 60
PERSIST_WINDOW = 20
RET120_WINDOW  = 120

# ── Umbrales (percentil walk-forward, expanding) ────────────────────────────
KURT_PANIC_PCT    = 0.80   # kurtosis en el 20% mas alto historico
SKEW_PANIC_PCT     = 0.20   # skew en el 20% mas negativo historico
DD_PANIC_MIN       = 0.15   # drawdown minimo 15% desde el maximo de 120d

KURT_BUBBLE_PCT     = 0.30   # kurtosis "demasiado calma" (percentil bajo)
RET120_BUBBLE_PCT    = 0.90   # retorno acumulado 120d en el 10% mas alto

TREND_PERSIST_PCT   = 0.70   # persistencia de signo en el 30% mas alto


def compute_features(returns: pd.Series) -> pd.DataFrame:
    """Momentos estadisticos rodantes sobre una serie de retornos diarios."""
    r = returns.dropna()

    kurt_60 = r.rolling(KURT_WINDOW).apply(
        lambda x: sci_kurt(x), raw=True)
    skew_60 = r.rolling(SKEW_WINDOW).apply(
        lambda x: sci_skew(x), raw=True)

    price = (1 + r).cumprod()
    roll_max = price.rolling(DD_WINDOW, min_periods=1).max()
    dd_120 = (price - roll_max) / roll_max  # negativo o cero

    mom_60 = r.rolling(TREND_WINDOW).mean()
    ret_120 = price / price.shift(RET120_WINDOW) - 1

    same_sign = (np.sign(r) == np.sign(r.shift(1))).astype(float)
    trend_persist_20 = same_sign.rolling(PERSIST_WINDOW).mean()

    feats = pd.DataFrame({
        "kurt_60": kurt_60,
        "skew_60": skew_60,
        "dd_120": dd_120,
        "mom_60": mom_60,
        "ret_120": ret_120,
        "trend_persist_20": trend_persist_20,
    }, index=r.index)
    return feats


def classify_state(feats: pd.DataFrame) -> pd.Series:
    """Clasifica cada fecha en PANIC/BUBBLE/HERD/NORMAL usando percentiles
    expanding (walk-forward, sin look-ahead)."""
    kurt_pct   = feats["kurt_60"].expanding().rank(pct=True)
    skew_pct   = feats["skew_60"].expanding().rank(pct=True)
    ret120_pct = feats["ret_120"].expanding().rank(pct=True)
    persist_pct = feats["trend_persist_20"].expanding().rank(pct=True)

    is_panic = (
        (kurt_pct >= KURT_PANIC_PCT) &
        (skew_pct <= SKEW_PANIC_PCT) &
        (feats["dd_120"].abs() >= DD_PANIC_MIN)
    )

    is_bubble = (
        (feats["mom_60"] > 0) &
        (kurt_pct <= KURT_BUBBLE_PCT) &
        (ret120_pct >= RET120_BUBBLE_PCT)
    )

    is_herd = (
        (feats["mom_60"] > 0) &
        (persist_pct >= TREND_PERSIST_PCT)
    )

    state = pd.Series("NORMAL", index=feats.index)
    state[is_herd]   = "HERD"
    state[is_bubble] = "BUBBLE"
    state[is_panic]  = "PANIC"   # maxima prioridad, se aplica al final
    return state


def classify_market_regime(returns: pd.Series) -> pd.DataFrame:
    """Punto de entrada unico: retornos -> features + estado conductual."""
    feats = compute_features(returns)
    feats["state"] = classify_state(feats)
    return feats


if __name__ == "__main__":
    # Smoke test rapido con ruido aleatorio
    idx = pd.date_range("2020-01-01", periods=500, freq="B")
    rng = np.random.default_rng(0)
    rets = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    out = classify_market_regime(rets)
    print(out["state"].value_counts())
