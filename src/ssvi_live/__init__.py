"""
ssvi_live: SSVI implied-volatility surfaces fit to live listed option
chains, following Gatheral & Jacquier (2013), "Arbitrage-free SVI
volatility surfaces" (https://arxiv.org/abs/1204.0646).

See ssvi_live.main.live_fit for the live-data entry point and
ssvi_live.main.demo for an offline demo requiring no network access.
"""

from ssvi_live.main import live_fit, demo
from ssvi_live.data import get_data_yfinance, pipeline
from ssvi_live.fit_ssvi import crossedness, new_fit, ssvi_diagnostics
from ssvi_live.models import (
    bsm,
    reverse_bsm,
    sqrt_svi,
    natural_svi,
)
from ssvi_live.diag import (
    surface_fly_condition,
    surface_cal_condition,
    natural_g,
    beta,
)
from ssvi_live.fit_objects import (
    RawData,
    FitConfig,
    DataSet,
    InitialFitObject,
    SliceMarketData,
    SliceParams,
    FitSlice,
    SSVI,
    SurfaceDiagnostics,
)

__all__ = [
    "live_fit",
    "ssvi_diagnostics",
    "demo",
    "get_data_yfinance",
    "pipeline",
    "crossedness",
    "new_fit",
    "bsm",
    "reverse_bsm",
    "sqrt_svi",
    "natural_svi",
    "surface_fly_condition",
    "surface_cal_condition",
    "natural_g",
    "beta",
    "RawData",
    "FitConfig",
    "DataSet",
    "InitialFitObject",
    "SliceMarketData",
    "SliceParams",
    "FitSlice",
    "SSVI",
    "SurfaceDiagnostics",
]
