from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass


# data
@dataclass
class RawData:
    ticker: str
    _timestamp: pd.Timestamp
    call_chain: pd.DataFrame
    put_chain: pd.DataFrame
    spot: float

    @property
    def time_stamp(self):
        if self._timestamp.tzinfo is None:
            return self._timestamp.tz_localize("UTC")
        else:
            return self._timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class FitConfig:
    """
    An immutable container of configuration parameters for the
    data-cleaning pipeline and the SSVI fit.
    Attribute values should not be mutated after construction.

    Attributes:
        time_scale (int): Number of periods used to annualize days-to-expiry. Defaults to 365.
        days_old (int): Maximum number of days since the last-executed trade at a given strike
            and expiry for that data to be included. Can be set to `np.inf`. Defaults to 7.
        flntol (int): minimum number of quotes in the ATM band to allow the calculation of
            forwards and rates by put-call parity regression. Defaults to 5.
        sbf (float): the spot boundary factor: the maximum half-distance from
            spot at which a strike is "at the money", as a percentage of the spot.
            Defaults to 0.1.
        bmf: the distance below the initial-guess forward to look for Near-but-OTM puts.
            Defaults to 0.15.
        bounds: Lower and upper bound tuples constraining the square-root SSVI parameters.
            Defaults set to Gatheral's recommendations
        penalty_factor: Multiplier applied to penalize crossedness violations
            during optimization. Defaults to 100000.
        kbp: If set to (a,b), crossedness (and no-arbitrage guarantees) are only considered
        within a guaranteed range equal to the traded range + padding = (min_k-a, max_k+b).
        Can be set to (0,0). If None, optimizer will attempt to eliminate calendar-arbitrage
        over all k. Defaults to None.
    """

    time_scale: int = 365
    days_old: int = 7
    flntol: int = 5
    sbf: float = 0.1
    bmf: float = 0.15
    bounds: tuple[tuple, tuple] = (
        (0, -np.inf, -0.999, 0, 1e-6),
        (np.inf, np.inf, 0.999, np.inf, np.inf),
    )
    penalty_factor: float = 1000000
    kbp: float | None = None


@dataclass(frozen=True)
class DataSet:
    ticker: str
    config: FitConfig
    timestamp: pd.Timestamp
    data: pd.DataFrame
    spot: float
    rate: float


# fitting
@dataclass
class InitialFitObject:
    thetas: dict[pd.Timestamp, float]
    rho: float
    eta: float


@dataclass(frozen=True)
class SliceMarketData:
    expiry: pd.Timestamp
    k: pd.Series[float]
    isPut: pd.Series[bool]
    forwardPrice: pd.Series[float]
    strike: pd.Series[float]
    mids: pd.Series[float]
    iv: pd.Series[float]
    bidIV: pd.Series[float]
    askIV: pd.Series[float]
    yte: float
    slice_index: pd.Index

    @classmethod
    def from_df_slice(cls, exp: pd.Timestamp, slice: pd.DataFrame):
        slice = slice.copy()
        return cls(
            exp,
            slice["logMoneyness"],
            slice["putData"],
            slice["forwardPrice"],
            slice["strike"],
            slice["invMid"],
            slice["midIV"],
            slice["bidIV"],
            slice["askIV"],
            slice["yearsToExpiry"].iloc[0],
            slice.index,
        )


@dataclass
class SliceParams:
    delta: float
    mu: float
    rho: float
    theta: float
    zeta: float

    def to_tuple(self):
        return (self.delta, self.mu, self.rho, self.theta, self.zeta)

    @classmethod
    def from_init(cls, rho: float, theta: float, eta: float):
        return cls(0, 0, rho, theta, eta / np.sqrt(theta))


@dataclass
class FitSlice:
    market_data: SliceMarketData
    slice_params: SliceParams
    converged: bool


@dataclass
class SurfaceDiagnostics:
    thetas_per_expiry: dict[pd.Timestamp, float]
    cal_cond_2: float
    cal_cond_2_bound: float
    but_cond_1: list | np.ndarray
    but_cond_2: float
    each_cross: dict[pd.Timestamp, float]
    per_exp_beta: dict[pd.Timestamp, float]


@dataclass
class SSVI:
    ticker: str
    config: FitConfig
    d_rate: float
    data: DataSet
    inits: InitialFitObject
    expiry_slices: list[FitSlice]
    all_slices: list[FitSlice]
    diagnostics: SurfaceDiagnostics | None = None
