from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from ssvi_live.models import bsm, natural_svi, sqrt_svi
from ssvi_live.fit_objects import (
    InitialFitObject,
    SliceParams,
    SliceMarketData,
    FitSlice,
    FitConfig,
    SSVI,
    DataSet,
    SurfaceDiagnostics,
)
from ssvi_live.diag import (
    surface_fly_condition,
    surface_cal_condition,
    beta,
)

log = logging.getLogger(__name__)


def _first_fit(
    df: pd.DataFrame,
    config: FitConfig,
    d_rate: float,
) -> InitialFitObject:
    """
    A one-shot fit (`scipy.optimize.least_squares`) of a single global SSVI.
    Returns fit parameters (rho, eta) as an initial guess for each slice.
    The objective is the residual in price-space.

    Theta is read off each expiry by linearly interpolating total variance between
    the two quotes bracketing k = 0 (section 4 intro).

    Raises RuntimeError if least_squares fails to converge.
    """
    totalVar = df["totalVar"]
    price_obs = df["invMid"]

    # find the atm variance (theta_0)
    theta_dict = dict()
    for exp, slice in df.groupby("expiryDate"):
        # _final_filter() guarantees quotes on both sides of the money per expiry,
        # so the w(k) intercept is guaranteed to exist and be positive.
        up = slice.loc[slice["logMoneyness"] > 0, "logMoneyness"].idxmin(skipna=True)
        low = slice.loc[slice["logMoneyness"] < 0, "logMoneyness"].idxmax(skipna=True)
        k = np.array((slice.loc[up, "logMoneyness"], slice.loc[low, "logMoneyness"]))
        theta = np.array((totalVar.loc[up], totalVar.loc[low]))
        _, b = np.polyfit(k, theta, deg=1)
        theta_dict[exp] = b
    theta_0 = df["expiryDate"].map(theta_dict)

    # The fit
    def obj(rho_eta):
        rho, eta = rho_eta
        totalVar_pred = sqrt_svi(df["logMoneyness"], rho, eta, theta_0)  # type: ignore sqrt_svi() is fully vectorized
        iv_pred = np.sqrt(totalVar_pred / df["yearsToExpiry"])
        price_pred = bsm(
            df["forwardPrice"],
            df["strike"],
            df["yearsToExpiry"],
            iv_pred,
            d_rate,
            df["putData"],
        )
        return price_pred - price_obs

    x0 = (-0.5, 1)
    bounds = ([config.bounds[0][2], 1e-6], [config.bounds[1][2], np.inf])
    result = least_squares(
        obj,
        x0,
        bounds=bounds,
    )

    # Reporting
    if not result.success:
        log.error(
            f"Global square-root SSVI fit did not converge: status={result.status}, "
            f"message={result.message}."
        )
        raise RuntimeError("Initial global SSVI fit failed to converge")
    rho_init, eta_init = result.x
    return InitialFitObject(theta_dict, rho_init, eta_init)


def _find_svi_roots(prev: SliceParams, next: SliceParams):
    """
    Quartic root finder that helps enforce the calendar-arbitrage condition.
    Called from within `crossedness`.
    Written by LLM and verified against natural_svi.
    """

    def to_hyperbola_form(p: SliceParams):
        alpha = p.delta + 0.5 * p.theta * (1.0 - p.rho * p.zeta * p.mu)
        beta = 0.5 * p.theta * p.rho * p.zeta
        gamma = 0.5 * p.theta * p.zeta
        mu = p.mu - (p.rho / p.zeta)
        sigma = np.sqrt(1.0 - p.rho**2) / p.zeta
        return alpha, beta, gamma, mu, sigma

    a1, b1, g1, m1, s1 = to_hyperbola_form(prev)
    a2, b2, g2, m2, s2 = to_hyperbola_form(next)

    l0, l1 = (a1 - a2), (b1 - b2)

    R1 = np.array([1.0, -2.0 * m1, m1**2 + s1**2])
    R2 = np.array([1.0, -2.0 * m2, m2**2 + s2**2])

    # P(k) = g2^2 * R2 - g1^2 * R1 - (l0 + l1*k)^2
    P = g2**2 * R2 - g1**2 * R1 - np.array([l1**2, 2.0 * l0 * l1, l0**2])
    # Q(k) = 2 * (l0 + l1*k) * g1
    Q = np.array([2.0 * l1 * g1, 2.0 * l0 * g1])

    # P(k)^2 - Q(k)^2 * R1(k) = 0
    P_sq = np.convolve(P, P)
    Q_sq = np.convolve(Q, Q)
    Q_sq_times_R1 = np.convolve(Q_sq, R1)

    quartic_coeffs = P_sq - Q_sq_times_R1

    potential_roots = np.roots(quartic_coeffs)

    real_roots = potential_roots[np.isreal(potential_roots)].real

    valid_crossings = []
    for r in real_roots:
        if np.allclose(
            natural_svi(r, **prev.__dict__), natural_svi(r, **next.__dict__), atol=1e-7
        ):
            valid_crossings = [r] + valid_crossings

    return sorted(list(set(np.round(valid_crossings, decimals=7))))


def crossedness(prev: FitSlice, next: FitSlice, kb: tuple | None = None) -> float:
    """
    Gatheral's algorithm for calculating the calendar-arbitrage violation
    between two adjacent-expiry curves.

    !Definition 5.1: crossedness is "null" when the slices do not cross

    Key change from paper: Looking for crossings over all k means many slices do not
    converge in the final fit.

    Args:
        prev (FitSlice): The earlier-expiry natural SSVI slice
        next (FitSlice): The later-expiry natural SSVI slice
        kb (tuple((float), (float)): restricts the crossedness measure to a range
        of log-moneyness values, and should be None when strictly reproducing the paper.

    Raises:
        RuntimeError: if the numerical root finder returns more than 4 roots,
        since the difference of two SSVI parameterizations should be quartic.

    Returns:
        float: c_max (gatheral 5.2). The
        largest positive differences in total variance-space
        between `prev` and `next` between crossings

    Public because it's used as a diagnostic when plotting.
    """
    prev_p = prev.slice_params
    next_p = next.slice_params
    func1 = lambda x: natural_svi(x, **prev_p.__dict__)
    func2 = lambda x: natural_svi(x, **next_p.__dict__)

    zeros = _find_svi_roots(prev_p, next_p)
    if len(zeros) > 4:
        raise RuntimeError("More than 4 roots in a quartic")

    if kb is not None:
        zeros = [z for z in zeros if kb[0] < z < kb[1]]
    if len(zeros) == 0:
        return 0
    k_t = (
        [zeros[0] - 1]
        + [(zeros[i - 1] + zeros[i]) / 2 for i in range(1, len(zeros))]
        + [zeros[-1] + 1]
    )
    c = max([0] + [func1(k_i) - func2(k_i) for k_i in k_t])

    return c


def _cross_penalty(
    params: tuple,
    fit: FitSlice,
    prev: FitSlice | None,
    next: FitSlice | None,
    d_rate: float,
    penalty_factor: float,
    kb: tuple,
):
    """
    Produces residual vector for one slice's fit in price-space, with
    crossedness penalty applied to enforce no cal-arb. (bullet 3 of 5.2)

    Signature is ordered for scipy's least_squares,
    params is a natural-SVI tuple in SliceParams order.
    See crossedness docstring for kb.
    """
    mkt = fit.market_data
    curr = FitSlice(mkt, SliceParams(*params), False)
    w_pred = natural_svi(mkt.k, *params)  # type: ignore
    iv_pred = np.sqrt(w_pred / mkt.yte)
    residuals = (
        bsm(mkt.forwardPrice, mkt.strike, mkt.yte, iv_pred, d_rate, mkt.isPut)
        - mkt.mids
    )

    penalty = []
    if prev is not None:
        prev_cross = crossedness(prev, curr, kb)
        penalty.append(prev_cross * penalty_factor)

    if next is not None:
        next_cross = crossedness(curr, next, kb)
        penalty.append(next_cross * penalty_factor)

    return np.concatenate([residuals, penalty])


def _refit(
    curr: FitSlice,
    prev: FitSlice | None,
    next: FitSlice | None,
    config: FitConfig,
    d_rate: float,
    kb: tuple,
) -> FitSlice:
    """
    Refits one expiry slice against its quotes, with a penalty for crossing its
    neighbors.
    The slice-by-slice step of the section 5.2 recipe.
    Does not mutate `curr`.
    """

    params = curr.slice_params.to_tuple()
    result = least_squares(
        _cross_penalty,
        x0=params,
        bounds=config.bounds,
        args=(curr, prev, next, d_rate, config.penalty_factor, kb),
        max_nfev=2000,
    )

    fitted_iv = np.sqrt(natural_svi(curr.market_data.k, *result.x) / curr.market_data.yte)  # type: ignore natural_svi() is vectorized
    rmse = float(np.sqrt(np.mean((fitted_iv - curr.market_data.iv) ** 2)))
    fitted_params = SliceParams(*result.x)
    log.debug(
        f"Slice {curr.market_data.expiry.normalize()}: {'FAILED' if not result.success else ''}"
        f" status = {result.status}, optimality = {result.optimality}, nfev = {result.nfev}, cost = {result.cost}."
        f" Fitted: delta={fitted_params.delta}, mu={fitted_params.mu} rho={fitted_params.rho},"
        f" theta={fitted_params.theta}, zeta={fitted_params.zeta}. RMSE = {rmse}."
    )
    if not result.success:
        log.warning(
            f"{curr.market_data.expiry} contracts slice did not converge and will be dropped"
        )

    new_slice = FitSlice(curr.market_data, fitted_params, result.success)
    return new_slice


def ssvi_diagnostics(ssvi: SSVI) -> SurfaceDiagnostics:
    """
    Runs diagnostic tests on the initial and final fit.

    Args:
        ssvi (SSVI): The fitted SSVI object

    Returns:
        SurfaceDiagnostics: Results of diagnostic functions
    """
    thetas = np.array(list(ssvi.inits.thetas.values()))
    theta_inc, cal_cond_2, upper_bound = surface_cal_condition(
        ssvi.inits.rho, ssvi.inits.eta, thetas
    )
    but_cond_1, but_cond_2 = surface_fly_condition(
        ssvi.inits.rho, ssvi.inits.eta, thetas
    )
    log.info(
        f"Initial fit: Thm 4.1(i) Theta is{' ' if theta_inc else ' NOT '}monotonically increasing in time."
        f" Thm 4.1 (ii) value = {cal_cond_2}, against upper bound {upper_bound}"
        f" Thm 4.2 (i) max value = {max(but_cond_1)}, against strict upper bound 4"
        f" Thm 4.2 (ii) value = {but_cond_2}, against inclusive upper bound 4"
    )
    if ssvi.config.kbp is None:
        kb = (-np.inf, np.inf)
    else:
        mink = ssvi.data.data["logMoneyness"].min()
        maxk = ssvi.data.data["logMoneyness"].max()
        kb = (mink - ssvi.config.kbp, maxk + ssvi.config.kbp)
    next_cross = {
        slice.market_data.expiry: crossedness(slice, ssvi.expiry_slices[idx + 1], kb=kb)
        for idx, slice in enumerate(ssvi.expiry_slices[:-1])
    }
    beta_dict = dict()
    for slice in ssvi.expiry_slices:
        exp = slice.market_data.expiry
        beta_dict[exp] = beta(
            slice.slice_params.rho, slice.slice_params.theta, slice.slice_params.zeta
        )
    log.info(
        f"Final fit: Max crossedness is {max(next_cross.values())}"
        f" Max Beta = {max(beta_dict.values())}"
    )
    return SurfaceDiagnostics(
        ssvi.inits.thetas,
        cal_cond_2,
        upper_bound,
        but_cond_1,
        but_cond_2,
        next_cross,
        beta_dict,
    )


def _fit_all(ssvi: SSVI):
    """
    Fits every slice of a surface in one forward sweep, shortest expiry to
    longest.

    Each slice is refit against its converged/future neighbours, so the backward-looking
    penalty sees an already-fitted `prev` while the forward-looking one sees an
    as-yet-unfitted `next`. According to 5.2, working forward or in reverse
    "seems to make little difference"; no second pass is made.
    Only converged fits are referenced as `prev`, unconverging slices are dropped.

    Mutates the passed SSVI object!
    """
    if ssvi.config.kbp is None:
        kb = (-np.inf, np.inf)
    else:
        mink = ssvi.data.data["logMoneyness"].min()
        maxk = ssvi.data.data["logMoneyness"].max()
        kb = (mink - ssvi.config.kbp, maxk + ssvi.config.kbp)

    for idx, slice in enumerate(ssvi.all_slices):
        if idx + 1 < len(ssvi.all_slices):
            next = ssvi.all_slices[idx + 1]
        else:
            next = None
        prev = None
        if idx > 0:
            for i in range(1, idx + 1):
                cand = ssvi.all_slices[idx - i]
                if cand.converged:
                    prev = cand
                    break

        new_slice = _refit(
            slice, prev, next, config=ssvi.config, d_rate=ssvi.d_rate, kb=kb
        )
        ssvi.all_slices[idx] = new_slice
    ssvi.expiry_slices = [s for s in ssvi.all_slices if s.converged]
    if len(ssvi.expiry_slices) < 2:
        log.error("Less than 2 expiries converged, a surface cannot be constructed")
        raise RuntimeError("Less than 2 expiries converged")
    ssvi.diagnostics = ssvi_diagnostics(ssvi)


def new_fit(data: DataSet) -> SSVI:
    """
    Builds a fitted SSVI surface from a DataSet.

    1. Fits the global square-root SSVI seed (`_first_fit`)
    2. Builds one slice per surviving expiry seeded from it (`SliceParams.from_init`)
    3. Refits each slice against its neighbours with a calendar-arbitrage penalty (`_fit_all`)

    Args:
        data: a DataSet object.

    Raises:
        RuntimeError: propagated from `first_fit` if the global fit fails
            to converge.
        RuntimeError: if less than two expiries survive.

    Returns:
        SSVI: A fitted surface, with `expiry_slices` sorted by expiry.
    """

    exp_slices: list[FitSlice] = []
    inits = _first_fit(data.data, data.config, data.rate)
    log.info(f"Initial square-root SSVI fit: rho={inits.rho} eta={inits.eta}")

    df = data.data
    for exp, slice in df.groupby("expiryDate"):
        exp_stamp = pd.to_datetime(exp)  # type: ignore
        mkt = SliceMarketData.from_df_slice(exp_stamp, slice)
        theta = inits.thetas[exp_stamp]
        params = SliceParams.from_init(inits.rho, theta, inits.eta)
        exp_slices.append(FitSlice(mkt, params, False))
    slices = sorted(exp_slices, key=lambda x: x.market_data.expiry)

    if len(slices) < 2:
        log.error("Less than 2 expiries survived, a surface cannot be constructed")
        raise RuntimeError("Less than 2 expiries survived to fit")

    new = SSVI(data.ticker, data.config, data.rate, data, inits, [], slices)
    _fit_all(new)
    return new
