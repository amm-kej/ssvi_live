from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.stats as scs
from scipy.optimize import brentq


def bsm(
    forward_price: float | pd.Series[float],
    strike: float | pd.Series[float],
    time: float | pd.Series[float],
    iv: float | pd.Series[float] | np.ndarray,
    rfr: float | pd.Series[float],
    is_put: bool | pd.Series[bool],
) -> float | np.ndarray:
    """
    Implements a Black-76 Model (forward discounted by rfr, Euro)
        call = exp(-rfr*T) * [F*N(d1) - K*N(d2)]
        put  = exp(-rfr*T) * [K*N(-d2) - F*N(-d1)]
    for d1 = [log(F/K) + iv^2 * T / 2] / (iv * sqrt(T)),
    d2 = d1 - iv*sqrt(T).

    Fully vectorized.

    Args:
        forward_price (float): forward price of the underlying at expiry.
        strike (float): option strike.
        time (float): years to expiry.
        iv (float): Black-76 implied volatility, annualized.
        rfr (float): annualized risk-free rate, used to discount both legs.
        is_put (bool): price a put (True) or a call (False).

    Returns:
        (float): The option price.
    """

    d1_a = np.log(forward_price / strike) + (0.5 * (iv**2)) * time
    d1_b = iv * np.sqrt(time)
    d1 = d1_a / d1_b
    d2 = d1 - iv * np.sqrt(time)

    coeff = np.where(is_put, -1, 1)

    price_a = -rfr * time
    price_b = strike * scs.norm.cdf(coeff * d2) * np.exp(price_a)
    price_c = forward_price * scs.norm.cdf(coeff * d1) * np.exp(price_a)

    price = np.where(is_put, price_b - price_c, price_c - price_b)
    return float(price) if np.ndim(price) == 0 else price


def reverse_bsm(
    forward_price: float,
    strike: float,
    time: float,
    price: float,
    rfr: float,
    put: bool,
) -> float:
    """
    Backs out the Black-76 implied volatility from the price.
    Uses Brent's method (scipy.optimize.brentq) on the [1e-6, 5] interval,
    to a tolerance of 1e-7

    NOT vectorized due to scipy `brentq`.

    Args:
        forward_price (float): forward price of the underlying at expiry.
        strike (float): option strike.
        time (float): years to expiry.
        price (float): option price.
        rfr (float): annualized risk-free rate, used to discount both legs.
        put (bool): price a put (True) or a call (False).

    Returns:
        float: The implied volatility, or NaN when no root exists on the bracket.
        When used in data.py, filters catch and remove these quotes;
        the warning can be safely ignored.
    """
    err = lambda iv: bsm(forward_price, strike, time, iv, rfr, put) - price
    try:
        return brentq(err, 1e-6, 5.0, xtol=1e-7)  # type: ignore
    except RuntimeError:
        return np.nan
    except ValueError:
        return np.nan


def sqrt_svi(k: float, rho: float, eta: float, theta: float) -> float:
    """
    Square-root SSVI parameterization: total variance as a function of log-moneyness,
    with a power-law ansatz phi(theta) = eta/sqrt(theta).
    Gatheral ex 4.2: w(k, theta) = theta/2 * {1 + rho*phi*k + sqrt((phi*k + rho)^2 + (1 - rho^2))}

    Used only for the fit of initial best-guess parameters(rho, eta). (See 5.2)
    Fully vectorized over `theta` and `k`.

    Args:
        k (float): log-moneyness.
        rho (float): SSVI correlation/skew parameter.
        eta (float): scale in phi(theta) = eta / sqrt(theta).
        theta (float): ATM total implied variance at this expiry (Definition 4.1).

    Returns:
        float: Total implied variance.
    """
    phi = eta / np.sqrt(theta)
    w = (theta / 2) * (1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + (1 - rho**2)))
    return w


def natural_svi(
    k: float,
    delta: float,
    mu: float,
    rho: float,
    theta: float,
    zeta: float,
) -> float:
    """
    SSVI natural parameterization  at a single expiry.

    Gatheral eq 3.2: w(k) = delta + theta/2 * {1 + zeta*rho*(k - mu) + sqrt((zeta*(k - mu) + rho)^2 + (1 - rho^2))}

    Note: `theta` is a variance scale fit to each curve, not the same as "theta" in
    the natural SVI. Gatheral calls this parameter "omega". In each fit, theta is seeded
    by theta_0 from the initial fit, but is then free.

    Fully vectorized over `theta` and `k`.
    Args:
        k (float): log-moneyness.
        delta, mu, rho, theta, zeta (float): smile parameters

    Returns:
        float: Total implied variance.
    """
    return delta + theta / 2 * (
        1 + zeta * rho * (k - mu) + np.sqrt((zeta * (k - mu) + rho) ** 2 + (1 - rho**2))
    )
