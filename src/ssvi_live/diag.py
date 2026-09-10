import numpy as np
from typing import Callable
from ssvi_live.models import natural_svi


def surface_cal_condition(
    rho: float, eta: float, thetas: np.ndarray
) -> tuple[bool, float, float]:
    """
    Evaluates Gatheral's calendar spread arbitrage condition (Theorem 4.1) for a square-root SSVI.
    'The SSVI surface is free of calendar spread arbitrage' iff:
        d_t(theta(t)) >= 0      for t > 0
        d_theta(theta*phi(theta)) <= 1/rho**2 * (1 + sqrt(1-rho**2)) * phi(theta)      for theta>0

    Args:
        rho, eta (float): Fit parameters of a square-root SSVI
        thetas (list[float]): a list of ATM variance terms in ascending order by expiry.

    Returns:
        (bool), (float), (float): whether discrete theta series is monotonically increasing,
        the left-hand value of Gatheral's condition 2 test,
        and right-hand boundary value.
    """
    theta_inc = all(np.diff(thetas) >= 0)

    # condition 2 is trivially true for this surface,
    # but included for completeness
    condition_2_value = eta / 2
    if rho == 0:
        upper_bound = np.inf
    else:
        upper_bound = (1 / (rho**2)) * (1 + np.sqrt(1 - rho**2)) * eta
    return theta_inc, condition_2_value, upper_bound


def surface_fly_condition(
    rho: float, eta: float, thetas: np.ndarray
) -> tuple[np.ndarray, float]:
    """
    Evaluates Gatheral's Theorem 4.2 for a sqrt SSVI.
    'A surface is free of butterfly arbitrage' iff for theta>0:
        condition 1 < 4,
        condition 2 <= 4.

    Args:
        rho, eta (float): Fit parameters of a square-root SSVI
        thetas (list[float]): a list of ATM variance terms in ascending order by expiry.

    Returns:
        (ndArray), (float): condition 1 results over thetas,
        the value of the condition 2 result.
    """
    condition_1_values = np.sqrt(thetas) * eta * (1 + np.abs(rho))
    condition_2_value = eta**2 * (1 + np.abs(rho))
    return condition_1_values, condition_2_value


def natural_g(
    delta: float,
    mu: float,
    rho: float,
    theta: float,
    zeta: float,
) -> Callable[[float], float]:
    """
    Gatheral lemma 2.2: "A slice is free of butterfly arbitrage" iff:
        g(k) >= 0 for all k in r
        lim_{k->inf}(d_+(k)) = -inf

    Args:
        delta, mu, rho, theta, zeta (float): natural svi parameters

    Returns:
        Callable[[float], float]: the function g(k) for a single natural svi curve.
    """

    def closed_nat(k):
        return natural_svi(k, delta, mu, rho, theta, zeta)

    def first_derivative(k):
        return (
            theta
            / 2
            * (
                zeta * rho
                + (zeta * (zeta * (k - mu) + rho))
                / np.sqrt((zeta * (k - mu) + rho) ** 2 - rho**2 + 1)
            )
        )

    def second_derivative(k):
        return (
            theta
            / 2
            * -(zeta**2 * (rho**2 - 1))
            / ((zeta * (k - mu) + rho) ** 2 - rho**2 + 1) ** (3 / 2)
        )

    def g(k):
        return (
            (1 - (k * first_derivative(k) / (2 * closed_nat(k)))) ** 2
            - ((first_derivative(k) ** 2) / 4) * (1 / closed_nat(k) + 1 / 4)
            + second_derivative(k) / 2
        )

    return g


def beta(rho: float, theta: float, zeta: float) -> float:
    """
    A natural svi slice is free of butterfly if beta < 2,
    equivalent to the d limit
    """
    return theta * zeta * (1 + rho) / 2
