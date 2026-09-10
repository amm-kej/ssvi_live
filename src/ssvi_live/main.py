import pandas as pd
import logging
from importlib.resources import files
from ssvi_live.data import get_data_yfinance, pipeline
from ssvi_live.fit_objects import FitConfig, SSVI, RawData
from ssvi_live.fit_ssvi import new_fit
from ssvi_live.plotting import generate_plots

log = logging.getLogger(__name__)


def demo(config: FitConfig | None = None) -> SSVI:
    """
    Runs the `live_fit` pipeline off of the bundled data
    (AMAT chains pulled in 2026-09-03).

    Args:
        config (FitConfig | None, optional):  A config package for
        the data cleaning and surface fit; Defaults to None and
        initializes to default values in the function body.

    Returns:
        SSVI: A fitted surface object which packages together each smile,
        metadata, and the dataset it was fit to.
    """
    if config is None:
        config = FitConfig()

    calls = pd.read_csv(files("ssvi_live") / "example_data" / "raw-calls-example.csv")  # type: ignore
    puts = pd.read_csv(files("ssvi_live") / "example_data" / "raw-puts-example.csv")  # type: ignore
    spot = 435.9100036621094

    raw = RawData("AMAT", pd.Timestamp("2026-09-03"), calls, puts, spot)
    data = pipeline(raw, config)
    surface = new_fit(data)
    generate_plots(surface)
    return surface


def live_fit(ticker: str, config: FitConfig | None = None) -> SSVI:
    """
    Builds a fitted SSVI surface for an equity option chain,
    with data fetched from `yfinance` at the time of running.

    Fetches and filters the chain (`data.get_data_yfinance`, `data.pipeline`),
    fits the global square-root SSVI seed, builds one slice per surviving expiry
    seeded from it, then refits each slice against its neighbours with
    the calendar-arbitrage penalty applied (`fit_ssvi.fit_all`).

    Expiries for which `first_fit` could not compute an ATM total variance are
    dropped before slice construction and do not appear in the result. Smiles
    which fail to converge on the first pass are recorded in attribute `all_slices`,
    but do not contribute to the arbitrage penalty or diagnostics and are not plotted.

    Args:
        ticker: yfinance ticker symbol.
        config: fitting/data configuration; defaults to FitConfig()'s defaults.

    Raises:
        RuntimeError: propagated from `data.get_data` if no chain is available
            for `ticker`, or from `fit_ssvi.first_fit` if the global fit fails
            to converge.

    Returns:
        SSVI: the fitted surface, with `expiry_slices` sorted by expiry.
    """
    if config is None:
        config = FitConfig()

    log.info(f"Fitting {ticker}")
    data = get_data_yfinance(ticker_code=ticker, config=config)
    surface = new_fit(data)
    generate_plots(surface)
    return surface


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("peewee").setLevel(logging.WARNING)  # yfinance dependency
    logging.getLogger("urllib3").setLevel(logging.WARNING)  # yfinance dependency
    demo()
