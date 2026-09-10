import logging
import yfinance as yf
import numpy as np
import pandas as pd
from ssvi_live.models import bsm, reverse_bsm
from ssvi_live.fit_objects import RawData, FitConfig, DataSet

log = logging.getLogger(__name__)

"""
Data-ingestion layer
"""


# sources
def _yfinance_source(ticker_code: str) -> RawData:
    """
    Pulls the full listed option chain and the spot price for `ticker_code` from yFinance.
    This is the only place where the external api is touched.
    """
    ticker = yf.Ticker(ticker_code)
    exps = ticker.options
    chains = {exp: ticker.option_chain(exp) for exp in exps}
    dict_puts = {exp: chain.puts.copy() for exp, chain in chains.items()}
    dict_calls = {exp: chain.calls.copy() for exp, chain in chains.items()}
    if len(dict_calls) == 0 or len(dict_puts) == 0:
        raise RuntimeError(f"No option chain available for {ticker_code}")

    df_calls = pd.concat(dict_calls, names=["expiryDate"])
    df_calls = df_calls.reset_index(level="expiryDate").reset_index(drop=True)
    df_puts = pd.concat(dict_puts, names=["expiryDate"])
    df_puts = df_puts.reset_index(level="expiryDate").reset_index(drop=True)

    spot: float = ticker.fast_info["last_price"]
    log.debug(
        f"Yfinance: Fetched {len(df_puts)+len(df_calls)} {ticker_code} contracts across"
        f" {df_calls['expiryDate'].nunique()} expiries. Spot = {spot}"
    )
    return RawData(ticker_code, pd.Timestamp.now(tz="UTC"), df_calls, df_puts, spot)


# source-invariant pipeline
def _verify_source(calls: pd.DataFrame, puts: pd.DataFrame):
    """
    Raises a keyerror if required columns don't exist in the source.
    """
    call_cols = calls.columns
    put_cols = puts.columns
    required_cols = ["expiryDate", "strike", "bid", "ask", "lastTradeDate"]
    for col in required_cols:
        if not col in call_cols:
            raise KeyError(f"Column '{col}' not in raw call chain data")
        if not col in put_cols:
            raise KeyError(f"Column '{col}' not in raw put chain data")


def _merge_sides(calls: pd.DataFrame, puts: pd.DataFrame) -> pd.DataFrame:
    """
    Merges calls/puts at the same expiries and strikes into one dataframe.
    Requires df to have "expiryDate" and "strike" cols.
    The merge drops strike-expiries which only exist on one side.
    """

    df = pd.merge(calls, puts, on=["expiryDate", "strike"], suffixes=["", "Put"])

    nbc = len(calls) - len(df)
    nbp = len(puts) - len(df)
    if nbc or nbp:
        log.debug(
            f"Merging put/call: Kept {len(df)} expiry-strikes quoted on both sides;"
            f" dropped {nbc} calls and {nbp} puts with no counterpart."
        )

    return df


def _first_filter(
    df: pd.DataFrame, data_timestamp: pd.Timestamp, days_old: int
) -> pd.DataFrame:
    """
    Filters the passed dataframe `df` for junk quotes.
    Requires cols "bid", "ask", "bidPut", "askPut", "lastTradeDate", "lastTradeDatePut"
    Only keeps strike/expiry pairs that satisfy: 1. numeric bid and ask,
    2. strike >0 , bid > 0, ask > bid, 3. Date of last trade within `days_old` of the
    data timestamp; on at least one side of the chain.
    """
    df = df.copy()
    date_cutoff = data_timestamp.normalize() - pd.Timedelta(days=days_old)
    good_calls = (
        (df["bid"].notna())
        & (df["ask"].notna())
        & (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["strike"] > 0)
        & (pd.to_datetime(df["lastTradeDate"], utc=True) >= date_cutoff)
    )
    df.loc[good_calls, "goodCall"] = True
    good_puts = (
        (df["bidPut"].notna())
        & (df["askPut"].notna())
        & (df["bidPut"] > 0)
        & (df["askPut"] > df["bidPut"])
        & (df["strike"] > 0)
        & (pd.to_datetime(df["lastTradeDatePut"], utc=True) >= date_cutoff)
    )
    df.loc[good_puts, "goodPut"] = True
    either_good = df[(df["goodPut"] == True) | (df["goodCall"] == True)].copy()
    return either_good


def _get_dates(
    df: pd.DataFrame, time_stamp: pd.Timestamp, time_scale: float
) -> tuple[pd.Series, pd.Series]:
    """
    returns exp dates as pd.Timestamp and a series of years
    to expiry values scaled by `time_scale`.
    Requires columns "expiryDate"
    All expiry dates are truncated to day units, so an expiry in 4 hours is 0
    time-to-expiry.
    """
    expDate = pd.to_datetime(df["expiryDate"], format="%Y-%m-%d", utc=True)
    tte = (expDate - time_stamp.normalize()).dt.days / time_scale
    return expDate, tte


def _get_init_forward(
    df: pd.DataFrame, spot: float, flntol: int, sbf: float
) -> pd.Series:
    """
    Computes and returns first-guess at the forward price.
    Requires cols "ask", "bid", "askPut", "bidPut", "yearsToExpiry", "strike"

    `flntol` is the minimum number of strikes from which F will be calculated.
    Setting less than 2 will raise.
    `sbf` is the spot boundary factor: the maximum half-distance from spot at which
    a strike is "at the money", as a percentage of the spot.
    Expiries that can't get priced are nan'd for every row, and get
    removed in _final_filter.
    """
    if flntol < 2:
        log.error("flntol set below minimum value.")
        raise RuntimeError("flntol value < 2")
    filter = (
        (df["goodCall"] == True)
        & (df["goodPut"] == True)
        & (df["strike"].between(spot * (1 - sbf), spot * (1 + sbf)))
    )
    filtered = df[filter].copy()

    fwd_0 = dict()
    for exp, slice in filtered.groupby("expiryDate"):
        if len(slice) < flntol:
            fwd_0[exp] = np.nan
            log.warning(
                f"Contracts expiring {exp} do not have enough atm quotes to calculate forward price and will be dropped."
            )
            continue
        mid = (slice["ask"] + slice["bid"]) / 2
        midPut = (slice["askPut"] + slice["bidPut"]) / 2
        cp = mid - midPut
        m, b = np.polyfit(slice["strike"], cp, deg=1)  # type: ignore polyfit() is vectorized
        f = -b / m

        if not np.isfinite(f):
            log.warning(f"F_0 regression failed")
            fwd_0[exp] = np.nan
        elif f <= 0:
            log.warning(f"Contracts expiring {exp} will be dropped: F_0 = {f}")
            fwd_0[exp] = np.nan
        else:
            fwd_0[exp] = f
            log.debug(f"F_0 for expiry {exp}: {f}.")

    f_init = df["expiryDate"].map(fwd_0)
    return f_init


def _get_flat_rate(df: pd.DataFrame, bmf: float) -> tuple[pd.Series, float]:
    """Computes the final forward and the discount rate.
    Requires cols: "ask", "bid", "askPut", "bidPut", "strike", "fInit", "yearsToExpiry"
    bmf is the below the money factor, it controls the wideness of the band about f_0
    to search for the forward.
    """
    filter = (
        (df["goodCall"] == True)
        & (df["goodPut"] == True)
        & (df["strike"].between(df["fInit"] * (1 - bmf), df["fInit"]))
    )
    filtered = df[filter].copy()

    fwd = dict()
    rates = dict()
    for exp, slice in filtered.groupby("expiryDate"):
        mid = (slice["ask"] + slice["bid"]) / 2
        midPut = (slice["askPut"] + slice["bidPut"]) / 2
        cp = mid - midPut
        if len(cp) < 2:
            log.warning(
                f"Contracts expiring {exp} don't have enough quotes to calculate discount rate."
            )
            fwd[exp] = np.nan
            continue
        m, b = np.polyfit(slice["strike"], cp, deg=1)  # type: ignore polyfit() is vectorized
        f = -b / m
        if not np.isfinite(m) or m >= 0 or f <= 0:
            log.warning(f"CP: {exp} contracts will be dropped: f = {f}.")
            fwd[exp] = np.nan
            continue
        yte = slice["yearsToExpiry"].iloc[0]
        rates[exp] = np.log(-1 / m) / yte
        fwd[exp] = f
        log.debug(f"Forward price for expiry {exp}: {f}.")

    forward = df["expiryDate"].map(fwd)
    rate_pd = df["expiryDate"].map(rates)
    if rate_pd.notna().sum() == 0:
        raise ValueError("No smile produced a rate in put-call parity.")
    rdf = pd.DataFrame({"tte": filtered["yearsToExpiry"], "rate": rate_pd})
    rdf = rdf.drop_duplicates(subset=["tte"]).dropna(subset=["rate"])
    time_scaled_rate = (rdf["rate"] * (rdf["tte"] ** 2)).sum()
    time_weighted_rate = time_scaled_rate / (rdf["tte"] ** 2).sum()
    return forward, time_weighted_rate


def _stitch_otm(df: pd.DataFrame) -> tuple[pd.Series, ...]:
    """
    Picks a canonical data source from one side of the chain for
    each strike/expiry so OTM data is always used.
    Requires cols "putData", "bid" + Put, "ask" + Put, "lastTradeDate" + Put
    Returns in order: bid, ask, lastTrade
    """
    put = df["putData"] == True
    call = df["putData"] == False
    bid = pd.concat([df.loc[put, "bidPut"], df.loc[call, "bid"]])
    ask = pd.concat([df.loc[put, "askPut"], df.loc[call, "ask"]])
    last_trade = pd.concat(
        [df.loc[put, "lastTradeDatePut"], df.loc[call, "lastTradeDate"]]
    )

    return bid, ask, last_trade


def _get_prices(df: pd.DataFrame, d_rate: float) -> tuple[pd.Series, ...]:
    """
    Returns mids by backing out IV from bids and asks and taking the mid in IV-space.
    This meaningfully improves the fit. See README.
    Requires a `df` already prepared by `_stitch_otm`, and with columns "forwardPrice",
    "strike" "yearsToExpiry", "bid", "ask", "putData".
    Returns in order: bid_iv, ask_iv, mid_iv, mid
    """
    ask_iv = df.apply(
        lambda row: reverse_bsm(
            row["forwardPrice"],
            row["strike"],
            row["yearsToExpiry"],
            row["ask"],
            d_rate,
            row["putData"],
        ),
        axis=1,
    )

    bid_iv = df.apply(
        lambda row: reverse_bsm(
            row["forwardPrice"],
            row["strike"],
            row["yearsToExpiry"],
            row["bid"],
            d_rate,
            row["putData"],
        ),
        axis=1,
    )
    mid_iv = (ask_iv + bid_iv) / 2

    mid = pd.Series(
        bsm(
            df["forwardPrice"],
            df["strike"],
            df["yearsToExpiry"],
            mid_iv,
            d_rate,
            df["putData"],
        ),
        index=df.index,
    )
    if (n := np.isnan(mid).sum()) > 0:
        log.debug(f"{n} prices could not be bsm inverted.")
    return bid_iv, ask_iv, mid_iv, mid


def _final_filter(
    df: pd.DataFrame,
    data_timestamp: pd.Timestamp,
    days_old: int,
) -> pd.DataFrame:
    """
    1. Refilters on existence and freshness with otm data.
    2. Filters out quotes with nonsensical computed columns.
    3. Removes quotes for which theta_0 cannot be computed in `first_fit`.
    Requires cols "bid", "ask", "invMid", "lastTrade", "yearsToExpiry",
    "forwardPrice", "logMoneyness".
    """
    date_cutoff = data_timestamp.normalize() - pd.Timedelta(days=days_old)
    # check data again
    good_data = (
        (df["bid"] > 0)
        & (df["ask"] > df["bid"])
        & (df["strike"] > 0)
        & (pd.to_datetime(df["lastTrade"], utc=True) >= date_cutoff)
        & (df["yearsToExpiry"] > 0)
    )
    fl = df[good_data].copy()
    log.debug(f"(Final filter: {len(df) - len(fl)} failed data checks.")
    good_calc = (
        (fl["midIV"].notna())
        & (fl["invMid"].notna())
        & (fl["forwardPrice"].notna())
        & (fl["logMoneyness"].notna())
    )
    cfl = fl[good_calc].copy()
    log.debug(f"(Final filter: {len(fl) - len(cfl)} failed computed col checks.")

    # Requires quotes on both sides of the money for each expiry for theta_0 in initial fit
    both_sided_mask = pd.Series(False, index=cfl.index, dtype=bool)
    for exp, slice in cfl.groupby("expiryDate"):
        above = slice["logMoneyness"] > 0
        below = slice["logMoneyness"] < 0
        if above.sum() <= 0 or below.sum() <= 0:
            log.warning(
                f"{exp} contracts dropped: only quoted on one side of the money."
            )
            continue
        both_sided_mask[slice.index] = True

    both_sided = cfl[both_sided_mask].copy()
    return both_sided


# Public API
def pipeline(raw: RawData, config: FitConfig) -> DataSet:
    """
    Turns any RawData object into a DataSet for fitting. `raw` can
    come from `_yfinance_source` or from a hand-built/offline chain.

    1. Validates the chain has the required columns (`_verify_source`)
    2. Merges calls/puts on expiry and strike (`_merge_sides`)
    3. Drops junk quotes (`_first_filter`)
    4. Adds time-to-expiry, and drops 0 tte data (`_get_dates`)
    5. Derives forward price and a flat discount rate (`_get_init_forward`, `_get_flat_rate`)
    6. Stitches together OTM data (`_stitch_otm`), and derives prices, log-moneyness, and total variance.
    7. Applies a final filtering step (`_final_filter`).

    Args:
        raw (RawData): unmerged call/put chains, spot, and collection timestamp.
        config (FitConfig): fitting/data-tolerance configuration.

    Returns:
        (DataSet): the filtered, enriched chain data ready for `fit_ssvi.new_fit`
    """
    # verify, merge, and first filter
    _verify_source(raw.call_chain, raw.put_chain)
    mrg = _merge_sides(raw.call_chain, raw.put_chain)
    fld = _first_filter(mrg, raw.time_stamp, config.days_old)
    log.info(f"First filter: Kept {len(fld)} of {len(mrg)} contracts.")

    # add columns
    fld["expiryDate"], fld["yearsToExpiry"] = _get_dates(
        fld, raw.time_stamp, config.time_scale
    )
    dtd = fld[fld["yearsToExpiry"] > 0].copy()
    log.info(f"{len(fld)-len(dtd)} quotes dropped for tte == 0.")
    dtd["fInit"] = _get_init_forward(dtd, raw.spot, config.flntol, config.sbf)
    dtd["forwardPrice"], rate = _get_flat_rate(dtd, config.bmf)
    log.info(f"Flat Rate = {rate}")
    dtd["logMoneyness"] = np.log(dtd["strike"] / dtd["forwardPrice"])
    dtd["putData"] = dtd["strike"] < dtd["forwardPrice"]
    dtd["bid"], dtd["ask"], dtd["lastTrade"] = _stitch_otm(dtd)
    dtd["bidIV"], dtd["askIV"], dtd["midIV"], dtd["invMid"] = _get_prices(dtd, rate)
    dtd["totalVar"] = dtd["yearsToExpiry"] * dtd["midIV"] ** 2

    # final filter
    final = _final_filter(dtd, raw.time_stamp, config.days_old)
    log.info(f"Final filter: Kept {len(final)} of {len(dtd)} quotes ")

    return DataSet(raw.ticker, config, raw.time_stamp, final, raw.spot, rate)


def get_data_yfinance(ticker_code: str, config: FitConfig) -> DataSet:
    """
    Fetches one ticker's option chain from yfinance and prepares it for SSVI fitting.

    Args:
        ticker_code: yfinance ticker symbol.
        config: fitting/data-tolerance configuration.

    Returns:
        DataSet: the filtered, augmented DataFrame, and market data.
    """

    raw = _yfinance_source(ticker_code)
    log.info("Data fetched from yfinance.")
    data = pipeline(raw, config)
    return data
