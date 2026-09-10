"""
This module was written by LLM and reviewed by hand. It is presentation only:
every number it draws is computed in `fit_ssvi` and `models`.
"""

from __future__ import annotations
import logging
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as pgo
import plotly.subplots as psp
from ssvi_live.diag import natural_g
from ssvi_live.fit_objects import SSVI, SurfaceDiagnostics
from ssvi_live.models import natural_svi

# A threshold is not a series. Every limit line in this module uses one reserved
# status color so a boundary is never mistaken for an expiry.
_LIMIT_LINE = "#b3261e"
# Expiry-independent readouts are captions, not data: they sit below the axis in
# a muted grey and only take the status color when the condition is violated.
_NOTE_COLOR = "rgba(0,0,0,0.62)"
_BETA_LIMIT = 2.0
_SURFACE_LIMIT = 4.0
log = logging.getLogger(__name__)


def _expiry_colors(n: int) -> list[str]:
    """
    One color per expiry, drawn from an ordinal ramp.

    Expiries are ordered, so identity comes from a single-hue ramp rather than a
    cycled categorical palette: past ~8 slices cycling repeats a hue and two
    expiries become indistinguishable. Index 0 is the front month.
    """
    if n < 1:
        return []
    if n == 1:
        return px.colors.sample_colorscale("Viridis", [0.5])  # type: ignore
    return px.colors.sample_colorscale("Viridis", [i / (n - 1) for i in range(n)])  # type: ignore


def _require_diagnostics(fit: SSVI) -> SurfaceDiagnostics:
    """
    Returns `fit.diagnostics`, or raises if the surface was never diagnosed.

    The diagnostic panels are presentation only: they read the values
    `fit_ssvi.ssvi_diagnostics` computed, and recompute nothing, so what is drawn
    is exactly what was asserted.
    """
    if fit.diagnostics is None:
        raise ValueError(
            "Surface carries no diagnostics; call fit_ssvi.ssvi_diagnostics(fit) "
            "before plotting arbitrage panels."
        )
    return fit.diagnostics


def _limit_note(name: str, value: float, limit: float) -> tuple[str, str]:
    """
    One-line "value vs. limit" caption for an expiry-independent condition.

    Returns the text and the color to draw it in: muted when the condition holds,
    the reserved status color when it does not.
    """
    if not np.isfinite(limit):
        return f"{name}   {value:.4g}   (bound unbounded at rho = 0)", _NOTE_COLOR
    holds = value <= limit
    margin = f"{value / limit:.3%} of limit" if limit else "n/a"
    text = f"{name}   {value:.4g} {'<=' if holds else '>'} {limit:.4g}   ({margin})"
    return text, (_NOTE_COLOR if holds else _LIMIT_LINE)


def smile_panels(fit: SSVI) -> pgo.Figure:
    """
    One subplot per expiry: the fitted smile, the observed mid IV, and a bid/ask
    IV error bar per quote. Each subplot title reports that slice's fit RMSE
    (fitted IV vs. observed mid IV, evaluated at the quoted strikes) so fit quality
    is visible at a glance next to the curve itself.

    Args:
        fit: a fitted SSVI surface.

    Returns:
        A faceted go.Figure, one panel per expiry.
    """
    slices = fit.expiry_slices
    n = len(slices)
    cols = min(4, n)
    rows = -(-n // cols)  # ceil division

    titles = []
    for slice_ in slices:
        mkt = slice_.market_data
        fitted_w = natural_svi(mkt.k, **slice_.slice_params.__dict__)  # type: ignore
        fitted_iv = np.sqrt(fitted_w / mkt.yte)
        rmse = float(np.sqrt(np.mean((fitted_iv - mkt.iv) ** 2)))
        titles.append(f"{mkt.expiry.date()}  (RMSE={rmse:.4f})")

    # Subplot titles carry a date and an RMSE, so they are wide relative to a
    # panel; the default horizontal_spacing of 0.2/cols lets adjacent titles
    # collide. Widen the gutters and the panels to give them room.
    fig = psp.make_subplots(
        rows=rows, cols=cols, subplot_titles=titles, horizontal_spacing=0.075
    )

    for idx, slice_ in enumerate(slices):
        row, col = idx // cols + 1, idx % cols + 1
        mkt = slice_.market_data

        k_grid = np.linspace(mkt.k.min(), mkt.k.max(), 200)
        fitted_w = natural_svi(k_grid, **slice_.slice_params.__dict__)  # type: ignore
        fitted_iv = np.sqrt(fitted_w / mkt.yte)

        # observed points + bid/ask band drawn first, muted, so the fitted curve
        # (added after) renders on top instead of getting buried under the error bars.
        # The surface is assembled from OTM quotes: puts below the forward, calls
        # above. One trace per side, so the join at k = 0 is visible rather than
        # inferred -- a kink or level shift there is the put/call parity error.
        for side_name, mask, color in (
            ("observed put (mid)", mkt.isPut, "rgba(31,119,180,0.6)"),
            ("observed call (mid)", ~mkt.isPut, "rgba(255,127,14,0.6)"),
        ):
            if not mask.any():
                continue
            fig.add_trace(
                pgo.Scatter(
                    x=mkt.k[mask],
                    y=mkt.iv[mask],
                    mode="markers",
                    name=side_name,
                    legendgroup=side_name,
                    showlegend=(idx == 0),
                    marker=dict(size=4, opacity=0.6, color=color),
                    error_y=dict(
                        type="data",
                        symmetric=False,
                        array=(mkt.askIV - mkt.iv)[mask],
                        arrayminus=(mkt.iv - mkt.bidIV)[mask],
                        color="rgba(120,120,120,0.3)",
                        thickness=1,
                        width=2,
                    ),
                ),
                row=row,
                col=col,
            )
        fig.add_trace(
            pgo.Scatter(
                x=k_grid,
                y=fitted_iv,
                mode="lines",
                name="fitted",
                legendgroup="fitted",
                showlegend=(idx == 0),
                line=dict(width=2.5),
            ),
            row=row,
            col=col,
        )

    fig.update_layout(
        height=300 * rows,
        width=340 * cols,
        title=f"{fit.ticker}: Smile Fit by Expiry",
    )
    fig.update_xaxes(title_text="Log-Moneyness (k)")
    fig.update_yaxes(title_text="Implied Volatility")
    return fig


def all_smiles(fit: SSVI) -> pgo.Figure:
    """
    Every fitted smile on one axis, overlaid against the observed total variance scatter
    for each expiry, so the whole term structure reads as a single picture.

    Args:
        fit: a fitted SSVI surface.

    Returns:
        A single-axis go.Figure with one fitted-curve/observed-scatter pair per
        expiry, color-matched and legend-grouped by expiry.
    """
    fig = pgo.Figure()
    colors = _expiry_colors(len(fit.expiry_slices))

    for idx, slice_ in enumerate(fit.expiry_slices):
        mkt = slice_.market_data
        color = colors[idx]
        label = str(mkt.expiry.date())

        k_grid = np.linspace(mkt.k.min(), mkt.k.max(), 200)
        fitted_w = natural_svi(k_grid, **slice_.slice_params.__dict__)  # type: ignore
        obs_w = (mkt.iv**2) * mkt.yte

        fig.add_trace(
            pgo.Scatter(
                x=k_grid,
                y=fitted_w,
                mode="lines",
                line=dict(color=color),
                name=label,
                legendgroup=label,
            )
        )
        fig.add_trace(
            pgo.Scatter(
                x=mkt.k,
                y=obs_w,
                mode="markers",
                marker=dict(color=color, size=5),
                name=label,
                legendgroup=label,
                showlegend=False,
            )
        )

    fig.update_layout(
        title=f"{fit.ticker}: Fitted Smiles vs. Observed Total Variance",
        xaxis_title="Log-Moneyness (k)",
        yaxis_title="Total Variance (w)",
    )
    return fig


def initial_fit_diagnostics(fit: SSVI) -> pgo.Figure:
    """
    The seed surface's arbitrage conditions: Gatheral Theorem 4.1 (calendar) and
    Theorem 4.2 (butterfly), evaluated on the global square-root SSVI fit that
    every slice is seeded from.

    Top: the expiry-dependent butterfly condition, one value per ATM total
    variance, drawn on a linear axis that reaches its limit of 4 -- the question
    this panel answers is how much headroom the surface has, so the limit stays
    in frame even when that makes the bars small. The condition is increasing in
    theta, so the longest expiry is the binding one.

    Bottom: the ATM total variance term structure, one point per expiry. This is
    Theorem 4.1's first condition, d(theta_t)/dt >= 0, which is a property of the
    data rather than of the parameterization and so is the calendar condition
    that can actually fail. Points are connected in expiry order because the
    condition is about consecutive differences, not levels; any expiry whose
    theta falls below its predecessor is marked in the limit color.

    Each panel carries its theorem's expiry-independent condition as a caption
    beneath the axis. Neither shares its panel's scale -- Theorem 4.1 (2)'s bound
    scales as 1/rho^2 and runs to thousands against a theta of order 0.1 -- so
    they are reported as values against their limits rather than drawn.

    Args:
        fit: a fitted SSVI surface carrying diagnostics.

    Raises:
        ValueError: if `fit.diagnostics` has not been populated.

    Returns:
        A two-row go.Figure: Theorem 4.2's expiry-dependent butterfly condition
        by expiry, and the ATM total variance term structure behind Theorem
        4.1's monotonicity condition, each captioned with its theorem's
        expiry-independent condition.
    """
    diag = _require_diagnostics(fit)

    # but_cond_1 is parallel to thetas_per_expiry's insertion order; zip before
    # sorting so a value never drifts away from its expiry.
    pairs = sorted(
        zip(diag.thetas_per_expiry.keys(), diag.but_cond_1), key=lambda pair: pair[0]
    )
    labels = [str(pd.Timestamp(exp).date()) for exp, _ in pairs]
    values = [float(v) for _, v in pairs]

    theta_pairs = sorted(diag.thetas_per_expiry.items(), key=lambda pair: pair[0])
    theta_labels = [str(pd.Timestamp(exp).date()) for exp, _ in theta_pairs]
    thetas = [float(theta) for _, theta in theta_pairs]
    # The condition constrains consecutive differences, so a violation belongs to
    # a point and its predecessor; the later of the two carries the mark.
    theta_drops = [i for i in range(1, len(thetas)) if thetas[i] < thetas[i - 1]]

    fig = psp.make_subplots(
        rows=2,
        cols=1,
        subplot_titles=(
            "Theorem 4.2 (1): Butterfly, by Expiry",
            "Theorem 4.1 (1): ATM Total Variance by Expiry",
        ),
        vertical_spacing=0.13,
        row_heights=[0.55, 0.45],
    )

    fig.add_trace(
        pgo.Bar(
            x=labels,
            y=values,
            marker_color=_expiry_colors(len(values)),
            showlegend=False,
            hovertemplate="%{x}<br>condition = %{y:.4f} (limit 4)<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=_SURFACE_LIMIT,
        line_dash="dash",
        line_width=1,
        line_color=_LIMIT_LINE,
        annotation_text="limit = 4",
        annotation_position="bottom left",
        row=1,  # type: ignore
        col=1,  # type: ignore
    )

    fig.add_trace(
        pgo.Scatter(
            x=theta_labels,
            y=thetas,
            mode="lines+markers",
            # The connector carries the shape of the term structure, not an
            # expiry's identity, so it stays neutral and lets the markers read.
            line=dict(color="rgba(0,0,0,0.25)", width=1),
            marker=dict(size=9, color=_expiry_colors(len(thetas))),
            showlegend=False,
            hovertemplate="%{x}<br>theta = %{y:.5f}<extra></extra>",
        ),
        row=2,
        col=1,
    )
    if theta_drops:
        fig.add_trace(
            pgo.Scatter(
                x=[theta_labels[i] for i in theta_drops],
                y=[thetas[i] for i in theta_drops],
                mode="markers",
                marker=dict(
                    size=15,
                    color=_LIMIT_LINE,
                    symbol="x-thin",
                    line=dict(color=_LIMIT_LINE, width=2.5),
                ),
                name="decrease vs. prior expiry",
                hovertemplate="%{x}<br>theta = %{y:.5f} (decrease)<extra></extra>",
            ),
            row=2,
            col=1,
        )

    # Row 1's ticks are hidden (see below), so its caption sits just under the
    # axis; row 2's has to clear a band of rotated date labels first.
    for row, offset, (name, value, limit) in (
        (
            1,
            -0.09,
            ("Theorem 4.2 (2): Butterfly", float(diag.but_cond_2), _SURFACE_LIMIT),
        ),
        (
            2,
            -0.44,
            (
                "Theorem 4.1 (2): Calendar",
                float(diag.cal_cond_2),
                float(diag.cal_cond_2_bound),
            ),
        ),
    ):
        text, color = _limit_note(name, value, limit)
        fig.add_annotation(
            text=text,
            showarrow=False,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=offset,
            xanchor="center",
            yanchor="top",
            font=dict(size=12, color=color),
            row=row,  # type: ignore
            col=1,  # type: ignore
        )

    fig.update_layout(
        height=860,
        width=900,
        title=(f"{fit.ticker}: Initial Square-Root SSVI Fit, Arbitrage Conditions"),
        bargap=0.35,
        margin=dict(b=150),
        # The only legend entry this figure can carry belongs to the theta panel,
        # so it sits beside that panel rather than at the top of the figure.
        legend=dict(yanchor="top", y=0.40, xanchor="left", x=1.01),
    )
    # Expiries are evenly spaced categories here, not points in time: a date axis
    # would space the bars by calendar gap and shrink the front months to hairlines.
    # The tick labels are dates; an axis title as well only collides with the
    # scatter's subplot title below. Both panels carry the same expiries in the
    # same order, so the top axis's tick labels would only repeat the bottom's;
    # dropping them lines the two panels up and frees the strip for the caption.
    fig.update_xaxes(type="category", showticklabels=False, row=1, col=1)
    fig.update_yaxes(
        title_text="Condition Value", range=[0, _SURFACE_LIMIT * 1.1], row=1, col=1
    )
    fig.update_xaxes(type="category", row=2, col=1)
    # Autoscaled, not zero-anchored: this panel exists to make a small decrease
    # visible, and a zero baseline would flatten the term structure against it.
    fig.update_yaxes(title_text="ATM Total Variance (theta)", row=2, col=1)
    return fig


def arbitrage_diagnostics(fit: SSVI) -> pgo.Figure:
    """
    Three-panel arbitrage diagnostic for the fitted slices: crossedness between
    each pair of adjacent expiries (top), the butterfly condition g(k) for every
    slice (middle), and the wing slope beta per slice (bottom).

    A surface free of both forms of arbitrage reads as a flat zero on top, curves
    holding at or above zero in the middle, and every bar clear of the beta = 2
    line at the bottom.

    Slices that failed to converge are absent: `fit.expiry_slices` carries only
    the converged fits, and `fit.all_slices` keeps the rest. Adjacency here is
    therefore adjacency among survivors, which is the same neighbour relation the
    forward sweep penalized against.

    Every slice's g is drawn on one shared grid spanning the widest quoted
    log-moneyness in the surface, not each slice's own strikes: g(k) >= 0 is a
    claim about all of R, so clipping each curve at its own last quote would
    imply the condition stops applying there. Beyond the shared grid the claim
    rests on beta, since g approaches 1/4 - beta^2/16 in the right wing.

    Args:
        fit: a fitted SSVI surface carrying diagnostics.

    Raises:
        ValueError: if `fit.diagnostics` has not been populated.

    Returns:
        A three-row go.Figure: crossedness by adjacent pair, g(k) by expiry, and
        beta by expiry.
    """
    diag = _require_diagnostics(fit)
    slices = fit.expiry_slices
    colors = _expiry_colors(len(slices))

    fig = psp.make_subplots(
        rows=3,
        cols=1,
        subplot_titles=(
            "Definition 5.1: Crossedness Against the Preceding Expiry",
            "Lemma 2.2 (1): g(k) by Expiry",
            "Lemma 2.2 (2): Wing Slope beta by Expiry",
        ),
        vertical_spacing=0.11,
    )
    # vertical_spacing is uniform, but only the top gap needs widening: the
    # crossedness panel carries rotated date labels that crowd the g(k) title,
    # while g(k) below it ends in a short axis title. Lift row 1's floor so the
    # extra clearance goes where it is needed instead of into every gap.
    top_floor, top_ceiling = fig.layout.yaxis.domain  # type: ignore
    fig.update_yaxes(domain=(top_floor + 0.035, top_ceiling), row=1, col=1)

    cal_labels = []
    cal_pairs = []
    cal_gaps = []
    for prev, nxt in zip(slices, slices[1:]):
        # Labelled by the later expiry: the full pair is long enough to collide
        # with the panel below, and it stays available in the hover.
        cal_labels.append(str(nxt.market_data.expiry.date()))
        cal_pairs.append(
            f"{prev.market_data.expiry.date()} -> {nxt.market_data.expiry.date()}"
        )
        cal_gaps.append(float(diag.each_cross[prev.market_data.expiry]))

    fig.add_trace(
        pgo.Bar(
            x=cal_labels,
            y=cal_gaps,
            customdata=cal_pairs,
            marker_color=colors[1:] if len(colors) > 1 else colors,
            showlegend=False,
            hovertemplate="%{customdata}<br>crossedness = %{y:.3e}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    gap_max = max(cal_gaps) if cal_gaps else 0.0
    if gap_max <= 0:
        # Every pair is clean. Left to itself Plotly picks a [-1, 1] range around
        # the flat zero, which reads as a broken panel rather than a passing one.
        fig.add_annotation(
            text="no crossing between adjacent expiries",
            showarrow=False,
            xref="x domain",
            yref="y domain",
            x=0.5,
            y=0.5,
            font=dict(size=12),
            row=1,
            col=1,
        )

    # One grid for every slice, spanning the widest quoted log-moneyness on the
    # surface: g is asserted on all of R, so per-slice clipping would read as the
    # condition lapsing outside each expiry's own strikes.
    all_k = pd.concat([s.market_data.k for s in slices])
    k_grid = np.linspace(all_k.min(), all_k.max(), 400)

    betas = []
    beta_labels = []
    for idx, slice_ in enumerate(slices):
        mkt = slice_.market_data
        exp = mkt.expiry
        label = str(exp.date())

        # g is rebuilt from the slice's own parameters rather than carried on
        # SurfaceDiagnostics: a closure cannot be serialized with the surface.
        # `natural_g` is a pure function of those five parameters, so this is the
        # same curve the diagnostics asserted, not a second opinion.
        g_values = natural_g(*slice_.slice_params.to_tuple())(k_grid)  # type: ignore

        fig.add_trace(
            pgo.Scatter(
                x=k_grid,
                y=g_values,
                mode="lines",
                line=dict(color=colors[idx], width=2),
                name=label,
                legendgroup=label,
                hovertemplate=label + "<br>k = %{x:.3f}<br>g = %{y:.4f}<extra></extra>",
            ),
            row=2,
            col=1,
        )
        betas.append(float(diag.per_exp_beta[exp]))
        beta_labels.append(label)

    fig.add_hline(
        y=0, line_dash="dash", line_width=1, line_color=_LIMIT_LINE, row=2, col=1  # type: ignore
    )

    fig.add_trace(
        pgo.Bar(
            x=beta_labels,
            y=betas,
            marker_color=colors,
            showlegend=False,
            hovertemplate="%{x}<br>beta = %{y:.4g} (limit 2)<extra></extra>",
        ),
        row=3,
        col=1,
    )
    fig.add_hline(
        y=_BETA_LIMIT,
        line_dash="dash",
        line_width=1,
        line_color=_LIMIT_LINE,
        annotation_text="beta = 2",
        annotation_position="bottom left",
        row=3,  # type: ignore
        col=1,  # type: ignore
    )

    fig.update_layout(
        height=1250,
        width=900,
        title=f"{fit.ticker}: Fitted-Slice Arbitrage Diagnostics",
        bargap=0.35,
        # Only the g(k) panel carries a legend, but with one entry per expiry it
        # needs three rows; parked under the last panel it collides with nothing.
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            title_text="Expiry",
        ),
        margin=dict(b=140),
    )
    fig.update_xaxes(
        title_text="Expiry (later leg of pair)", type="category", row=1, col=1
    )
    fig.update_yaxes(
        title_text="Crossedness (Total Variance)",
        # Plotly's default SI-prefix ticks render a numerically-zero residue as
        # "100n"/"50n" with the unit nowhere on the axis, which reads as a large
        # violation. Force an explicit exponent so the scale is unambiguous.
        exponentformat="e",
        tickformat=".1e",
        range=[0, max(gap_max * 1.25, 1e-8)],
        row=1,
        col=1,
    )
    fig.update_xaxes(title_text="Log-Moneyness (k)", row=2, col=1)
    fig.update_yaxes(title_text="g(k)", row=2, col=1)
    fig.update_xaxes(title_text="Expiry", type="category", row=3, col=1)
    if betas and all(b > 0 for b in betas):
        # beta sits one to two decades below its limit, so a linear axis would
        # either flatten the bars or push the limit line off-scale. Log keeps the
        # spread across expiries and the distance to 2 readable together.
        fig.update_yaxes(
            title_text="beta (log scale)",
            type="log",
            range=[np.log10(min(betas) / 3), np.log10(_BETA_LIMIT * 2)],
            row=3,
            col=1,
        )
    else:
        fig.update_yaxes(title_text="beta", row=3, col=1)
    return fig


def surface_3d(
    fit: SSVI, yte_range: tuple[float, float] | None = None
) -> pgo.Figure | None:
    """
    The fitted surface in 3D: each traded expiry's fitted smile, evaluated on a
    shared log-moneyness grid, stacked at its own time-to-expiry to form a ribbon
    mesh. The model is fit independently per expiry with no constraint tying
    adjacent slices together in the T direction (beyond the calendar-arbitrage
    penalty at the quoted strikes), so this is a surface through the fitted
    smiles rather than a continuous-in-T interpolation -- read it as "one curve
    per traded maturity," not as a claim about maturities in between.

    Args:
        fit: a fitted SSVI surface.
        yte_range: optional (min, max) time-to-expiry filter, inclusive. The
            front of the curve carries a 1/T factor in iv = sqrt(w/T), so the
            shortest-dated slices span a far larger vertical range than the
            rest and compress everything behind them. Passing e.g. (0.1, 1.0)
            drops those slices and rescales the z-axis to the remaining band,
            making its structure legible. None (default) plots every slice.

    Raises:
        ValueError: if `yte_range` selects fewer than two slices.

    Returns:
        A go.Figure containing a single go.Surface trace: log-moneyness x
        time-to-expiry x implied vol.
    """
    slices = fit.expiry_slices
    if yte_range is not None:
        lo, hi = yte_range
        slices = [s for s in slices if lo <= s.market_data.yte <= hi]
        if len(slices) < 2:
            log.error(
                f"yte_range {yte_range} selects {len(slices)} slice(s); "
                "a surface needs at least 2."
            )
            return

    all_k = pd.concat([s.market_data.k for s in slices])
    k_grid = np.linspace(all_k.min(), all_k.max(), 200)

    yte = np.array([s.market_data.yte for s in slices])
    iv_matrix = np.array(
        [
            np.sqrt(natural_svi(k_grid, **s.slice_params.__dict__) / s.market_data.yte)  # type: ignore
            for s in slices
        ]
    )

    fig = pgo.Figure(
        data=[pgo.Surface(x=k_grid, y=yte, z=iv_matrix, colorscale="Viridis")]
    )
    suffix = (
        ""
        if yte_range is None
        else f" (T in [{yte_range[0]:g}, {yte_range[1]:g}] Years)"
    )
    fig.update_layout(
        title=f"{fit.ticker}: Fitted SSVI Surface{suffix}",
        scene=dict(
            xaxis_title="Log-Moneyness (k)",
            yaxis_title="Time to Expiry (Years)",
            zaxis_title="Implied Volatility",
        ),
        height=700,
        width=900,
    )
    return fig


def generate_plots(fit: SSVI):
    smile_panels(fit).show()
    all_smiles(fit).show()
    initial_fit_diagnostics(fit).show()
    arbitrage_diagnostics(fit).show()
    full = surface_3d(fit)
    if full is not None:
        full.show()
    limd = surface_3d(fit, yte_range=(0.1, 1.0))
    if limd is not None:
        limd.show()
