"""
MMM pooled-variable splitter.

Replaces one combined media variable in a model output file with the campaigns
it was built from, across every sheet that references it.

    python mmm_splitter.py split OUTPUT.xlsx IMPORT.xlsx           check only
    python mmm_splitter.py split OUTPUT.xlsx IMPORT.xlsx --apply   write it

Diagnostics, for working out how a modelling file is put together:

    python mmm_splitter.py headers OUTPUT.xlsx
    python mmm_splitter.py check-decay OUTPUT.xlsx
    python mmm_splitter.py check-effectiveness OUTPUT.xlsx
    python mmm_splitter.py check-coefficients OUTPUT.xlsx
    python mmm_splitter.py compare ORIGINAL.xlsx RESULT.xlsx

Settings live in config.py, the only file you edit.

The two input files are never modified. The original output file is copied and
every change is made to the copy.

This file is the source. (It was once assembled from separate step modules by
build_single.py; those no longer exist, so edit this file directly.)
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from scipy.signal import lfilter
from scipy.stats import spearmanr

try:
    from config import (SPLITS, UNTOUCHED, STRUCTURE_SHEET, DATASHEET,
                        SPENDS_SHEET, CONTRIB_SHEET, CURVES_SHEET, DATA_SHEET,
                        DATA_DATE_COL, DECAY_IS_RETENTION,
                        ACCEPT_TRANSFORM_MISMATCH, ACCEPT_RAW_MISMATCH,
                        CURVE_MODE, SMOOTHING, RESCALE_TO_POOLED,
                        STRUCTURE_SHEET_ALTERNATIVES)
    import config as _config_module
    # Newer settings are read with a default, so an older config.py still runs.
    BUILD_MISSING_CURVES = getattr(_config_module, "BUILD_MISSING_CURVES", True)
    CURVE_STEPS = int(getattr(_config_module, "CURVE_STEPS", 350))
    COMPOSITE_COLUMNS = dict(getattr(_config_module, "COMPOSITE_COLUMNS", {}))
except ImportError as _err:
    # An older config.py alongside a newer mmm_splitter.py. The two are a
    # matched pair; updating one without the other is the most common way this
    # tool breaks between colleagues.
    if getattr(_err, "name", None) == "config":
        # No config.py at all, rather than an outdated one.
        print("\nSTOPPED. There is no config.py in this folder.")
        print("\n  mmm_splitter.py needs a file named exactly 'config.py'")
        print("  sitting beside it. If you were sent one under another name,")
        print("  rename it to config.py.")
        print("\n  Windows sometimes hides file extensions and saves it as")
        print("  config.py.txt -- turn on View > File name extensions in")
        print("  Explorer to check.\n")
        sys.exit(1)
    print("\nSTOPPED. config.py is missing a setting this version needs:")
    print(f"    {_err}")
    print("\n  Your config.py is older than mmm_splitter.py. Copy the current")
    print("  config.py alongside it -- they are updated as a pair. Your own")
    print("  SPLITS block can be pasted into the new one.\n")
    sys.exit(1)
except SyntaxError as _err:
    # config.py is edited by hand, so a stray bracket is a normal mistake.
    # Python's own message is accurate but unreadable to anyone who does not
    # write Python, so show the offending lines instead.
    print("\nSTOPPED. There is a typing mistake in config.py -- nothing else "
          "was run.\n")
    print(f"  Python says: {_err.msg}")
    if _err.lineno:
        print(f"  It noticed the problem at line {_err.lineno}, but the cause "
              f"is usually a\n  few lines ABOVE that.\n")
        try:
            _lines = Path(_err.filename).read_text().splitlines()
            _lo = max(0, _err.lineno - 9)
            for _i in range(_lo, min(len(_lines), _err.lineno + 1)):
                _mark = ">>" if _i + 1 == _err.lineno else "  "
                print(f"  {_mark} {_i + 1:>4} | {_lines[_i]}")
        except Exception:
            pass
    print("""
  Most common causes:
    - a missing  )  at the end of a Campaign(...) line
    - a missing  ]  or an extra one around the campaigns list
    - a missing comma between two entries
    - smart quotes from Word instead of plain " quotes

  Every Split(...) entry should look exactly like this:

      Split(
          pooled="the combined variable",
          campaigns=[
              Campaign("new name", "raw column", "spend column"),
              Campaign("new name", "raw column", "spend column"),
          ],
      ),
""")
    sys.exit(1)


# Who is reading the messages. The command line points people at commands and
# config.py; the hosted app sets this to "app", and the same findings then
# point at what the app offers instead (its settings, its Diagnose buttons).
INTERFACE = "cli"


def _hint(cli: str, app: str) -> str:
    return app if INTERFACE == "app" else cli


def add_composite_columns(data: pd.DataFrame, spec: dict) -> tuple[pd.DataFrame, dict]:
    """
    Columns a campaign can name that are not on DATA themselves: a composite
    kept whole as one campaign (COMPOSITE_COLUMNS = {name: [DATA columns]}).
    Each is the plain sum of its DATA columns, built here so that everything
    downstream -- the sum check, the allocation, spends, curves -- treats it
    like any other column. A real DATA column of the same name wins.
    """
    made = {}
    data = data.copy()
    for name, parts in (spec or {}).items():
        if name in data.columns:
            continue
        missing = [p for p in parts if p not in data.columns]
        if missing:
            raise ValueError(f"'{name}' is to be built from DATA columns that "
                             f"do not exist: {', '.join(missing)}")
        data[name] = data[list(parts)].apply(pd.to_numeric, errors="coerce") \
            .fillna(0.0).sum(axis=1)
        made[name] = list(parts)
    return data, made


# The structure sheet name exactly as config.py states it. Inputs may switch
# STRUCTURE_SHEET to an alternative for one file; every new Inputs starts again
# from this, so one file's alternative never leaks into the next run.
_CONFIGURED_STRUCTURE_SHEET = STRUCTURE_SHEET


# Column names the code refers to, in the spelling it uses. Exports are not
# consistent about case -- one app version writes 'Period name', another
# 'Period Name' -- so sheets are read through canon_cols, which renames any
# case/whitespace variant to this spelling and returns the map back to the
# file's own text, so what is written keeps exactly the source header.
CANON_COLS = ["Product", "Market", "Period", "Period name", "Variable",
              "Category", "Contribution", "Spend", "Cost", "Actual", "Raw",
              "Transformed"]


def canon_cols(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    wanted = {c.casefold(): c for c in CANON_COLS}
    ren, back = {}, {}
    for c in df.columns:
        key = str(c).strip().casefold()
        if key in wanted and str(c) != wanted[key] and wanted[key] not in ren.values():
            ren[c] = wanted[key]
            back[wanted[key]] = c
    return df.rename(columns=ren), back


def read_canon(*args, **kwargs) -> pd.DataFrame:
    """pd.read_excel, with the standard columns under their standard names."""
    return canon_cols(pd.read_excel(*args, **kwargs))[0]


# ==========================================================================
#  ADSTOCK AND ALLOCATION
# ==========================================================================

"""
Campaign-level decomposition of a pooled MMM media variable.

Applies to models where the pooled variable was built as a PLAIN SUM of
campaign-level volumes, and the transformation order is ADSTOCK -> SATURATION.

    A_t      = adstock(sum_i x_i,t) = sum_i adstock(x_i,t)   [exact: adstock is linear]
    C_t      = beta * f(A_t)                                  [f is NOT decomposable]
    C_i,t    = C_t * (A_i,t / A_t)                            [Aumann-Shapley allocation]

The allocation is exactly additive by construction: sum_i C_i,t == C_t for every t.
Adstock form is pluggable -- any linear, time-invariant filter preserves the identity.
"""

# --------------------------------------------------------------------------
# Adstock
# --------------------------------------------------------------------------

class RecencyKernel:
    """
    The exact carryover weights of a recency-decay variable ('R 10,0% (6)').

    The platform's recency decay is a finite filter of W+1 taps whose weights
    fall progressively below (1-d)^i -- it is NOT a truncated geometric, and
    rebuilding it as one makes the transform check score ~0.97 on a variable
    whose parameters are perfectly right. The taps are recovered from the
    model's own contribution numbers (recover_recency_kernel) and passed
    wherever a max_lag would go; geometric_adstock then applies them as is.
    """

    def __init__(self, taps, label: str = ""):
        self.taps = np.asarray(taps, dtype=float)
        self.label = label

    def __len__(self):
        return len(self.taps)

    def __repr__(self):
        return (f"recovered {len(self.taps)}-tap kernel"
                + (f" for {self.label}" if self.label else ""))


def shift_series(x: np.ndarray, lag: int) -> np.ndarray:
    """Positive lag delays (x_{t-lag}); negative lag is a lead (x_{t+|lag|}).
    Positions the shift vacates are zero."""
    x = np.asarray(x, dtype=float)
    if not lag:
        return x
    if abs(lag) >= len(x):
        return np.zeros_like(x)
    if lag > 0:
        return np.concatenate([np.zeros(lag), x[:-lag]])
    return np.concatenate([x[-lag:], np.zeros(-lag)])


def geometric_adstock(x, decay, peak_lag: int = 0, max_lag=None):
    """
    Geometric carryover: A_t = x_t + decay * A_{t-1}

    peak_lag shifts the impulse response: positive delays it, negative is a
    lead (the platform allows both).
    max_lag truncates the tail; None = infinite (standard recursive form).
    A RecencyKernel in place of max_lag applies those exact taps instead, and
    decay is then only a label.

    Linear and time-invariant, so adstock(a + b) == adstock(a) + adstock(b).
    """
    x = np.asarray(x, dtype=float)
    taps = getattr(max_lag, "taps", None)
    if taps is None and not 0.0 <= decay < 1.0:
        raise ValueError(f"decay must be in [0, 1), got {decay}")

    x = shift_series(x, int(peak_lag or 0))

    if taps is not None:
        return np.convolve(x, taps)[: len(x)]

    if max_lag is None:
        # a_t = x_t + decay * a_{t-1}, starting from zero
        return lfilter([1.0], [1.0, -float(decay)], x)

    weights = decay ** np.arange(max_lag + 1)
    return np.convolve(x, weights)[: len(x)]


ADSTOCK_FORMS = {"geometric": geometric_adstock}


# --------------------------------------------------------------------------
# Allocation
# --------------------------------------------------------------------------

def campaign_shares(
    raw: pd.DataFrame,
    decay: float,
    peak_lag: int = 0,
    max_lag: int | None = None,
    form: str = "geometric",
) -> pd.DataFrame:
    """
    raw: one column per campaign, one row per day, values = raw volume
         (impressions) in the SAME units that were summed to build the pooled
         variable.

    Returns daily allocation shares, one column per campaign, rows summing to
    1.0 -- except on days where every campaign is dark, which sum to 0.0.
    """
    if raw.isna().any().any():
        raise ValueError(
            "NaNs in raw campaign data. Decide explicitly: 0 = campaign dark, "
            "NaN = data missing. They are not the same thing and must not be "
            "silently coerced."
        )
    if (raw < 0).any().any():
        raise ValueError("Negative volumes in raw campaign data.")

    fn = ADSTOCK_FORMS[form]
    adstocked = raw.apply(
        lambda col: fn(col.values, decay, peak_lag=peak_lag, max_lag=max_lag),
        axis=0,
        result_type="broadcast",
    )

    total = adstocked.sum(axis=1)
    # All campaigns dark AND no carryover -> pooled contribution is 0 anyway.
    # Emit zero shares rather than 0/0; the caller allocates 0 to everyone.
    shares = adstocked.div(total.where(total > 0), axis=0).fillna(0.0)
    return shares


def allocate(
    pooled: pd.Series,
    shares: pd.DataFrame,
    tolerance: float = 1e-9,
) -> pd.DataFrame:
    """
    Split a pooled daily series (contribution, spend, impressions -- anything
    additive) across campaigns using the supplied shares.

    Raises if the split fails to reproduce the pooled series.
    """
    if not pooled.index.equals(shares.index):
        raise ValueError("Index mismatch between pooled series and shares.")

    split = shares.mul(pooled, axis=0)

    # Days with zero shares (all dark) must have had nothing to allocate.
    dark = shares.sum(axis=1) == 0
    orphaned = pooled[dark & (pooled.abs() > tolerance)]
    if len(orphaned):
        raise ValueError(
            f"{len(orphaned)} day(s) carry a non-zero pooled value but zero "
            f"campaign volume -- first at {orphaned.index[0]}, value "
            f"{orphaned.iloc[0]:.6g}. The campaign list is incomplete, the "
            f"date alignment is off, or the decay used here does not match "
            f"the fitted model."
        )

    residual = (split.sum(axis=1) - pooled).abs().max()
    if residual > tolerance:
        raise ValueError(f"Additivity broken: max daily residual {residual:.3e}")

    return split


def reconciliation_report(
    pooled: pd.Series,
    split: pd.DataFrame,
    label: str = "",
) -> dict:
    """Numbers to eyeball before anything gets written to the output file."""
    recombined = split.sum(axis=1)
    return {
        "variable": label,
        "days": len(pooled),
        "pooled_total": float(pooled.sum()),
        "split_total": float(recombined.sum()),
        "abs_diff": float(abs(recombined.sum() - pooled.sum())),
        "max_daily_residual": float((recombined - pooled).abs().max()),
        "campaign_totals": split.sum().to_dict(),
        "campaign_share_of_total": (
            split.sum() / split.sum().sum()
        ).round(4).to_dict(),
        "negatives_introduced": bool(
            (split.values < 0).any() and not (pooled.values < 0).any()
        ),
        "nan_introduced": bool(np.isnan(split.values).any()),
    }


# ==========================================================================
#  TRANSFORM VERIFICATION
# ==========================================================================

"""
Verify that the decay and lag read from 'T structure' actually reproduce the
model's contribution -- before they are used to allocate anything.

Because contribution = beta * f(adstock(raw)) and f is monotonic increasing,
adstock(raw) must be a strictly monotonic function of contribution. Spearman
correlation of 1.0 confirms the parameters; anything less means the adstock
being rebuilt here is not the one the model fitted, and every share derived
from it would be wrong.

This catches convention mismatches that are otherwise invisible: decay stored
as half-life rather than a retention rate, normalised adstock, lag applied at
a different point, or a lag column that simply disagrees with the data.
"""

PASS_THRESHOLD = 0.999


def parse_decay(value) -> tuple[float | None, int | None, str]:
    """
    Read a Decay cell that may not be a plain number.

    Modelling software sometimes writes a formatted string such as
    'R 80.0% (30)' instead of 0.8. The percentage is taken as the decay value
    and any bracketed integer is reported as a possible lag or window, but
    nothing is assumed: whatever is parsed is checked against the model's own
    contribution numbers by verify_transform before it is used.

    Returns (value, bracketed number, description of what was read).
    """
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None, None, "empty"
    if isinstance(value, (int, float)):
        return float(value), None, f"number {value}"

    text = str(value).strip()
    try:
        v = float(text.replace(",", "."))
        # A numeric cell of 1.0 is a legitimate decay. A *string* above 1 is
        # almost certainly a percentage that lost its sign.
        if v > 1:
            return v / 100.0, None, f"{v / 100:g} (read {text!r} as a percentage)"
        return v, None, f"number {text}"
    except ValueError:
        pass

    bracket = re.search(r"\((\d+)\)", text)
    extra = int(bracket.group(1)) if bracket else None

    pct = re.search(r"(\d+(?:[.,]\d+)?)\s*%", text)
    if pct:
        v = float(pct.group(1).replace(",", ".")) / 100.0
        return v, extra, f"{v:g} (from the percentage in {text!r})"

    num = re.search(r"(\d+(?:[.,]\d+)?)", text)
    if num:
        v = float(num.group(1).replace(",", "."))
        if v > 1:
            v /= 100.0
        return v, extra, f"{v:g} (from the first number in {text!r})"

    return None, extra, f"unreadable: {text!r}"


def _score(raw: np.ndarray, contribution: np.ndarray,
           decay: float, lag: int, max_lag: int | None = None) -> float:
    """
    Rank correlation between rebuilt adstock and the model's contribution.

    Measured only on days where stored contribution can still tell values
    apart. A burst campaign is dark most of the year; its carryover decays to
    values the sheet rounds to the same figure, producing hundreds of tied
    contributions against hundreds of distinct adstock values. Those ties
    collapse the correlation and say nothing about whether the adstock is
    right. Excluding them measures the fit where the data can actually
    resolve it.

    Absolute value: a negative coefficient (competitor spend, price) makes
    contribution fall as the variable rises, so a perfect fit there is -1.

    A retention rate outside [0, 1) is not a valid carryover -- returned as
    NaN rather than raising, since impossibility under one convention is
    itself evidence about which convention a file uses.
    """
    if not 0.0 <= decay < 1.0:
        return float("nan")
    a = geometric_adstock(raw, decay, peak_lag=lag, max_lag=max_lag)
    if np.allclose(a, a[0]):
        return 0.0

    c = np.asarray(contribution, dtype=float)
    active = c > np.nanmin(c)
    if active.sum() < 30:                      # too little signal to judge
        active = np.ones_like(c, dtype=bool)

    r = spearmanr(a[active], c[active]).statistic
    return 0.0 if np.isnan(r) else abs(float(r))


EXACT_TOLERANCE = 1e-6       # relative, for the exact replication check


def _warm_up(decay: float, lag: int, max_lag) -> int:
    """Days at the start of the datasheet whose transform depends on history
    before it -- excluded from the exact check, since the datasheet does not
    carry that history."""
    taps = getattr(max_lag, "taps", None)
    if taps is not None:
        span = len(taps)
    elif isinstance(max_lag, (int, np.integer)):
        span = int(max_lag) + 1
    elif decay <= 0:
        span = 1
    else:
        span = int(np.ceil(np.log(1e-12) / np.log(decay))) if decay < 1 else 10**6
    return min(abs(int(lag or 0)) + span, 180)


def _exact_error(raw, contribution, decay, lag, max_lag, coefficient,
                 curve, alpha) -> float:
    if not 0.0 <= decay < 1.0:
        return float("nan")
    f = coefficient * model_transform(raw, decay, lag, max_lag, curve, alpha)
    w = _warm_up(decay, lag, max_lag)
    if len(raw) - w < 30:
        w = 0
    scale = float(np.abs(contribution[w:]).max()) or 1.0
    return float(np.abs(f[w:] - contribution[w:]).max()) / scale


def verify_transform(
    raw: pd.Series,
    contribution: pd.Series,
    decay: float,
    lag: int,
    threshold: float = PASS_THRESHOLD,
    search_on_fail: bool = True,
    max_lag: int | None = None,
    coefficient: float | None = None,
    curve: str | None = None,
    alpha: float | None = None,
) -> dict:
    """
    Returns a verdict. On failure, optionally grid-searches decay and lag to
    report what the data actually supports.

    EXACT when the coefficient (and curve and Alpha, for a curved variable)
    are known: rebuild coefficient x curve(adstock(lag(raw))) and compare it
    with the model's contribution, to 1e-6 relative. This is the check that
    means something. The rank-correlation test below is kept only as the
    fallback when those are missing: on an S curve most days carry
    contribution so close to zero that its ranks are floating-point noise, and
    perfectly correct parameters score ~0.97 (seen on two variables of the
    reference export that replicate to 1e-16).
    """
    raw = np.asarray(raw, dtype=float)
    contribution = np.asarray(contribution, dtype=float)
    curved_ok = curve is None or alpha is not None
    if coefficient is not None and curved_ok:
        err = _exact_error(raw, contribution, decay, lag, max_lag,
                           coefficient, curve, alpha)
        result = {
            "method": "exact",
            "declared_decay": decay, "declared_lag": lag, "max_lag": max_lag,
            "score": err, "passed": bool(err <= EXACT_TOLERANCE),
            "threshold": EXACT_TOLERANCE,
        }
        if result["passed"] or not search_on_fail:
            return result
        grid = []
        for L in range(-3, 8):
            for d in np.round(np.arange(0.0, 0.96, 0.01), 2):
                grid.append((_exact_error(raw, contribution, d, L, max_lag,
                                          coefficient, curve, alpha), d, L))
        grid = [g for g in grid if np.isfinite(g[0])]
        grid.sort()
        if grid:
            e, d, L = grid[0]
            result["best_fit"] = {"decay": float(d), "lag": int(L), "score": e}
            result["alternatives"] = [{"decay": float(d), "lag": int(L),
                                       "score": e} for e, d, L in grid[:5]]
        return result

    score = _score(raw, contribution, decay, lag, max_lag)
    result = {
        "method": "rank",
        "declared_decay": decay,
        "declared_lag": lag,
        "max_lag": max_lag,
        "score": round(score, 6),
        "passed": score >= threshold,
        "threshold": threshold,
    }
    if result["passed"] or not search_on_fail:
        return result

    best = []
    for L in range(-3, 8):                    # the platform allows leads
        for d in np.round(np.arange(0.0, 0.96, 0.01), 2):
            best.append((_score(raw, contribution, d, L, max_lag), d, L))
    best.sort(reverse=True)
    top = best[0]
    result["best_fit"] = {"decay": float(top[1]), "lag": int(top[2]),
                          "score": round(top[0], 6)}
    result["alternatives"] = [
        {"decay": float(d), "lag": int(L), "score": round(s, 6)}
        for s, d, L in best[:5]
    ]
    return result


def share_impact(raw, decay_a: float, lag_a: int, window_a,
                 decay_b: float, lag_b: int, window_b) -> float:
    """
    How far apart are two parameter sets, measured where it matters?

    Not by comparing the parameters -- by comparing the daily shares of the
    period total that each produces, since those shares are what the split
    actually uses. Returned as the percentage of the total allocation that
    moves between days.
    """
    raw = np.asarray(raw, dtype=float)
    a = geometric_adstock(raw, decay_a, peak_lag=lag_a, max_lag=window_a)
    b = geometric_adstock(raw, decay_b, peak_lag=lag_b, max_lag=window_b)
    if a.sum() <= 0 or b.sum() <= 0:
        return float("inf")
    # Total variation: half the summed absolute difference between the two
    # share series, i.e. the fraction of the whole allocation that moves from
    # one day to another. Reported as a percentage.
    #
    # Not the largest single-day difference: over a thousand days a typical
    # daily share is itself only ~0.09 percentage points, so a "0.09pp"
    # difference sounds small while being the size of an entire day.
    return float(np.abs(a / a.sum() - b / b.sum()).sum() / 2 * 100)


def campaign_share_impact(campaign_raw: pd.DataFrame,
                          decay_a: float, lag_a: int, window_a,
                          decay_b: float, lag_b: int, window_b) -> float:
    """
    Largest change to any campaign's TOTAL share, in percentage points.

    This is the number that reaches the output. Day-level movement largely
    cancels once each campaign is summed over the period, so a measure taken
    per day overstates how much the parameter choice actually matters.
    """
    def shares(dec, lg, win):
        ad = campaign_raw.apply(
            lambda col: geometric_adstock(col.values.astype(float), dec,
                                          peak_lag=lg, max_lag=win),
            axis=0, result_type="broadcast")
        total = ad.sum().sum()
        return ad.sum() / total * 100 if total else None

    a, b = shares(decay_a, lag_a, window_a), shares(decay_b, lag_b, window_b)
    if a is None or b is None:
        return float("inf")
    return float((a - b).abs().max())


def explain(result: dict, variable: str) -> str:
    if result.get("method") == "exact":
        head = (f"decay={result['declared_decay']:.4g} "
                f"lag={result['declared_lag']}")
        if result["passed"]:
            return (f"Transform check PASSED for '{variable}': {head} rebuilds "
                    f"the model's contribution exactly (relative error "
                    f"{result['score']:.1e})")
        b = result.get("best_fit", {})
        lines = [f"Transform check FAILED for '{variable}'.",
                 f"  T structure says {head}; rebuilding the contribution "
                 f"from it is out by {result['score']:.2e} (relative; need "
                 f"{result['threshold']:.0e})."]
        if b:
            lines += [f"  Closest from the data: decay={b['decay']} "
                      f"lag={b['lag']} (error {b['score']:.2e}).",
                      "",
                      "  If the closest is ~0, the sheet states its parameters "
                      "in a different convention than",
                      "  the one rebuilt here. If nothing is close, the Raw "
                      "series is not the modelled input",
                      "  or the transform has a form this tool does not "
                      "rebuild."]
        return "\n".join(lines)
    if result["passed"]:
        return (f"Transform check PASSED for '{variable}': "
                f"decay={result['declared_decay']} lag={result['declared_lag']} "
                f"(score {result['score']:.6f})")

    b = result.get("best_fit", {})
    lines = [
        f"Transform check FAILED for '{variable}'.",
        f"  T structure says decay={result['declared_decay']} "
        f"lag={result['declared_lag']}, scoring {result['score']:.4f} "
        f"(need {result['threshold']}).",
    ]
    if b:
        lines += [
            f"  Best fit from the data: decay={b['decay']} lag={b['lag']} "
            f"(score {b['score']:.6f}).",
            "",
            "  If the best fit scores ~1.0, the sheet's parameters are stated in a",
            "  different convention than the one rebuilt here -- reconcile before",
            "  splitting. If nothing scores ~1.0, contribution is not a monotonic",
            "  function of this raw series, so 'Raw' may not be the modelled input.",
        ]
    return "\n".join(lines)


# ==========================================================================
#  THE MODEL'S OWN TRANSFORM
# ==========================================================================

"""
What the platform does to a raw series, reproduced: lag -> adstock -> curve.

Needed for two things the shares alone cannot give: recovering a recency
kernel from the model's contribution numbers, and building per-campaign
response curves ("own_curve") from the model rather than from the pooled
curve's picture. Both are checked against the model's own output before use.
"""


def parse_curve(text) -> str | None:
    """'S Curve (20,0%)' -> 'S Curve'; 'Diminishing Returns (80,0%)' ->
    'Diminishing Returns'; blank -> None (a linear variable)."""
    if text is None or (isinstance(text, float) and np.isnan(text)):
        return None
    t = str(text).strip().casefold()
    if t.startswith("s curve"):
        return "S Curve"
    if t.startswith("diminishing"):
        return "Diminishing Returns"
    if not t:
        return None
    raise ValueError(f"Unrecognised response curve {text!r}.")


def apply_curve(a, curve: str | None, alpha: float | None):
    a = np.asarray(a, dtype=float)
    if curve is None:
        return a
    v = float(alpha) * a
    if curve == "S Curve":
        return (v / (1.0 + v)) * (1.0 - np.exp(-v))
    return 1.0 - np.exp(-v)


def invert_curve(y, curve: str | None, alpha: float | None) -> np.ndarray:
    """The adstocked level that produced curve output y. NaN where y cannot
    be inverted reliably (at or beyond saturation)."""
    y = np.asarray(y, dtype=float)
    if curve is None:
        return y.copy()
    out = np.full_like(y, np.nan)
    ok = (y >= 0) & (y < 0.999)
    if curve == "Diminishing Returns":
        out[ok] = -np.log1p(-y[ok]) / alpha
        return out
    # S curve: G(v) = v/(1+v) * (1 - e^-v) is strictly increasing; bisect.
    lo, hi = np.zeros(ok.sum()), np.full(ok.sum(), 1e4)
    target = y[ok]
    for _ in range(200):
        mid = (lo + hi) / 2
        g = (mid / (1 + mid)) * (1 - np.exp(-mid))
        lo, hi = np.where(g < target, mid, lo), np.where(g < target, hi, mid)
    out[ok] = (lo + hi) / 2 / alpha
    return out


def model_transform(x, retention, lag, kernel, curve, alpha) -> np.ndarray:
    return apply_curve(geometric_adstock(x, retention, peak_lag=lag,
                                         max_lag=kernel), curve, alpha)


def recover_recency_kernel(raw, contribution, coefficient: float,
                           curve: str | None, alpha: float | None,
                           lag: int, window: int, label: str = ""):
    """
    The exact taps of a recency-decay variable, solved from the model itself:
    contribution / coefficient is the curve output, the curve inverts with the
    frozen Alpha, and the adstocked level is then a linear filter of the
    lagged raw series -- W+1 unknowns, ~1000 equations.

    Only days whose whole filter span lies inside the datasheet are used, so
    no history before the window is needed. Returns (RecencyKernel | None,
    message). None means the solve did not reproduce the model, and the
    caller falls back to the truncated geometric with a warning.
    """
    raw = np.asarray(raw, dtype=float)
    contribution = np.asarray(contribution, dtype=float)
    if not coefficient:
        return None, "coefficient is zero"
    level = invert_curve(contribution / coefficient, curve, alpha)
    cols = [shift_series(raw, lag + i) for i in range(window + 1)]
    X = np.column_stack(cols)
    n = len(raw)
    usable = np.zeros(n, dtype=bool)
    usable[window + max(lag, 0): n - max(-lag, 0)] = True
    usable &= np.isfinite(level)
    if usable.sum() < 3 * (window + 1):
        return None, f"only {int(usable.sum())} usable days"
    taps, *_ = np.linalg.lstsq(X[usable], level[usable], rcond=None)
    fit = X[usable] @ taps
    scale = float(np.abs(level[usable]).max()) or 1.0
    resid = float(np.abs(fit - level[usable]).max()) / scale
    if resid > 1e-6 or abs(taps[0] - 1.0) > 1e-4:
        return None, (f"solve did not reproduce the model (relative residual "
                      f"{resid:.1e}, first tap {taps[0]:.4f})")
    return RecencyKernel(taps, label), f"relative residual {resid:.1e}"


# ==========================================================================
#  WORKBOOK SCANNER
# ==========================================================================

"""
Step 0 -- find the pooled variable everywhere it appears in the workbook.

Run this before writing handling for any new sheet. It reports, per sheet,
whether the variable sits as a column header (wide format), as values inside a
column (long format), both, or not at all -- and exactly where.

Nothing is hardcoded: header rows are detected, and row and column extents come
from the sheet itself.
"""

@dataclass
class Hit:
    sheet: str
    orientation: str          # 'column_header' | 'row_value' | 'not_found'
    header_row: int | None = None
    column_index: int | None = None
    column_letter: str | None = None
    key_column: str | None = None      # for row_value: the column holding names
    match_count: int = 0
    first_row: int | None = None
    last_row: int | None = None
    total_rows: int | None = None
    total_columns: int | None = None

    def describe(self) -> str:
        if self.orientation == "not_found":
            return f"  {self.sheet:<28} not present"
        if self.orientation == "column_header":
            return (f"  {self.sheet:<28} WIDE  -- column {self.column_letter} "
                    f"(#{self.column_index}) of {self.total_columns}, "
                    f"header row {self.header_row}, {self.total_rows} data rows")
        return (f"  {self.sheet:<28} LONG  -- '{self.key_column}' column "
                f"{self.column_letter}, {self.match_count} rows "
                f"({self.first_row}-{self.last_row}) of {self.total_rows}, "
                f"header row {self.header_row}")


def _detect_header_row(ws, max_scan: int = 15) -> int:
    """
    First row with at least two non-empty string cells starting at column A.
    Handles sheets with a title or blank rows above the headers.
    """
    for r in range(1, min(max_scan, ws.max_row) + 1):
        vals = [ws.cell(row=r, column=c).value
                for c in range(1, min(ws.max_column, 12) + 1)]
        strings = [v for v in vals if isinstance(v, str) and v.strip()]
        if vals and vals[0] is not None and len(strings) >= 2:
            return r
    return 1


def scan_sheet(ws, variable: str) -> Hit:
    # Cells are compared stripped, so the name must be too: exports carry
    # names with a trailing space ('M-X_Inv (ABC) '), which otherwise match
    # nowhere and the safety scan silently reports the variable absent.
    variable = str(variable).strip()
    header_row = _detect_header_row(ws)
    n_cols = ws.max_column
    n_rows = ws.max_row - header_row

    headers = {}
    for c in range(1, n_cols + 1):
        v = ws.cell(row=header_row, column=c).value
        if v is not None:
            headers[str(v).strip()] = c

    if variable in headers:
        c = headers[variable]
        return Hit(ws.title, "column_header", header_row, c,
                   get_column_letter(c), total_rows=n_rows, total_columns=n_cols)

    # Long format: scan each column for cells matching the variable name.
    for c in range(1, n_cols + 1):
        rows = [r for r in range(header_row + 1, ws.max_row + 1)
                if str(ws.cell(row=r, column=c).value).strip() == variable]
        if rows:
            key = ws.cell(row=header_row, column=c).value
            return Hit(ws.title, "row_value", header_row, c,
                       get_column_letter(c),
                       key_column=str(key).strip() if key else None,
                       match_count=len(rows), first_row=rows[0], last_row=rows[-1],
                       total_rows=n_rows, total_columns=n_cols)

    return Hit(ws.title, "not_found", header_row,
               total_rows=n_rows, total_columns=n_cols)


def scan_workbook(path: str | Path, variable: str) -> list[Hit]:
    wb = load_workbook(path, read_only=False, data_only=True)
    return [scan_sheet(wb[name], variable) for name in wb.sheetnames]


def scan_report(path: str | Path, variable: str, handled: set[str] | None = None,
           skipped: set[str] | None = None) -> str:
    """
    handled  sheets the pipeline rewrites
    skipped  sheets where the variable appears but is deliberately left alone
             (recorded so it stays a decision, not an oversight)
    """
    handled = handled or set()
    skipped = skipped or set()
    hits = scan_workbook(path, variable)
    lines = [
        "=" * 72,
        f"SCAN: '{variable}' in {Path(path).name}",
        "=" * 72,
    ]
    for h in hits:
        mark = ""
        if h.orientation != "not_found":
            if h.sheet in handled:
                mark = "  [handled]"
            elif h.sheet in skipped:
                mark = "  [skipped by choice]"
            else:
                mark = "  <-- NEEDS HANDLING"
        lines.append(h.describe() + mark)

    unhandled = [h.sheet for h in hits if h.orientation != "not_found"
                 and h.sheet not in handled and h.sheet not in skipped]
    lines += ["", f"appears on {sum(h.orientation != 'not_found' for h in hits)} "
                  f"sheet(s); {len(unhandled)} not yet handled", "=" * 72]
    return "\n".join(lines)


# ==========================================================================
#  LOADING THE TWO FILES
# ==========================================================================

"""
MMM variable splitter -- Step 1: load and inspect.

Takes the two file paths, opens both read-only, and reports what it found.
Changes nothing.

Assumptions locked in:
  - output file has a sheet called 'T structure', headers on row 1
  - import file has a sheet called 'DATA', headers on row 1
"""

VARIABLE_COL = "Variable"


class Inputs:
    """Both workbooks, opened and inventoried."""

    def __init__(self, output_path: str | Path, import_path: str | Path):
        self.output_path = Path(output_path)
        self.import_path = Path(import_path)

        for label, p in [("Output", self.output_path), ("Import", self.import_path)]:
            if not p.exists():
                raise FileNotFoundError(f"{label} file not found: {p}")
            if p.suffix.lower() not in {".xlsx", ".xlsm"}:
                raise ValueError(f"{label} file is not an Excel workbook: {p.name}")

        self.output_sheets = load_workbook(
            self.output_path, read_only=True, data_only=True
        ).sheetnames
        self.import_sheets = load_workbook(
            self.import_path, read_only=True, data_only=True
        ).sheetnames

        # Some exports call it 'Structure' rather than 'T structure'. Use the
        # first candidate that is present AND carries a 'Variable' column: a
        # sheet of the same name but a different layout is not the same sheet,
        # and misreading it would be worse than not finding it.
        global STRUCTURE_SHEET
        STRUCTURE_SHEET = _CONFIGURED_STRUCTURE_SHEET
        candidates = [STRUCTURE_SHEET] + [
            c for c in STRUCTURE_SHEET_ALTERNATIVES if c != STRUCTURE_SHEET]
        chosen = None
        for cand in candidates:
            if cand not in self.output_sheets:
                continue
            try:
                head = pd.read_excel(self.output_path, sheet_name=cand, nrows=0)
            except Exception:
                continue
            if any(str(c).strip() == VARIABLE_COL for c in head.columns):
                chosen = cand
                break
        if chosen is None:
            present = [c for c in candidates if c in self.output_sheets]
            raise ValueError(
                f"No usable model structure sheet. Looked for "
                f"{', '.join(repr(c) for c in candidates)}."
                + (f"\n{', '.join(repr(p) for p in present)} exists but has "
                   f"no '{VARIABLE_COL}' column, so it is a different kind of "
                   f"sheet." if present else
                   f"\nSheets present: {', '.join(self.output_sheets)}"))
        if chosen != STRUCTURE_SHEET:
            print(f"Using '{chosen}' as the model structure sheet "
                  f"(no '{STRUCTURE_SHEET}' in this file)")
            STRUCTURE_SHEET = chosen
        self._require(DATA_SHEET, self.import_sheets, self.import_path)

        # T structure: full read, it is small.
        self.structure = pd.read_excel(self.output_path, sheet_name=chosen)

        # Header cells often carry trailing spaces or line breaks that are
        # invisible in Excel. Strip them for matching, but keep the original
        # text so the sheet is written back exactly as it was.
        self.structure_headers = {
            str(c).strip(): c for c in self.structure.columns
        }
        self.structure.columns = [str(c).strip() for c in self.structure.columns]
        if VARIABLE_COL not in self.structure.columns:
            raise ValueError(
                f"No '{VARIABLE_COL}' column in '{STRUCTURE_SHEET}'. "
                f"Found: {', '.join(map(str, self.structure.columns[:8]))}..."
            )
        self.model_variables = (
            self.structure[VARIABLE_COL].dropna().astype(str).str.strip().tolist()
        )

        # DATA: headers only for now, the daily rows come later.
        self.data_columns = [
            str(c).strip()
            for c in pd.read_excel(
                self.import_path, sheet_name=DATA_SHEET, nrows=0
            ).columns
        ]

    @staticmethod
    def _require(sheet: str, available: list[str], path: Path) -> None:
        if sheet not in available:
            raise ValueError(
                f"Sheet '{sheet}' not found in {path.name}. "
                f"Sheets present: {', '.join(available)}"
            )

    def data(self) -> pd.DataFrame:
        """The daily raw data. Read on demand, not at load time."""
        df = pd.read_excel(self.import_path, sheet_name=DATA_SHEET)
        df.columns = [str(c).strip() for c in df.columns]
        return df

    def date_column(self) -> str:
        """
        Find the date column on the DATA sheet.

        Import files name it differently from project to project -- Date,
        Period, Day, Kuupäev. Rather than assume, look for the column that
        actually parses as a run of dates. Config can override.
        """
        if DATA_DATE_COL:
            if DATA_DATE_COL not in self.data_columns:
                raise ValueError(
                    f"DATA_DATE_COL is set to '{DATA_DATE_COL}' but that column "
                    f"is not on the '{DATA_SHEET}' sheet.\n"
                    f"Columns found: {', '.join(self.data_columns[:12])}..."
                )
            return DATA_DATE_COL

        df = pd.read_excel(self.import_path, sheet_name=DATA_SHEET, nrows=400)
        df.columns = [str(c).strip() for c in df.columns]

        best, best_n = None, 0
        for col in df.columns:
            s = df[col]
            if s.isna().all():
                continue
            if pd.api.types.is_numeric_dtype(s) and not \
                    pd.api.types.is_datetime64_any_dtype(s):
                # pd.to_datetime reads numbers as epoch nanoseconds, so any
                # numeric column with unique values would pass as "dates".
                continue
            try:
                # Probing every column for dates makes pandas complain about
                # ambiguous formats on the numeric ones. Expected -- silence it.
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    parsed = pd.to_datetime(s, errors="coerce")
            except Exception:
                continue
            n = int(parsed.notna().sum())
            # a real date column parses almost entirely and does not repeat
            if n >= 0.9 * len(s) and parsed.dropna().is_unique and n > best_n:
                best, best_n = col, n

        if best is None:
            raise ValueError(
                f"No date column found on the '{DATA_SHEET}' sheet. "
                + _hint("Set DATA_DATE_COL in config.py to the correct name.\n",
                        "Send this to whoever maintains the tool; the date "
                        "column name has to be set for this project.\n")
                + f"Columns found: {', '.join(self.data_columns[:12])}..."
            )
        return best

    def find_columns(self, substring: str) -> list[str]:
        """DATA columns containing a substring -- for locating campaign series."""
        s = substring.lower()
        return [c for c in self.data_columns if s in c.lower()]

    def report(self, filter_data_cols: str | None = None) -> str:
        lines = [
            "=" * 66,
            "OUTPUT FILE (to be modified)",
            f"  {self.output_path.name}",
            f"  sheets: {', '.join(self.output_sheets)}",
            "",
            f"  '{STRUCTURE_SHEET}' -- {len(self.structure)} rows, "
            f"{len(self.structure.columns)} columns",
            "",
            "  Variables in the model:",
        ]
        for i, v in enumerate(self.model_variables, 1):
            lines.append(f"    {i:>3}. {v}")

        cols = (
            self.find_columns(filter_data_cols)
            if filter_data_cols
            else self.data_columns
        )
        header = (
            f"  '{DATA_SHEET}' -- {len(cols)} of {len(self.data_columns)} columns "
            f"matching '{filter_data_cols}'"
            if filter_data_cols
            else f"  '{DATA_SHEET}' -- {len(self.data_columns)} columns"
        )

        lines += [
            "",
            "=" * 66,
            "IMPORT FILE (source of the replacement variables)",
            f"  {self.import_path.name}",
            f"  sheets: {', '.join(self.import_sheets)}",
            "",
            header,
        ]
        shown = cols if filter_data_cols else cols[:20]
        for c in shown:
            lines.append(f"      {c}")
        if len(cols) > len(shown):
            lines.append(f"      ... and {len(cols) - len(shown)} more")

        lines += [
            "",
            "=" * 66,
            "NEXT: name the variable to drop, and the DATA columns that replace it.",
            "=" * 66,
        ]
        return "\n".join(lines)


def load_inputs(output_path, import_path) -> Inputs:
    return Inputs(output_path, import_path)


# ==========================================================================
#  SHEET: T datasheet variables
# ==========================================================================

"""
Step 2 -- split a pooled variable in 'T datasheet variables'.

Per-column treatment:
  Product, Market   copied from the dropped rows
  Period name       same dates
  Variable          new campaign name
  Category          copied from the dropped rows (same by definition)
  Contribution      ALLOCATED by adstocked share of Raw
  Spend             ACTUAL, from DATA -- not allocated
  Cost              1.01
  Actual            0
  Raw               ACTUAL, from DATA -- not allocated
  Transformed       0

Only Contribution is modelled output and therefore allocated. Spend and Raw
are observed facts and are taken straight from the import file.
"""

DATE_COL = "Period name"

# Columns this step computes. Everything else on the sheet is copied from the
# pooled variable's own rows, so a project with extra columns keeps them and a
# project with different constants keeps those too -- nothing is assumed about
# what a datasheet contains beyond the five fields the split actually touches.
COMPUTED_COLS = ["Variable", "Contribution", "Spend", "Raw"]


def round_preserving_sum(split: pd.DataFrame, target: pd.Series,
                         decimals: int = 2) -> pd.DataFrame:
    """
    Round each row's values so they still sum to the (rounded) target.

    Plain rounding leaves a few cents unaccounted for across thousands of rows.
    Largest-remainder assigns each leftover unit to whichever campaign was
    rounded down hardest, so the daily totals tie out exactly.
    """
    scale = 10 ** decimals
    scaled = split * scale
    floors = np.floor(scaled)
    remainders = scaled - floors
    target_units = np.round(target * scale)

    out = floors.copy()
    deficit = (target_units - floors.sum(axis=1)).round().astype(int)

    for i, n in enumerate(deficit.values):
        if n == 0:
            continue
        order = remainders.iloc[i].sort_values(ascending=(n < 0)).index
        for col in order[: abs(n)]:
            out.iloc[i, out.columns.get_loc(col)] += np.sign(n)

    return out / scale


def sum_tolerance(series) -> float:
    """
    How far apart two daily series may be and still count as equal.

    Relative to the series' own scale. A fixed 1.0 was loose for GRPs (a whole
    GRP on a day that runs 3) and needlessly tight for impressions in the
    millions; one part in a million of the largest day suits both.
    """
    scale = float(np.nanmax(np.abs(np.asarray(series, dtype=float)))) \
        if len(series) else 0.0
    return max(1e-6 * scale, 1e-9)


def verify_plain_sum(pooled_raw: pd.Series, campaign_raw: pd.DataFrame,
                     tolerance: float | None = None,
                     accept_reason: str | None = None,
                     max_accepted_pct: float = 1.0) -> dict | None:
    """
    The pooled variable's Raw series must equal the sum of the campaign Raw
    columns. If it does not, either the wrong DATA columns were named or the
    pooled variable was not built as a plain sum -- both invalidate the split.
    """
    if tolerance is None:
        tolerance = sum_tolerance(pooled_raw)
    recombined = campaign_raw.sum(axis=1)
    diff = (recombined - pooled_raw).abs()
    if diff.max() > tolerance and accept_reason is not None:
        # The pooled Raw is known to be wrong and the campaign columns are the
        # truth. Proceed, but only if the error is small enough that it is
        # plausibly what was described -- and report exactly what it is.
        total = float(pooled_raw.sum())
        shortfall = float((pooled_raw - recombined).sum())
        pct = abs(shortfall) / total * 100 if total else float("inf")
        if pct > max_accepted_pct:
            raise ValueError(
                f"ACCEPT_RAW_MISMATCH lists this variable, but the campaign "
                f"columns differ from the pooled Raw by {pct:.2f}% "
                f"(limit {max_accepted_pct}%). Too large to accept.")
        return {
            "reason": accept_reason,
            "days_affected": int((diff > tolerance).sum()),
            "pooled_total": total,
            "campaign_total": float(recombined.sum()),
            "difference": shortfall,
            "difference_pct": pct,
        }
    if diff.max() > tolerance:
        worst = diff.idxmax()
        raise ValueError(
            f"Campaign Raw columns do not sum to the pooled variable's Raw.\n"
            f"  worst day: index {worst}\n"
            f"  pooled     {pooled_raw.loc[worst]:,.2f}\n"
            f"  campaigns  {recombined.loc[worst]:,.2f}\n"
            f"  days off   {int((diff > tolerance).sum())}\n"
            f"  difference {diff.max():,.2f}\n"
            f"Check the DATA column names, or whether the pooled variable was "
            f"built from something other than a plain sum."
        )


def split_datasheet(
    datasheet: pd.DataFrame,
    data: pd.DataFrame,
    pooled_var: str,
    campaigns: dict[str, dict[str, str]],
    decay: float,
    lag: int = 0,
    data_date_col: str = "Date",
    max_lag: int | None = None,
    accept_raw_mismatch: str | None = None,
    burn_in_days: int | None = 365,
    smoothing: dict | None = None,
    rescale_to_pooled: str | None = None,
    ref_point: str | None = None,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    """
    campaigns maps new variable name -> {'raw': DATA column, 'spend': DATA column}

    Returns (rebuilt datasheet, reconciliation report, allocated contribution).
    The third value is the per-campaign daily contribution, for reuse by any
    other sheet holding contribution -- so every sheet carries the same figures.
    """
    block = datasheet[datasheet["Variable"] == pooled_var].copy()
    if block.empty:
        raise ValueError(f"'{pooled_var}' not found in '{DATASHEET}'.")
    block = block.sort_values(DATE_COL).reset_index(drop=True)

    data = data.copy()
    data[data_date_col] = pd.to_datetime(data[data_date_col])
    block[DATE_COL] = pd.to_datetime(block[DATE_COL])

    # Align DATA to the modelling period, in the datasheet's own date order.
    aligned = (
        block[[DATE_COL]]
        .merge(data, left_on=DATE_COL, right_on=data_date_col, how="left")
    )

    # Carryover on the first days of the model does not start from nothing --
    # it comes from activity before the window opened. Where the import file
    # holds those earlier days, use them to seed the adstock and then discard
    # them. Without this the opening days are understated, and with a lag the
    # first few reconstruct as exactly zero while the model shows contribution.
    period_start = block[DATE_COL].min()
    prior = data[data[data_date_col] < period_start].sort_values(data_date_col)
    if burn_in_days is not None:
        prior = prior[prior[data_date_col]
                      >= period_start - pd.Timedelta(days=burn_in_days)]
    n_prior = len(prior)
    missing = aligned[[c["raw"] for c in campaigns.values()]].isna().any(axis=1)
    if missing.any():
        first = block.loc[missing.idxmax(), DATE_COL]
        raise ValueError(
            f"{missing.sum()} modelling day(s) have no matching row in DATA -- "
            f"first is {first:%Y-%m-%d}. Date coverage or format mismatch."
        )

    campaign_raw = pd.DataFrame(
        {name: aligned[cols["raw"]].astype(float) for name, cols in campaigns.items()}
    )

    # If the pooled variable was smoothed before it went into the model, apply
    # the same smoothing to each campaign. A rolling average is linear, so the
    # smoothed parts add up to the smoothed total -- and the shares then
    # reflect what the model actually saw rather than the raw spikes.
    if smoothing:
        win = int(smoothing.get("window", 1))
        cen = bool(smoothing.get("centred", False))
        campaign_raw = campaign_raw.rolling(win, center=cen,
                                            min_periods=1).mean()
    prior_raw = pd.DataFrame(
        {name: pd.to_numeric(prior[cols["raw"]], errors="coerce").fillna(0.0)
         for name, cols in campaigns.items()}
    ) if n_prior else None
    pooled_raw_series = block["Raw"].astype(float).reset_index(drop=True)

    # A few days adjusted by hand before modelling cannot be decomposed by any
    # rule. Where that has been declared, treat the pooled figure as
    # authoritative on those days and share it out in proportion to what each
    # campaign actually ran.
    rescale_note = None
    if rescale_to_pooled is not None:
        parts = campaign_raw.sum(axis=1)
        off = (parts - pooled_raw_series).abs() > sum_tolerance(pooled_raw_series)
        if off.any():
            n_days = int(off.sum())
            vol_gap = float(abs((parts - pooled_raw_series)[off].sum())
                            / pooled_raw_series.sum()) if pooled_raw_series.sum() else 1.0
            day_share = n_days / len(block)
            if day_share > MAX_RESCALE_DAY_SHARE or vol_gap > MAX_RESCALE_VOLUME:
                raise ValueError(
                    f"RESCALE_TO_POOLED lists '{pooled_var}', but the "
                    f"disagreement is too broad to be a hand adjustment:\n"
                    f"  {n_days} day(s) affected ({day_share:.1%} of the "
                    f"period, limit {MAX_RESCALE_DAY_SHARE:.0%})\n"
                    f"  {vol_gap:.2%} of volume (limit "
                    f"{MAX_RESCALE_VOLUME:.0%})\n"
                    f"This looks like a missing component. "
                    + _hint("Run check-sum.",
                            "Use 'Diagnose: which columns are missing?' for "
                            "this variable."))
            factor = pd.Series(1.0, index=campaign_raw.index)
            live = off & (parts > 0)
            factor[live] = pooled_raw_series[live] / parts[live]
            campaign_raw = campaign_raw.mul(factor, axis=0)
            rescale_note = {
                "reason": rescale_to_pooled,
                "days": n_days,
                "volume_share": vol_gap,
                "first": block[DATE_COL][off.values].min(),
                "last": block[DATE_COL][off.values].max(),
            }

    raw_note = verify_plain_sum(
        pooled_raw_series, campaign_raw,
        accept_reason=accept_raw_mismatch)

    # Adstocked share of the modelling input drives the contribution split.
    # max_lag truncates the carryover after N days. Some software states a
    # recency window alongside the decay ('R 80.0% (30)' = 80% decays away,
    # over 30 days). At high decay this changes nothing; at low decay an
    # untruncated adstock keeps accumulating carryover the model never had.
    seeded = (pd.concat([prior_raw, campaign_raw], ignore_index=True)
              if n_prior else campaign_raw)
    adstocked = seeded.apply(
        lambda c: geometric_adstock(c.values, decay, peak_lag=lag,
                                    max_lag=max_lag),
        axis=0, result_type="broadcast",
    )
    panel_dates = (list(pd.to_datetime(prior[data_date_col])) if n_prior else []) \
        + list(block[DATE_COL])
    campaign_panel = seeded.copy()
    campaign_panel.index = pd.DatetimeIndex(panel_dates)
    if n_prior:
        adstocked = adstocked.iloc[n_prior:].reset_index(drop=True)
    total = adstocked.sum(axis=1)
    shares = adstocked.div(total.where(total > 0), axis=0).fillna(0.0)

    pooled_contrib = block["Contribution"].astype(float).reset_index(drop=True)

    # A day with no carryover anywhere cannot carry contribution -- unless the
    # variable has a reference point. Then contribution is beta*(f(A) - ref):
    # every day carries the same constant -beta*ref, visible on its own on the
    # dark days. That constant belongs to no campaign (it is what the variable
    # is measured against), so it is shared out by each campaign's share of
    # the rest, and only the part above it is split by daily carryover.
    dark = total <= 0
    offset, offset_note = 0.0, None
    if ref_point and (dark & (pooled_contrib.abs() > 1e-6)).any():
        on_dark = pooled_contrib[dark]
        c0 = float(on_dark.median())
        if np.allclose(on_dark, c0, rtol=1e-9, atol=1e-6):
            offset = c0
            offset_note = {"reference_point": ref_point, "offset": c0,
                           "dark_days": int(dark.sum())}
    elif ref_point and not dark.any():
        offset_note = {"reference_point": ref_point, "offset": None,
                       "dark_days": 0}
    orphaned = pooled_contrib[dark & ((pooled_contrib - offset).abs() > 1e-6)]
    if len(orphaned):
        raise ValueError(
            f"{len(orphaned)} day(s) have zero campaign carryover but non-zero "
            f"pooled contribution (first at row {orphaned.index[0]}, "
            f"{orphaned.iloc[0]:,.2f}).\n"
            f"The decay={decay} lag={lag} taken from T structure probably do not "
            f"reproduce this model's contribution.\n"
            + (f"No data before {period_start:%Y-%m-%d} was available in the "
               f"import file to seed\nthe carryover, so the opening days "
               f"reconstruct as zero. If the import file\ncovers earlier "
               f"dates, that would fix this."
               if n_prior == 0 else
               f"{n_prior} day(s) of prior data were used to seed the "
               f"carryover, so this is not\na start-of-period effect.")
        )
    if offset:
        above = shares.mul(pooled_contrib - offset, axis=0)
        weights = above.sum() / above.sum().sum()
        split_contrib = above + np.outer(np.full(len(above), offset),
                                         weights.values)
        split_contrib = pd.DataFrame(split_contrib, columns=shares.columns,
                                     index=shares.index)
    else:
        split_contrib = shares.mul(pooled_contrib, axis=0)

    residual = (split_contrib.sum(axis=1) - pooled_contrib).abs().max()
    if residual > 1e-6:
        raise ValueError(f"Contribution additivity broken: residual {residual:.3e}")

    # Do NOT round. The source file's own precision is whatever it is, and
    # imposing 2 decimals here makes the five parts drift from the pooled
    # total by a few cents across a thousand days. Full precision ties out
    # exactly and matches how the file already stores contribution.
    rounded_residual = residual

    # Start from a copy of the pooled variable's own rows. Every column the
    # split does not compute -- Product, Market, Category, Cost, Actual,
    # Transformed, and anything else a project happens to carry -- is taken
    # from the source rather than assumed, so it stays whatever that file uses.
    new_rows = []
    for name, cols in campaigns.items():
        rows = block.copy().reset_index(drop=True)
        rows["Variable"] = name
        rows["Contribution"] = split_contrib[name].values
        rows["Spend"] = aligned[cols["spend"]].astype(float).values
        rows["Raw"] = campaign_raw[name].values
        new_rows.append(rows)
    new_block = pd.concat(new_rows, ignore_index=True)

    # Rebuild in place: the new block sits where the pooled block was.
    idx = datasheet.index[datasheet["Variable"] == pooled_var]
    before = datasheet.loc[: idx[0] - 1] if idx[0] > 0 else datasheet.iloc[:0]
    after = datasheet.loc[idx[-1] + 1:]
    rebuilt = pd.concat([before, new_block, after],
                        ignore_index=True)[list(datasheet.columns)]

    # Handed to every other sheet that needs contribution, so the numbers are
    # identical everywhere rather than independently recomputed.
    allocated = split_contrib.copy()
    allocated.insert(0, "Period name", block[DATE_COL].values)

    report = {
        "pooled_variable": pooled_var,
        "days": len(block),
        "max_lag": max_lag,
        "reference_point": offset_note,
        "campaign_panel": campaign_panel,
        "burn_in_days_used": n_prior,
        "smoothing": smoothing,
        "rescaled_to_pooled": rescale_note,
        "raw_mismatch_accepted": raw_note,
        "rows_removed": len(block),
        "rows_added": len(new_block),
        "contribution_pooled": round(float(pooled_contrib.sum()), 2),
        "contribution_split": round(float(split_contrib.sum().sum()), 2),
        "max_daily_residual_unrounded": float(residual),
        "max_daily_residual_rounded": float(rounded_residual),
        "spend_pooled": round(float(block["Spend"].sum()), 2),
        "spend_split": round(float(new_block["Spend"].sum()), 2),
        "raw_pooled": round(float(block["Raw"].sum()), 2),
        "raw_split": round(float(new_block["Raw"].sum()), 2),
        "per_campaign": {
            name: {
                "contribution": round(float(split_contrib[name].sum()), 2),
                "spend": round(float(aligned[c["spend"]].sum()), 2),
                "share_of_contribution": round(
                    float(split_contrib[name].sum() / pooled_contrib.sum()), 4),
            }
            for name, c in campaigns.items()
        },
    }
    return rebuilt, report, allocated


# ==========================================================================
#  SHEET: T structure
# ==========================================================================

"""
Step 3 -- replace the pooled row in 'T structure' with one row per campaign.

Column treatment:
  INHERIT    copied unchanged from the pooled row -- the five campaigns sit on
             the same fitted curve, so these are genuinely shared
  RECOMPUTE  derived from the new per-campaign datasheet totals
  BLANK      cannot be split. One coefficient was estimated, with one standard
             error, from one column of the design matrix. There is no honest
             way to produce five.
"""

INHERIT = ["Decay", "Response Curve", "Alpha", "Lag", "Category",
           "Coefficients actual"]

# Columns that are proportional to something campaign-specific, so each
# campaign gets the pooled value scaled by its own share rather than a copy.
# The constant of proportionality never has to be known: anchoring on the
# pooled row's own value cancels it out.
#
#   normalized   is proportional to the variable's total contribution, so the
#                five scale by contribution share and sum back to the pooled
#                value exactly
#   standardized is proportional to the standard deviation of daily
#                contribution, so the five scale by sd ratio -- these do NOT
#                sum to the pooled value, and should not: sd is not additive
SCALED = {
    "Coefficients normalized": "contribution",
    "Coefficients standardized": "sd",
}

BLANK = ["Coefficients SE", "Coefficients 95% CI", "Coefficients t-Stat",
         "Coefficients p-Value", "Coefficients *", "VIF"]

EFFECTIVENESS_CANDIDATES = {
    "contribution_per_raw":    lambda c, s, r: c / r,
    "contribution_per_1k_raw": lambda c, s, r: c / r * 1_000,
    "contribution_per_1m_raw": lambda c, s, r: c / r * 1_000_000,
    "contribution_per_spend":  lambda c, s, r: c / s,
}

# Some formulas are computed over a window rather than the whole model.
EFFECTIVENESS_WINDOWS = ("all", "last12m")

# ROI and CPU look obvious but are not universal: one model divides
# contribution by spend over the whole period, another over the last 12 months,
# another against a different denominator entirely. Writing the wrong one
# produces numbers in a different unit from every other row on the sheet --
# internally consistent, and wrong. So derive it from the file.
# c = contribution, s = spend, v = value: contribution x the per-period
# 'Cost' (the KPI's unit value) summed day by day. Current exports define ROI
# as v / s over the whole period -- confirmed exact (to 1e-15) on every
# variable of the reference export.
RATIO_CANDIDATES = {
    "ROI": {
        "value_per_spend": lambda c, s, v=np.nan: v / s if s else np.nan,
        "contribution_per_spend": lambda c, s, v=np.nan: c / s if s else np.nan,
        "contribution_per_spend_pct":
            lambda c, s, v=np.nan: c / s * 100 if s else np.nan,
        "profit_per_spend": lambda c, s, v=np.nan: (c - s) / s if s else np.nan,
    },
    "CPU": {
        "spend_per_contribution": lambda c, s, v=np.nan: s / c if c else np.nan,
        "spend_per_1k_contribution":
            lambda c, s, v=np.nan: s / c * 1000 if c else np.nan,
    },
}


def detect_ratio(structure: pd.DataFrame, datasheet: pd.DataFrame, col: str,
                 tolerance: float = 0.002,
                 min_match: float = 0.75) -> tuple[str, str] | None:
    """Work out how ROI or CPU is defined, by formula and by time window."""
    if col not in structure.columns or "Variable" not in structure.columns:
        return None
    rows = structure[structure[col].notna()]
    best, best_rate = None, 0.0
    for window in EFFECTIVENESS_WINDOWS:
        totals = _window_totals(datasheet, window)
        for key, fn in RATIO_CANDIDATES[col].items():
            hits = tested = 0
            for _, row in rows.iterrows():
                v = row["Variable"]
                if v not in totals.index:
                    continue
                c, s, val = totals.loc[v, ["Contribution", "Spend", "Value"]]
                if s <= 0 or c == 0:
                    continue
                got, expected = fn(c, s, val), float(row[col])
                if not np.isfinite(got) or not expected:
                    continue
                tested += 1
                if abs(got - expected) / abs(expected) < tolerance:
                    hits += 1
            rate = hits / tested if tested else 0.0
            if rate > best_rate:
                best, best_rate = (key, window), rate
    return best if best_rate >= min_match else None

DATE_COL = "Period name"


def window_start(dates, end=None) -> pd.Timestamp:
    """
    First period of the trailing year that ends at `end`, counted in the
    data's own periods, as the platform counts it: 365 days for a daily
    model, the last 52 weeks (the final week included) for a weekly one, 12
    months for a monthly one. Used for the curve window and every "last 12
    months" figure.
    """
    u = pd.DatetimeIndex(pd.to_datetime(pd.Series(list(dates))).dropna()
                         .unique()).sort_values()
    end = u.max() if end is None else pd.Timestamp(end)
    u = u[u <= end]
    if len(u) < 2:
        return end
    step = float(u.to_series().diff().dt.total_seconds().median()) / 86400.0
    n = max(1, int(CURVE_WINDOW_DAYS // max(step, 1.0)))
    return u[-n] if len(u) >= n else u[0]


def _window_totals(datasheet: pd.DataFrame, window: str) -> pd.DataFrame:
    d = datasheet
    if window == "last12m":
        dates = pd.to_datetime(d[DATE_COL])
        end = dates.max()
        d = d[dates >= window_start(dates, end)]
    d = d.assign(Value=(d["Contribution"] * d["Cost"]) if "Cost" in d.columns
                 else np.nan)
    return d.groupby("Variable")[["Contribution", "Spend", "Raw", "Value"]].sum(
        min_count=1)


def detect_scalable(structure: pd.DataFrame, datasheet: pd.DataFrame,
                    tolerance: float = 0.002,
                    min_match: float = 0.70) -> dict[str, str]:
    """
    Confirm which of the SCALED columns really are proportional to what we
    think, before scaling anything. A column that turns out not to be
    proportional is safer inherited than rescaled on a wrong assumption.
    """
    ok: dict[str, str] = {}
    totals = datasheet.groupby("Variable")["Contribution"].sum()
    sds = datasheet.groupby("Variable")["Contribution"].std()

    for col, basis in SCALED.items():
        if col not in structure.columns:
            continue
        ref = totals if basis == "contribution" else sds
        ratios = []
        for _, row in structure[structure[col].notna()].iterrows():
            v = row["Variable"]
            if v not in ref.index:
                continue
            d, e = float(ref.loc[v]), float(row[col])
            if d and np.isfinite(d) and e:
                ratios.append(e / d)
        a = np.array([x for x in ratios if np.isfinite(x)])
        if len(a) < 5:
            continue
        med = float(np.median(a))
        rate = float(np.mean(np.abs(a - med) / abs(med) < tolerance))
        if rate >= min_match:
            ok[col] = basis
    return ok


def detect_effectiveness(structure: pd.DataFrame, datasheet: pd.DataFrame,
                         tolerance: float = 0.001,
                         min_match: float = 0.75) -> tuple[str, str] | None:
    """
    Work out how Effectiveness is defined, by formula and by time window.

    Returns (formula key, window) or None.

    A file can carry stale Effectiveness values for variables added after the
    column was last calculated, so a formula is accepted on a strong majority
    rather than unanimity -- but the majority must be decisive, and the winner
    is whichever combination the most variables agree on.
    """
    if "Effectiveness" not in structure.columns:
        return None
    if "Variable" not in structure.columns:
        return None

    rows = structure[structure["Effectiveness"].notna()]
    best, best_rate = None, 0.0

    for window in EFFECTIVENESS_WINDOWS:
        totals = _window_totals(datasheet, window)
        for key, fn in EFFECTIVENESS_CANDIDATES.items():
            hits, tested = 0, 0
            for _, row in rows.iterrows():
                v = row["Variable"]
                if v not in totals.index:
                    continue
                c, s, r = totals.loc[v, ["Contribution", "Spend", "Raw"]]
                if r <= 0 or c == 0 or (key.endswith("spend") and s <= 0):
                    continue
                tested += 1
                expected = float(row["Effectiveness"])
                try:
                    got = fn(c, s, r)
                except ZeroDivisionError:
                    continue
                if expected and abs(got - expected) / abs(expected) < tolerance:
                    hits += 1
            rate = hits / tested if tested else 0.0
            if rate > best_rate:
                best, best_rate = (key, window), rate

    return best if best_rate >= min_match else None


def split_structure(
    structure: pd.DataFrame,
    new_datasheet: pd.DataFrame,
    pooled_var: str,
    campaign_names: list[str],
    effectiveness_formula: tuple[str, str] | None = None,
    scalable: dict[str, str] | None = None,
    ratio_formulas: dict[str, tuple[str, str]] | None = None,
) -> tuple[pd.DataFrame, dict]:
    """Returns the rebuilt T structure and a report."""
    idx = structure.index[structure["Variable"] == pooled_var]
    if len(idx) == 0:
        raise ValueError(f"'{pooled_var}' not found in T structure.")
    if len(idx) > 1:
        raise ValueError(f"'{pooled_var}' appears {len(idx)} times in T structure.")
    pooled_row = structure.loc[idx[0]]

    block = new_datasheet[new_datasheet["Variable"].isin(campaign_names)]
    totals = block.groupby("Variable")[["Contribution", "Spend", "Raw"]].sum()

    # Rebuild the pooled variable's own daily series by summing the campaigns,
    # which reproduces it exactly. Needed to anchor the scaled columns.
    scalable = scalable or {}
    pooled_daily = (block.groupby(DATE_COL)["Contribution"].sum()
                    if scalable else None)
    pooled_total = float(totals["Contribution"].sum()) if scalable else 0.0
    pooled_sd = float(pooled_daily.std()) if scalable is not None and \
        pooled_daily is not None else 0.0
    camp_sd = block.groupby("Variable")["Contribution"].std()

    ratio_formulas = ratio_formulas or {}
    ratio_totals = {
        col: _window_totals(block, win)
        for col, (_, win) in ratio_formulas.items()
    }

    fn, eff_totals = None, None
    if effectiveness_formula:
        key, window = effectiveness_formula
        fn = EFFECTIVENESS_CANDIDATES.get(key)
        # Effectiveness may be measured over a different window than ROI/CPU.
        eff_totals = _window_totals(block, window)
    new_rows = []
    for name in campaign_names:
        c, s, r = totals.loc[name, ["Contribution", "Spend", "Raw"]]
        row = {col: np.nan for col in structure.columns}
        row["Variable"] = name
        for col in INHERIT:
            if col in structure.columns:
                row[col] = pooled_row[col]

        for col, basis in scalable.items():
            if col not in structure.columns or pd.isna(pooled_row[col]):
                continue
            if basis == "contribution":
                share = c / pooled_total if pooled_total else 0.0
            else:
                share = (float(camp_sd.loc[name]) / pooled_sd
                         if pooled_sd else 0.0)
            row[col] = float(pooled_row[col]) * share
        for col in BLANK:
            if col in structure.columns:
                row[col] = np.nan
        # ROI and CPU are written only where the file's own definition was
        # identified. Left blank otherwise: a blank cell is obviously missing,
        # a plausible number in the wrong unit is not.
        for col in ("ROI", "CPU"):
            if col not in structure.columns:
                continue
            spec = ratio_formulas.get(col)
            if spec is None:
                row[col] = np.nan
                continue
            key, _ = spec
            t = ratio_totals[col]
            if name not in t.index:
                row[col] = np.nan
                continue
            rc, rs, rv = t.loc[name, ["Contribution", "Spend", "Value"]]
            val = RATIO_CANDIDATES[col][key](rc, rs, rv)
            # full precision, as the file stores it
            row[col] = float(val) if np.isfinite(val) else np.nan
        if fn is not None and "Effectiveness" in structure.columns:
            ec, es, er = eff_totals.loc[name, ["Contribution", "Spend", "Raw"]] \
                if name in eff_totals.index else (0, 0, 0)
            if er > 0 and ec != 0:
                row["Effectiveness"] = round(fn(ec, es, er), 5)
        new_rows.append(row)

    new_block = pd.DataFrame(new_rows)[structure.columns]
    before = structure.loc[: idx[0] - 1] if idx[0] > 0 else structure.iloc[:0]
    after = structure.loc[idx[0] + 1:]
    rebuilt = pd.concat([before, new_block, after], ignore_index=True)

    report = {
        "pooled_variable": pooled_var,
        "rows": f"{len(structure)} -> {len(rebuilt)}",
        "inherited": [c for c in INHERIT if c in structure.columns],
        "scaled": {k: v for k, v in scalable.items() if k in structure.columns},
        "blanked": [c for c in BLANK if c in structure.columns],
        "effectiveness_formula": (
            f"{effectiveness_formula[0]} over {effectiveness_formula[1]}"
            if effectiveness_formula else "not detected -- blanked"),
        "recomputed": {c: f"{k} over {w}" for c, (k, w) in ratio_formulas.items()},
        "not_recomputed": [c for c in ("ROI", "CPU")
                           if c in structure.columns and c not in ratio_formulas],
        "pooled_roi": (round(float(pooled_row["ROI"]), 4)
                       if "ROI" in structure.columns
                       and pd.notna(pooled_row.get("ROI")) else None),
        "campaign_roi": (dict(zip(campaign_names, new_block["ROI"]))
                         if "ROI" in structure.columns else {}),
        "campaign_cpu": (dict(zip(campaign_names, new_block["CPU"]))
                         if "CPU" in structure.columns else {}),
    }
    return rebuilt, report


# ==========================================================================
#  SHEET: (SPENDS DEF)
# ==========================================================================

"""
Step 5 -- '(SPENDS DEF)': wide format, one column per model variable.

Columns A-D are fixed (Product, Market, Period, Period name) and untouched.
From column E onward each column is a model variable holding ACTUAL SPEND,
regardless of what the column name implies -- 'M-TV_TRP_Sale' is TV Sale spend,
not TRPs. So nothing here is allocated: the five campaign columns are read
straight from DATA and dropped in where the pooled column was.

Date alignment uses 'Period name' (the real daily date), not 'Period'
(the software's year-dayofyear string).
"""

FIXED_COLS = ["Product", "Market", "Period", "Period name"]
DATE_COL = "Period name"


def find_header_row(path, sheet_name: str, expect: str = "Product",
                    max_scan: int = 10) -> int:
    """
    Locate the header row. Returns a 0-based index for pandas' `header=`.
    Some sheets carry a title or blank rows above the headers.
    """
    probe = pd.read_excel(path, sheet_name=sheet_name, header=None, nrows=max_scan)
    for i in range(len(probe)):
        if str(probe.iloc[i, 0]).strip() == expect:
            return i
    raise ValueError(
        f"No header row found in '{sheet_name}' -- no cell in column A equals "
        f"'{expect}' within the first {max_scan} rows."
    )


def split_spends(
    spends: pd.DataFrame,
    data: pd.DataFrame,
    pooled_var: str,
    campaigns: dict[str, dict[str, str]],
    data_date_col: str = "Date",
    tolerance: float = 0.01,
    model_period: tuple | None = None,
    rescale_to_pooled: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Replaces the pooled spend column with one column per campaign, in place.

    campaigns maps new variable name -> {'raw': ..., 'spend': DATA column}
    Only the 'spend' entry is used here.
    """
    if pooled_var not in spends.columns:
        raise ValueError(
            f"'{pooled_var}' not a column in '{SPENDS_SHEET}'. "
            f"Variable columns present: "
            f"{[c for c in spends.columns if c not in FIXED_COLS][:10]}..."
        )

    spends = spends.copy()
    spends[DATE_COL] = pd.to_datetime(spends[DATE_COL])
    data = data.copy()
    data[data_date_col] = pd.to_datetime(data[data_date_col])

    spend_cols = {n: c["spend"] for n, c in campaigns.items()}
    aligned = (
        spends[[DATE_COL]]
        .merge(data[[data_date_col] + list(spend_cols.values())],
               left_on=DATE_COL, right_on=data_date_col, how="left")
    )

    missing = aligned[list(spend_cols.values())].isna().any(axis=1)
    if missing.any():
        first = spends.loc[missing.idxmax(), DATE_COL]
        raise ValueError(
            f"{missing.sum()} row(s) in '{SPENDS_SHEET}' have no matching date "
            f"in DATA -- first is {first:%Y-%m-%d}. This sheet may cover a wider "
            f"period than the import file, or the date formats differ."
        )

    # The pooled column must equal the sum of the campaign spends -- but only
    # on days the model actually used. This sheet routinely carries a wider
    # date range than the model, and what the pooled column contained before
    # the model began is outside the scope of a tool that splits a model's
    # variables. Disagreements there are reported, not treated as errors.
    pooled_series = spends[pooled_var].astype(float).values
    recombined = aligned[list(spend_cols.values())].sum(axis=1).values
    diff = np.abs(recombined - pooled_series)

    in_model = np.ones(len(spends), dtype=bool)
    outside_note = None
    if model_period is not None:
        start, end = model_period
        in_model = ((spends[DATE_COL] >= start) & (spends[DATE_COL] <= end)).values
        out_bad = (diff > tolerance) & ~in_model
        if out_bad.any():
            outside_note = {
                "rows": int(out_bad.sum()),
                "total_rows_outside": int((~in_model).sum()),
                "difference": float(
                    (pooled_series - recombined)[out_bad].sum()),
                "first": spends.loc[out_bad, DATE_COL].min(),
                "last": spends.loc[out_bad, DATE_COL].max(),
            }

    checked = diff.copy()
    checked[~in_model] = 0.0
    diff = checked

    # Same treatment as the datasheet: where a few days were adjusted by hand
    # before modelling, the pooled figure is what the model used, so take it as
    # authoritative on those days and share it out in proportion to what each
    # campaign actually ran.
    rescale_note = None
    if rescale_to_pooled is not None and diff.max() > tolerance:
        off = diff > tolerance
        n_days = int(off.sum())
        day_share = n_days / max(int(in_model.sum()), 1)
        vol_gap = (float(abs((recombined - pooled_series)[off].sum())
                         / pooled_series.sum())
                   if pooled_series.sum() else 1.0)
        if day_share > MAX_RESCALE_DAY_SHARE or vol_gap > MAX_RESCALE_VOLUME:
            raise ValueError(
                f"RESCALE_TO_POOLED lists '{pooled_var}', but on "
                f"'{SPENDS_SHEET}' the disagreement is too broad to be a hand "
                f"adjustment:\n  {n_days} day(s) affected "
                f"({day_share:.1%}, limit {MAX_RESCALE_DAY_SHARE:.0%})\n"
                f"  {vol_gap:.2%} of spend (limit {MAX_RESCALE_VOLUME:.0%})")
        factor = np.ones(len(spends))
        live = off & (recombined > 0)
        factor[live] = pooled_series[live] / recombined[live]
        for col in spend_cols.values():
            aligned[col] = aligned[col].astype(float).values * factor
        recombined = aligned[list(spend_cols.values())].sum(axis=1).values
        diff = np.abs(recombined - pooled_series)
        diff[~in_model] = 0.0
        rescale_note = {
            "reason": rescale_to_pooled,
            "days": n_days,
            "volume_share": vol_gap,
            "first": spends.loc[off, DATE_COL].min(),
            "last": spends.loc[off, DATE_COL].max(),
        }

    if diff.max() > tolerance:
        i = int(diff.argmax())
        off = diff > tolerance
        dates = spends[DATE_COL]

        lines = [
            f"Campaign spends do not sum to '{pooled_var}' in "
            f"'{SPENDS_SHEET}'.",
            f"  rows that do not add up: {int(off.sum())} of {len(spends)}",
            f"  worst row: {spends.iloc[i][DATE_COL]:%Y-%m-%d}",
            f"    sheet          {pooled_series[i]:,.2f}",
            f"    sum of parts   {recombined[i]:,.2f}",
            f"    difference     {pooled_series[i] - recombined[i]:,.2f}",
            f"  total difference across all rows: "
            f"{(pooled_series - recombined).sum():,.2f}",
        ]

        # This sheet often covers a wider date range than the model itself.
        # A mismatch confined to rows outside the modelling period means
        # something different from one inside it.
        if model_period is not None:
            start, end = model_period
            inside = (dates >= start) & (dates <= end)
            n_in = int((off & inside).sum())
            n_out = int((off & ~inside).sum())
            lines.append(
                f"  of those, {n_in} fall inside the modelling period "
                f"({start:%Y-%m-%d} to {end:%Y-%m-%d})\n"
                f"  and {n_out} fall outside it")
            if n_in == 0:
                lines.append(
                    "\n  Every mismatch is OUTSIDE the modelling period, so it "
                    "cannot affect\n  the split -- but this sheet's pooled "
                    "column and its parts disagree on\n  days the model never "
                    "saw. Worth understanding before overriding.")
        lines.append("\nCheck the DATA spend columns named in the config.")
        raise ValueError("\n".join(lines))

    # Rebuild column order, five new columns where the pooled one was.
    pos = list(spends.columns).index(pooled_var)
    new_order = (list(spends.columns)[:pos] + list(campaigns)
                 + list(spends.columns)[pos + 1:])

    for name, col in spend_cols.items():
        spends[name] = aligned[col].astype(float).values
    spends = spends.drop(columns=[pooled_var])[new_order]

    report = {
        "sheet": SPENDS_SHEET,
        "outside_model_period": outside_note,
        "rescaled_to_pooled": rescale_note,
        "pooled_column": pooled_var,
        "column_position": pos + 1,
        "columns": f"{len(new_order) - len(campaigns) + 1} -> {len(new_order)}",
        "rows": len(spends),
        "pooled_spend_total": round(float(pooled_series.sum()), 2),
        "split_spend_total": round(float(spends[list(campaigns)].sum().sum()), 2),
        "max_row_difference": round(float(diff.max()), 4),
        "per_campaign_total": {
            n: round(float(spends[n].sum()), 2) for n in campaigns
        },
    }
    return spends, report


# ==========================================================================
#  SHEET: Individual contributions
# ==========================================================================

"""
Step 6 -- 'Individual contributions': wide format, one column per model
variable, holding daily CONTRIBUTION.

Same shape as '(SPENDS DEF)' -- columns A-D fixed, variables from column E,
pooled column can sit anywhere -- but the values are different in kind.

'(SPENDS DEF)' holds spend, which is observed fact read from the import file.
This sheet holds contribution, which is modelled output and has to be
allocated. So nothing is recomputed here: the allocation produced for
'T datasheet variables' is reused verbatim, which is what keeps the two sheets
agreeing to the cent instead of drifting apart through independent rounding.
"""

FIXED_COLS = ["Product", "Market", "Period", "Period name"]
DATE_COL = "Period name"


def split_contributions(
    contributions: pd.DataFrame,
    allocated: pd.DataFrame,
    pooled_var: str,
    campaign_names: list[str],
    tolerance: float = 1e-6,
    sheet_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    sheet_label = sheet_name or CONTRIB_SHEET
    """
    contributions : the sheet, as read
    allocated     : per-campaign daily contribution from split_datasheet,
                    with a 'Period name' column
    """
    if pooled_var not in contributions.columns:
        raise ValueError(
            f"'{pooled_var}' not a column in '{sheet_label}'. Variable columns "
            f"present: {[c for c in contributions.columns if c not in FIXED_COLS][:10]}..."
        )
    missing_names = [n for n in campaign_names if n not in allocated.columns]
    if missing_names:
        raise ValueError(f"Allocation is missing campaigns: {missing_names}")

    contributions = contributions.copy()
    contributions[DATE_COL] = pd.to_datetime(contributions[DATE_COL])
    allocated = allocated.copy()
    allocated[DATE_COL] = pd.to_datetime(allocated[DATE_COL])

    aligned = contributions[[DATE_COL]].merge(
        allocated[[DATE_COL] + campaign_names], on=DATE_COL, how="left")

    gaps = aligned[campaign_names].isna().any(axis=1)
    if gaps.any():
        first = contributions.loc[gaps.idxmax(), DATE_COL]
        raise ValueError(
            f"{gaps.sum()} row(s) in '{sheet_label}' have no allocated "
            f"contribution -- first is {first:%Y-%m-%d}. This sheet's date range "
            f"differs from 'T datasheet variables'."
        )

    # The pooled column here must match the pooled contribution the allocation
    # was derived from. If it does not, the two sheets disagree at source and
    # splitting would bake that disagreement in.
    pooled_here = contributions[pooled_var].astype(float).values
    allocated_sum = aligned[campaign_names].sum(axis=1).values
    diff = np.abs(allocated_sum - pooled_here)
    if diff.max() > tolerance:
        i = int(diff.argmax())
        raise ValueError(
            f"'{pooled_var}' in '{sheet_label}' does not match the allocated "
            f"contribution.\n"
            f"  worst row: {contributions.iloc[i][DATE_COL]:%Y-%m-%d}\n"
            f"  this sheet {pooled_here[i]:,.4f}\n"
            f"  allocation {allocated_sum[i]:,.4f}\n"
            f"  max diff   {diff.max():,.4f}\n"
            f"This sheet and 'T datasheet variables' hold different contribution "
            f"figures for the same variable."
        )

    pos = list(contributions.columns).index(pooled_var)
    new_order = (list(contributions.columns)[:pos] + campaign_names
                 + list(contributions.columns)[pos + 1:])

    for name in campaign_names:
        contributions[name] = aligned[name].values
    contributions = contributions.drop(columns=[pooled_var])[new_order]

    report = {
        "sheet": sheet_label,
        "pooled_column": pooled_var,
        "column_position": pos + 1,
        "columns": f"{len(new_order) - len(campaign_names) + 1} -> {len(new_order)}",
        "rows": len(contributions),
        "pooled_contribution_total": round(float(pooled_here.sum()), 2),
        "split_contribution_total": round(
            float(contributions[campaign_names].sum().sum()), 2),
        "max_row_difference": round(float(diff.max()), 6),
        "reused_allocation": True,
        "per_campaign_total": {
            n: round(float(contributions[n].sum()), 2) for n in campaign_names
        },
    }
    return contributions, report


WEEKLY_CONTRIB_SHEET = "Individual weekly contributions"


def weekly_coverage(daily: pd.DataFrame, weekly: pd.DataFrame) -> list:
    """
    Which days each row of 'Individual weekly contributions' sums, read from
    the two sheets rather than assumed.

    Each row is dated by the first day of its week and runs to the day before
    the next row's date. The LAST row is the one exports disagree on: one
    version sums to the model's final day, another stops short and leaves the
    final day or two out of the weekly sheet altogether. So its end is found
    by trying each possible end date and keeping the one that reproduces that
    row for every variable on both sheets. Every earlier week is checked the
    same way. Returns [(start, end), ...] in the sheet's row order.
    """
    d = daily.copy()
    d[DATE_COL] = pd.to_datetime(d[DATE_COL])
    d = d.sort_values(DATE_COL)
    starts = list(pd.to_datetime(weekly[DATE_COL]))
    cols = [c for c in weekly.columns if c in d.columns and c not in FIXED_COLS
            and pd.api.types.is_numeric_dtype(weekly[c])
            and pd.api.types.is_numeric_dtype(d[c])]
    W = weekly[cols].to_numpy(dtype=float)
    scale = max(float(np.nanmax(np.abs(W))) if W.size else 0.0, 1.0)
    tol = 1e-9 * scale + 1e-6

    def summed(a, b):
        m = (d[DATE_COL] >= a) & (d[DATE_COL] <= b)
        return d.loc[m, cols].to_numpy(dtype=float).sum(axis=0)

    out = []
    for i, a in enumerate(starts):
        if i + 1 < len(starts):
            b = starts[i + 1] - pd.Timedelta(days=1)
            ends = [b]
        else:
            ends = [x for x in d[DATE_COL] if x >= a][::-1]   # longest first
        hit = next((b for b in ends
                    if np.nanmax(np.abs(summed(a, b) - W[i])) <= tol), None)
        if hit is None:
            raise ValueError(
                f"'{WEEKLY_CONTRIB_SHEET}': the week starting {a:%Y-%m-%d} is "
                f"not the sum of any run of days on '{CONTRIB_SHEET}', so its "
                f"rows cannot be split from the daily figures.")
        out.append((a, hit))
    return out


def weekly_allocation(allocated: pd.DataFrame, coverage: list) -> pd.DataFrame:
    """
    Daily allocated contribution summed into the weekly sheet's own rows,
    using the day ranges weekly_coverage found. Days the weekly sheet leaves
    out are left out here too.
    """
    days = pd.to_datetime(allocated[DATE_COL]).to_numpy()
    rows = []
    for a, b in coverage:
        m = (days >= np.datetime64(a)) & (days <= np.datetime64(b))
        r = allocated.loc[m].drop(columns=[DATE_COL]).sum()
        r[DATE_COL] = a
        rows.append(r)
    out = pd.DataFrame(rows)
    return out[[DATE_COL] + [c for c in out.columns if c != DATE_COL]] \
        .reset_index(drop=True)


# ==========================================================================
#  SHEET: Response Curves
# ==========================================================================

"""
Step 7 -- 'Response Curves'.

Layout (all extents discovered, never assumed):
  row 1              headers
  next 3*N rows      summary block: three groups of N rows -- Average, Maximum,
                     Diminishing Point -- one row per channel, identified ONLY
                     by position. Columns C/D carry that channel's pressure and
                     spend at the point.
  remaining rows     the curves, 251 points per channel

  Each channel is 5 columns [pressure, spend, <bare>, Efficiency, Marginal
  Efficiency]. The FIRST channel has only 3, borrowing the global ' pressure'
  and ' spend' in C/D.

How the split works
-------------------
The pooled channel has one fitted curve. Copying it five times would let an
optimiser believe five channels can each deliver the pooled channel's full
response -- five times the effect the model supports. So the curve is scaled
down instead:

    pressure_i = w_i * pooled pressure
    response_i = w_i * pooled response
    spend_i    = pressure_i * (that campaign's own cost per unit)

with w_i the campaign's share of allocated contribution. The five response
curves then sum to the pooled curve at every point, so the achievable maximum
across them equals the pooled maximum exactly.

The campaigns still differ, because spend uses each campaign's OWN cost per
unit: a cheaper campaign sits higher on the efficiency curve. That is a real
difference. What they do NOT differ in is shape -- the model estimated one
saturation curve, so no campaign saturates faster than another.

Maximum and Diminishing Point scale with w_i, so their definitions never need
to be reverse-engineered. Average is recomputed from each campaign's own
last-12-month observed volume, which is the one place the campaigns' differing
spend levels enter directly.
"""

FIXED = ["Product", "Market", " pressure", " spend"]
_FIXED_KEYS = {"product", "market", "pressure", "spend"}


def _is_fixed(h: str) -> bool:
    """Product, Market and the sheet's shared pressure/spend pair. App
    versions differ on the pair's spelling (' pressure' / ' spend' in one,
    'pressure' / 'Spend' in another), so match ignoring case and spaces."""
    return str(h).strip().casefold() in _FIXED_KEYS


def global_columns(headers: list[str]) -> tuple[str, str]:
    """The shared pressure and spend headers, as this file spells them."""
    gp = next((h for h in headers if str(h).strip().casefold() == "pressure"),
              None)
    gs = next((h for h in headers if str(h).strip().casefold() == "spend"),
              None)
    if (gp is None or gs is None) and len(headers) > 3 and \
            [str(h).strip().casefold() for h in headers[:2]] == ["product", "market"]:
        # Labels not recognisable: the shared pair is columns 3 and 4.
        gp, gs = gp or headers[2], gs or headers[3]
    return gp or GLOBAL_PRESSURE, gs or GLOBAL_SPEND
SUMMARY_COLS = ["Average", "Maximum", "Diminishing Point"]
GLOBAL_PRESSURE, GLOBAL_SPEND = " pressure", " spend"

SUFFIXES = [" Marginal Efficiency", " Efficiency", " pressure", " spend"]


@dataclass
class Block:
    name: str
    cols: dict[str, str] = field(default_factory=dict)   # sub -> column header
    labels: dict[str, str] = field(default_factory=dict)  # sub -> suffix as written
    first_col: int = 0
    last_col: int = 0

    @property
    def borrows_global(self) -> bool:
        return "pressure" not in self.cols


# Variable names known to be on the curve sheet: the model's own, plus the
# campaigns a run is about to add. Set by cmd_split; lets parse_blocks fall
# back to reading blocks by position when the labels are not recognisable.
KNOWN_VARIABLES: set[str] = set()


def _positional_blocks(headers: list[str], known: set[str]) -> list[Block] | None:
    """
    Blocks read by position: each channel is five consecutive columns,
    pressure, spend, the bare KPI column (named exactly as the variable),
    Efficiency, Marginal Efficiency; the first channel's pressure/spend are
    the sheet's shared pair. Anchored on the bare columns, so it does not
    care how the labels are spelled -- including exports or anonymised copies
    where they are tokens. None if the layout does not fit exactly.
    """
    known_s = {k.strip() for k in known}
    bare = [i for i, h in enumerate(headers) if str(h).strip() in known_s]
    if not bare or [str(h).strip().casefold() for h in headers[:2]] != \
            ["product", "market"]:
        return None
    covered = set()
    blocks = []
    for k, j in enumerate(bare):
        if j - 2 < 2 or j + 2 >= len(headers):
            return None
        name = headers[j]
        subs = ("pressure", "spend", "bare", "Efficiency", "Marginal Efficiency")
        b = Block(name, first_col=j, last_col=j + 2)
        for off, sub in zip(range(-2, 3), subs):
            h = headers[j + off]
            if k == 0 and off < 0:
                continue                       # the shared pair
            b.cols[sub] = h
            if str(h).startswith(name):
                b.labels[sub] = h[len(name):]
            covered.add(j + off)
        if k > 0:
            b.first_col = j - 2
        blocks.append(b)
    covered |= {0, 1, bare[0] - 2, bare[0] - 1}
    rest = {i for i, h in enumerate(headers) if h not in SUMMARY_COLS}
    if rest != covered or len(covered) != 4 + 5 * len(blocks) - 2:
        return None
    return blocks


def parse_blocks(headers: list[str]) -> list[Block]:
    """Group the variable columns into channel blocks, preserving order.

    By label suffix first. If that leaves blocks that are not variables, or
    variables without their five columns, and the variable names are known,
    read the blocks by position instead (see _positional_blocks)."""
    by_label = _parse_blocks_by_label(headers)
    if not KNOWN_VARIABLES:
        return by_label
    known_s = {k.strip() for k in KNOWN_VARIABLES}
    clean = all(b.name.strip() in known_s and
                {"bare", "Efficiency", "Marginal Efficiency"} <= set(b.cols)
                for b in by_label)
    if clean:
        return by_label
    by_position = _positional_blocks(headers, KNOWN_VARIABLES)
    return by_position if by_position is not None else by_label


def _parse_blocks_by_label(headers: list[str]) -> list[Block]:
    """Group the variable columns into channel blocks by label suffix."""
    blocks: dict[str, Block] = {}
    order: list[str] = []
    for i, h in enumerate(headers):
        if _is_fixed(h) or h in SUMMARY_COLS:
            continue
        sub, name = "bare", h
        for suf in SUFFIXES:
            # case-insensitive: one app version writes ' spend', another ' Spend'
            if h.casefold().endswith(suf.casefold()):
                sub, name = suf.strip().casefold(), h[: -len(suf)]
                sub = {"efficiency": "Efficiency",
                       "marginal efficiency": "Marginal Efficiency"}.get(sub, sub)
                break
        if name not in blocks:
            blocks[name] = Block(name, first_col=i)
            order.append(name)
        blocks[name].cols[sub] = h
        blocks[name].labels[sub] = h[len(name):]
        blocks[name].last_col = i
    return [blocks[n] for n in order]


def summary_column_start(headers: list[str]) -> int:
    """Position of the first summary column, or len(headers) if absent."""
    hits = [i for i, h in enumerate(headers) if h in SUMMARY_COLS]
    return min(hits) if hits else len(headers)


def partition_blocks(blocks: list[Block], summary_start: int
                     ) -> tuple[list[Block], list[Block]]:
    """
    Split channels by whether they sit before or after the summary columns.

    A channel block appended AFTER Average / Maximum / Diminishing Point has no
    row in the summary section -- that section was written when the sheet had
    fewer channels and was never extended. So the summary block is sized by the
    channels that precede it, not by the total.
    """
    leading = [b for b in blocks if b.last_col < summary_start]
    trailing = [b for b in blocks if b.last_col > summary_start]
    return leading, trailing


def split_sections(sheet: pd.DataFrame, n_channels: int) -> tuple[int, int]:
    """
    Returns (summary_rows, curve_rows). n_channels is the number of channels
    that actually have summary rows -- see partition_blocks.
    """
    n_sum = 3 * n_channels
    if n_sum >= len(sheet):
        raise ValueError(
            f"'{CURVES_SHEET}': {n_channels} channels implies a {n_sum}-row "
            f"summary block, but the sheet has only {len(sheet)} rows."
        )
    # The summary block must be exactly the rows where the summary columns live.
    filled = sheet[SUMMARY_COLS].notna().any(axis=1)
    last_summary = int(np.max(np.where(filled)[0])) if filled.any() else -1
    if last_summary != n_sum - 1:
        raise ValueError(
            f"'{CURVES_SHEET}': summary values end at row {last_summary + 2} but "
            f"{n_channels} channels with summary rows imply row {n_sum + 1}.\n"
            f"Implied channel count: {(last_summary + 1) / 3:.2f} "
            f"(should be a whole number).\n"
            f"The summary section does not divide evenly into three groups."
        )
    return n_sum, len(sheet) - n_sum


def campaign_weights(allocated: pd.DataFrame, names: list[str]) -> pd.Series:
    """Share of allocated contribution, which is what makes the curves sum."""
    totals = allocated[names].sum()
    if totals.sum() <= 0:
        raise ValueError("Allocated contribution totals to zero.")
    return totals / totals.sum()


def last_12m_stats(
    data: pd.DataFrame,
    campaigns: dict[str, dict[str, str]],
    period_end: pd.Timestamp,
    date_col: str = "Date",
) -> pd.DataFrame:
    """Mean daily volume and cost per unit over the final 12 months."""
    d = data.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    start = window_start(d[date_col], period_end)
    window = d[(d[date_col] >= start) & (d[date_col] <= period_end)]
    if window.empty:
        raise ValueError(
            f"No DATA rows between {start:%Y-%m-%d} and {period_end:%Y-%m-%d}."
        )
    rows = {}
    for name, c in campaigns.items():
        vol, spend = window[c["raw"]].sum(), window[c["spend"]].sum()
        fallback = False
        if vol > 0 and spend > 0:
            cpu = float(spend / vol)
        else:
            # A campaign that ran earlier but not in the last 12 months has no
            # cost in this window. Falling back to its whole-period cost is
            # better than a zero cost, which would make its spend axis all
            # zeros and its efficiency curve meaningless.
            all_vol, all_spend = d[c["raw"]].sum(), d[c["spend"]].sum()
            cpu = float(all_spend / all_vol) if all_vol > 0 else 0.0
            fallback = True
        rows[name] = {
            "active_in_window": bool(vol > 0),
            "volume_in_window": float(vol),
            # Mean over days the campaign ran -- the platform's own x-bar
            # and what its Average marker reports. A calendar-day mean puts a
            # flighted campaign's marker far down its curve.
            "avg_volume": float(window.loc[window[c["raw"]] > 0, c["raw"]].mean())
                          if (window[c["raw"]] > 0).any() else 0.0,
            "cpu": cpu,
            "cpu_from_full_period": fallback,
            "days": len(window),
        }
    return pd.DataFrame(rows).T


# ==========================================================================
#  BUILDING THE CURVE SHEETS WHEN THE EXPORT HAS NONE
# ==========================================================================

"""
Some exports carry no 'Response Curves' / 'T ROI curves'. This builds both from
the model itself, with the platform's own formula (ported from
mmm_response_curves.py, which reproduces the vendor's sheets to 1e-15):

    KPI_average(x) = coefficient / n_active
                     * SUM over the window of cost_t * f(z scaled to x)_t

    z         the variable's Raw series ('T datasheet variables')
    window    the trailing 365 days of the model
    scaled    only the window is multiplied by x / x_bar; earlier days keep
              their real values, so KPI(0) carries the carryover into it
    x_bar     mean of the non-zero window values; n_active = how many
    cost_t    '(COSTS DEF)', the KPI's unit value, PER PERIOD
    f         lag -> adstock (recency kernels recovered exactly) -> curve

Grid: CURVE_STEPS + 1 points (config; 351 by default, as the platform) from 0
to twice the window maximum ('100%'). Unit cost: mean
non-zero spend over mean non-zero volume. Markers: Average at x_bar, Maximum
at the window maximum, Diminishing Point at the grid point of highest ROI (0
when that is the first defined point). Variables: paid media ('M-'), in
'T structure' order, with spend in the window. Each variable's transform is
first checked against its own contribution; one that does not rebuild exactly
gets no curve, rather than a curve from parameters that are not the model's.
"""

# CURVE_STEPS comes from config.py (default 350, the platform's default): the
# number of steps in the curve grid of a built sheet.
CURVE_LENGTH = 2.0                 # '100%': the grid ends at 2 x window max
MEDIA_PREFIX = "M-"


def _marginal(kpi: np.ndarray, spend: np.ndarray) -> np.ndarray:
    """Central differences, one-sided at the ends -- the vendor's own."""
    out = np.full_like(kpi, np.nan)
    if len(kpi) < 2:
        return out
    with np.errstate(divide="ignore", invalid="ignore"):
        out[0] = (kpi[1] - kpi[0]) / (spend[1] - spend[0])
        out[-1] = (kpi[-1] - kpi[-2]) / (spend[-1] - spend[-2])
        if len(kpi) > 2:
            out[1:-1] = (kpi[2:] - kpi[:-2]) / (spend[2:] - spend[:-2])
    return out


def _variable_transform(row: pd.Series, block: pd.DataFrame):
    """(retention, lag, kernel, curve, alpha, coefficient, message) for one
    variable, with a recency kernel recovered where the decay is one."""
    stated, bracketed, _ = parse_decay(row.get("Decay"))
    if stated is None:
        stated = 1.0 if not DECAY_IS_RETENTION else 0.0     # no carryover
    retention = stated if DECAY_IS_RETENTION else 1.0 - stated
    lag = 0 if pd.isna(row.get("Lag")) else int(row.get("Lag"))
    coef = _num(row.get("Coefficients actual"))
    curve = parse_curve(row.get("Response Curve"))
    alpha = _num(row.get("Alpha"))
    kernel = bracketed
    if coef is None:
        return None, "no 'Coefficients actual'"
    if bracketed is not None:
        rk, msg = recover_recency_kernel(block["Raw"], block["Contribution"],
                                         coef, curve, alpha, lag,
                                         int(bracketed))
        if rk is None:
            return None, f"recency kernel not recovered ({msg})"
        kernel = rk
    err = _exact_error(block["Raw"].to_numpy(dtype=float),
                       block["Contribution"].to_numpy(dtype=float),
                       retention, lag, kernel, coef, curve, alpha)
    if not err <= EXACT_TOLERANCE:
        return None, (f"its decay/lag/curve do not rebuild its contribution "
                      f"(relative error {err:.1e})")
    return (retention, lag, kernel, curve, alpha, coef), None


def build_curve_sheets(structure: pd.DataFrame, datasheet: pd.DataFrame,
                       spends: pd.DataFrame | None, cost: pd.Series | None
                       ) -> tuple[pd.DataFrame | None, pd.DataFrame | None, dict]:
    """Both curve sheets for every paid-media variable, or (None, None, report)
    with the reason when they cannot be built."""
    report = {"built": [], "skipped": {}, "reason": None, "unit_cost": False}
    if cost is None:
        # No '(COSTS DEF)': the KPI is not converted to value, and the
        # platform draws its curves with a cost of 1 per period -- confirmed
        # on a weekly export, every value to 2e-12.
        d0 = pd.to_datetime(datasheet["Period name"]).drop_duplicates()
        cost = pd.Series(1.0, index=pd.DatetimeIndex(d0))
        report["unit_cost"] = True

    ds = datasheet.copy()
    ds["Period name"] = pd.to_datetime(ds["Period name"])
    raw = ds.pivot_table(index="Period name", columns="Variable", values="Raw",
                         aggfunc="first").sort_index()
    end = raw.index.max()
    start = window_start(raw.index, end)
    win = np.asarray((raw.index >= start) & (raw.index <= end))
    cost_w = pd.to_numeric(cost, errors="coerce").reindex(raw.index[win])
    if cost_w.isna().any():
        report["reason"] = (f"'(COSTS DEF)' misses {int(cost_w.isna().sum())} "
                            f"day(s) of the curve window.")
        return None, None, report
    cost_w = cost_w.to_numpy(dtype=float)

    if spends is not None and "Period name" in spends.columns:
        sp = spends.copy()
        sp.index = pd.to_datetime(sp["Period name"], errors="coerce")
        sp = sp[sp.index.notna()]
        sp = sp[~sp.index.duplicated()]
    else:
        sp = ds.pivot_table(index="Period name", columns="Variable",
                            values="Spend", aggfunc="first")
    sp.columns = [str(c).strip() for c in sp.columns]

    curves = []
    for _, row in structure.iterrows():
        name = str(row["Variable"])
        key = name.strip()
        if not key.startswith(MEDIA_PREFIX) or name not in raw.columns:
            continue
        if parse_curve(row.get("Response Curve")) is None:
            continue                  # linear: the platform draws no curve
        if key not in sp.columns:
            continue
        spend = pd.to_numeric(sp[key], errors="coerce").reindex(raw.index[win]) \
            .fillna(0.0).to_numpy()
        if not np.abs(spend).sum() > 0:
            continue
        z = raw[name].fillna(0.0).to_numpy(dtype=float)
        inside = z[win]
        nz = inside[inside != 0]
        if not len(nz):
            report["skipped"][name] = "no activity in the curve window"
            continue
        block = ds[ds["Variable"] == name].sort_values("Period name")
        params, why = _variable_transform(row, block)
        if params is None:
            report["skipped"][name] = why
            continue
        ret, lag, kern, curve, alpha, coef = params
        x_bar, x_max, n_act = float(nz.mean()), float(inside.max()), len(nz)
        sp_nz = spend[spend != 0]
        unit_cost = float(sp_nz.mean()) / x_bar

        def at(level):
            scaled = z.copy()
            scaled[win] *= level / x_bar
            f = model_transform(scaled, ret, lag, kern, curve, alpha)
            return coef * float(np.sum(f[win] * cost_w)) / n_act

        grid = np.linspace(0.0, CURVE_LENGTH * x_max, CURVE_STEPS + 1)
        kpi = np.array([at(g) for g in grid])
        spend_axis = grid * unit_cost
        with np.errstate(divide="ignore", invalid="ignore"):
            roi = np.where(spend_axis != 0, kpi / spend_axis, np.nan)
        usable = np.where(np.isfinite(roi), roi, -np.inf)
        di = int(np.argmax(usable))
        di = 0 if di <= 1 else di
        curves.append({
            "name": name, "pressure": grid, "spend": spend_axis, "bare": kpi,
            "Efficiency": roi, "Marginal Efficiency": _marginal(kpi, spend_axis),
            "Average": (x_bar, x_bar * unit_cost, at(x_bar)),
            "Maximum": (x_max, x_max * unit_cost, at(x_max)),
            "Diminishing Point": (float(grid[di]), float(spend_axis[di]),
                                  float(kpi[di])),
        })
        report["built"].append(name)

    if not curves:
        report["reason"] = "no paid-media variable could be given a curve."
        return None, None, report

    product = ds["Product"].iloc[0] if "Product" in ds.columns else None
    market = ds["Market"].iloc[0] if "Market" in ds.columns else None

    # wide: 'Response Curves'
    cols = ["Product", "Market"]
    for i, c in enumerate(curves):
        n = c["name"]
        # Exactly as the platform spells them: the shared pair is ' pressure'
        # and ' spend' -- leading space, lower case, the label with an empty
        # name in front -- and every other channel '<name> pressure' /
        # '<name> spend'. (Anonymised copies show 'pressure' / 'Spend'; the
        # real export, and the budget optimiser reading it, use these.)
        cols += ([GLOBAL_PRESSURE, GLOBAL_SPEND] if i == 0 else
                 [f"{n} pressure", f"{n} spend"])
        cols += [n, f"{n} Efficiency", f"{n} Marginal Efficiency"]
    cols += SUMMARY_COLS
    rows = []
    for label in SUMMARY_COLS:
        for c in curves:
            p, s_, k = c[label]
            r = dict.fromkeys(cols)
            r.update({"Product": product, "Market": market, cols[2]: p,
                      cols[3]: s_, label: k})
            rows.append(r)
    for j in range(CURVE_STEPS + 1):
        r = dict.fromkeys(cols)
        r.update({"Product": product, "Market": market})
        for i, c in enumerate(curves):
            b = 2 + 5 * i
            for off, sub in enumerate(("pressure", "spend", "bare",
                                       "Efficiency", "Marginal Efficiency")):
                r[cols[b + off]] = c[sub][j]
        rows.append(r)
    wide = pd.DataFrame(rows, columns=cols)

    # tall: 'T ROI curves'
    blocks = []
    for c in curves:
        mk = pd.DataFrame({
            "x-axis": [c[l][1] for l in ROI_MARKERS],
            "KPI": [c[l][2] for l in ROI_MARKERS],
            "Reference": ROI_MARKERS, "Dummy Size": 3})
        body = pd.DataFrame({"x-axis": c["spend"], "KPI": c["bare"],
                             "Reference": np.nan, "Dummy Size": 1})
        blk = pd.concat([mk, body], ignore_index=True)
        blk.insert(0, "Variable", c["name"])
        blocks.append(blk)
    tall = pd.concat(blocks, ignore_index=True)
    tall.insert(0, "Market", market)
    tall.insert(0, "Product", product)
    tall = tall[["Product", "Market", "x-axis", "KPI", "Variable",
                 "Reference", "Dummy Size"]]
    return wide, tall, report


class OwnCurveGateError(ValueError):
    """own_curve could not reproduce the pooled curve, so it will not build
    campaign curves from the same engine."""


CURVE_WINDOW_DAYS = 365      # the platform's default response-curve window


def _native_shape(v, curve):
    v = np.asarray(v, dtype=float)
    if curve == "S Curve":
        return (v / (1.0 + v)) * (1.0 - np.exp(-v))
    if curve == "Diminishing Returns":
        return 1.0 - np.exp(-v)
    return v


def _native_slope(x, b, a, curve):
    v = a * np.asarray(x, dtype=float)
    if curve == "S Curve":
        dg = (1.0 - np.exp(-v)) / (1.0 + v) ** 2 + (v / (1.0 + v)) * np.exp(-v)
    elif curve == "Diminishing Returns":
        dg = np.exp(-v)
    else:
        dg = np.ones_like(v)
    return b * a * dg


def fit_native(x, y, curve):
    """
    b * shape(a * x), the platform's own curve form with its own height (b)
    and one curve parameter (a), least squares to (x, y). For each a, the best
    b is closed-form; a is searched on a log grid then refined. A linear
    variable gets a straight line (a = 1, shape = identity).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if curve is None:
        b = float(x @ y / (x @ x)) if x @ x > 0 else 0.0
        return b, 1.0
    xm = float(x.max()) or 1.0

    def best_b(a):
        phi = _native_shape(a * x, curve)
        d = float(phi @ phi)
        if d <= 0:
            return 0.0, np.inf
        b = float(phi @ y) / d
        return b, float(np.sum((b * phi - y) ** 2))

    grid = np.geomspace(1e-3, 60.0, 400) / xm
    errs = [best_b(a)[1] for a in grid]
    j = int(np.argmin(errs))
    lo, hi = np.log(grid[max(j - 1, 0)]), np.log(grid[min(j + 1, len(grid) - 1)])
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda la: best_b(np.exp(la))[1], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-10})
    a = float(np.exp(r.x)) if r.success else float(grid[j])
    return best_b(a)[0], a


def fit_platform_native(A, cost_w, y, curve):
    """
    (coefficient, alpha) so that the platform's own formula on the campaign's
    own series -- coefficient x SUM over the window of cost_t x f(alpha x
    adstocked raw_t) -- reproduces y at every scale in A (rows: scales,
    columns: window days). For each alpha the best coefficient is closed
    form; alpha is searched on a log grid and refined.
    """
    y = np.asarray(y, dtype=float)
    if curve is None:
        phi = (A * cost_w).sum(axis=1)
        d = float(phi @ phi)
        return (float(phi @ y) / d if d else 0.0), 1.0
    ref = float(np.mean(A[A > 0])) if (A > 0).any() else 1.0

    def best(alpha):
        phi = (apply_curve(A, curve, alpha) * cost_w).sum(axis=1)
        d = float(phi @ phi)
        if d <= 0:
            return 0.0, np.inf
        c = float(phi @ y) / d
        return c, float(np.sum((c * phi - y) ** 2))

    grid = np.geomspace(1e-4, 50.0, 300) / ref
    errs = [best(a)[1] for a in grid]
    j = int(np.argmin(errs))
    lo = np.log(grid[max(j - 1, 0)])
    hi = np.log(grid[min(j + 1, len(grid) - 1)])
    from scipy.optimize import minimize_scalar
    r = minimize_scalar(lambda la: best(np.exp(la))[1], bounds=(lo, hi),
                        method="bounded", options={"xatol": 1e-10})
    alpha = float(np.exp(r.x)) if r.success else float(grid[j])
    return best(alpha)[0], alpha


def own_curves(pooled_var, active, model, p_pool, r_pool, valid, data,
               campaigns, period_end, date_col, kind: str = "own") -> dict:
    """
    Each campaign's own response curve: scale THAT campaign inside the curve
    window, hold every other campaign at what it actually ran, and read the
    KPI the model then predicts -- with the platform's own curve formula
    (coefficient x per-period cost x transformed series, averaged over active
    periods). The curve is the increment over that campaign at zero, so it
    starts at 0.

    Why: the model has one saturation curve over the campaigns' combined
    carryover, so a campaign's next unit lands where the whole variable
    already operates. This is the only construction whose marginal ROI at
    current spend is the model's own answer. The curves do not sum to the
    pooled curve; the difference is saturation the campaigns share, and it is
    reported rather than hidden.

    Gate first: the same engine must rebuild the pooled curve on this sheet
    to 1e-6. If it cannot, nothing built from it is trustworthy.
    """
    if not model:
        raise OwnCurveGateError(
            f"own_curve for '{pooled_var}' needs the model's coefficient, "
            f"curve and cost series, which were not supplied.")
    panel = model["panel"].sort_index()
    cost = model["cost"]
    if cost is None:
        # No '(COSTS DEF)': the platform's curves use a cost of 1 per period.
        cost = pd.Series(1.0, index=pd.DatetimeIndex(panel.index))
    names = list(campaigns)
    X = panel[names].to_numpy(dtype=float)
    dates = panel.index
    start = window_start(dates, period_end)
    win = np.asarray((dates >= start) & (dates <= period_end))
    cost_w = pd.to_numeric(cost, errors="coerce").reindex(dates[win])
    if cost_w.isna().any():
        raise OwnCurveGateError(
            f"own_curve for '{pooled_var}': the cost series misses "
            f"{int(cost_w.isna().sum())} day(s) of the curve window.")
    cost_w = cost_w.to_numpy(dtype=float)
    coef, curve, alpha = model["coef"], model["curve"], model["alpha"]
    ret, lag, kern = model["retention"], model["lag"], model["kernel"]

    def response(scales) -> float:
        """Summed coef * cost * f over the window with campaign k's window
        values multiplied by scales[k]; history before the window as is."""
        m = np.where(win[:, None], np.asarray(scales)[None, :], 1.0)
        total = (X * m).sum(axis=1)
        f = model_transform(total, ret, lag, kern, curve, alpha)
        return coef * float(np.sum(f[win] * cost_w))

    # ---- gate: rebuild the pooled curve -----------------------------------
    pooled_w = X[win].sum(axis=1)
    nz = pooled_w[pooled_w != 0]
    xbar, n_act = float(nz.mean()), len(nz)
    levels = p_pool[valid]
    rebuilt = np.array([response(np.full(len(names), lv / xbar)) / n_act
                        for lv in levels])
    ref = r_pool[valid]
    gate = float(np.max(np.abs(rebuilt - ref)) /
                 max(float(np.max(np.abs(ref))), 1e-12))
    if gate > 1e-6:
        raise OwnCurveGateError(
            f"own_curve for '{pooled_var}': rebuilding the pooled curve on "
            f"'{CURVES_SHEET}' from the model is out by {gate:.1e} (relative; "
            f"limit 1e-6).\n  The curve window, cost series or transform used "
            f"here differ from the platform's,\n  so campaign curves built the "
            f"same way would not be trustworthy. Use shared_shape or\n  "
            f"additive for this variable, and send this message on.")
    ratio = float(levels.max()) / float(pooled_w.max())   # '100%' -> 2.0

    # ---- one curve per campaign -------------------------------------------
    d = data.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    d = d.set_index(date_col)
    built, summary, increments = {}, {}, {}
    n_pts = len(p_pool)

    native = {}
    if kind == "native":
        # Joint-move curves: the whole group scaled by s inside the window,
        # each campaign's share of the response it gets day by day (its share
        # of the adstocked pressure -- the same rule that splits the
        # contribution). Adstock and lag are linear, so each campaign's
        # carry-over is its history part H plus s times its window part W.
        H = np.column_stack([geometric_adstock(np.where(win, 0.0, X[:, k]),
                                               ret, peak_lag=lag, max_lag=kern)
                             for k in range(len(names))])
        Wn = np.column_stack([geometric_adstock(np.where(win, X[:, k], 0.0),
                                                ret, peak_lag=lag, max_lag=kern)
                              for k in range(len(names))])

        def joint(sc):
            Ai = H + sc * Wn
            A = Ai.sum(axis=1)
            f = apply_curve(A, curve, alpha)
            with np.errstate(divide="ignore", invalid="ignore"):
                share = np.where(A[:, None] > 0, Ai / A[:, None], 0.0)
            return coef * (f[win, None] * cost_w[:, None] * share[win]).sum(axis=0)

        model_days = model.get("model_dates")
        for name in active:
            k = names.index(name)
            xi = X[win, k]
            nzi = xi[xi != 0]
            if not len(nzi):
                continue
            xb, ni, mx = float(nzi.mean()), len(nzi), float(xi.max())
            ss = np.linspace(0.0, mx / xb, 61)
            y = np.array([joint(sc)[k] for sc in ss])           # window totals
            # The campaign's own adstocked series at each scale, window days
            Aw = np.array([(H[:, k] + sc * Wn[:, k])[win] for sc in ss])
            coef_i, alpha_i = fit_platform_native(Aw, cost_w, y, curve)
            fit = coef_i * (apply_curve(Aw, curve, alpha_i) * cost_w).sum(axis=1)
            # The platform states the curve parameter as the share of the
            # maximum reached at the variable's mean non-zero raw value over
            # the model period ('S Curve (40,0%)').
            raw_all = X[:, k] if model_days is None else \
                X[np.isin(dates, model_days), k]
            raw_nz = raw_all[raw_all != 0]
            pct = float(_native_shape(alpha_i * raw_nz.mean(), curve)) \
                if (curve and len(raw_nz)) else None
            native[name] = {
                "coefficient": coef_i, "alpha": alpha_i, "type": curve or "linear",
                "percent": pct, "H": H[:, k], "W": Wn[:, k],
                "fit_error": float(np.max(np.abs(fit - y)) /
                                   max(float(np.max(np.abs(y))), 1e-12))}

        def native_window_total(name, sc):
            nv = native[name]
            A = (nv["H"] + sc * nv["W"])[win]
            return nv["coefficient"] * float(np.sum(
                apply_curve(A, curve, nv["alpha"]) * cost_w))

    for name in active:
        k = names.index(name)
        xi = X[win, k]
        nzi = xi[xi != 0]
        if not len(nzi):
            raise OwnCurveGateError(
                f"own_curve: '{name}' has no activity in the {CURVE_WINDOW_DAYS}"
                f"-day curve window, so it has no current level to scale.")
        xbar_i, n_i, max_i = float(nzi.mean()), len(nzi), float(xi.max())
        sp = pd.to_numeric(d[campaigns[name]["spend"]], errors="coerce") \
            .reindex(dates[win]).fillna(0.0).to_numpy()
        sp_nz = sp[sp != 0]
        # the platform's unit cost: mean non-zero spend over mean non-zero volume
        cpu = float(sp_nz.mean()) / xbar_i if len(sp_nz) else 0.0
        base = np.ones(len(names))

        def at(level, k=k, xbar_i=xbar_i, n_i=n_i):
            sc = base.copy()
            sc[k] = level / xbar_i
            return response(sc) / n_i

        zero = at(0.0)
        grid = np.linspace(0.0, ratio * max_i, int(valid.sum()))
        if kind == "native":
            kpi = np.array([native_window_total(name, g / xbar_i)
                            for g in grid]) / n_i
        else:
            kpi = np.array([at(g) for g in grid]) - zero
        spend = grid * cpu
        with np.errstate(divide="ignore", invalid="ignore"):
            roi = np.where(spend > 0, kpi / spend, np.nan)
        marg = np.gradient(kpi, spend) if len(kpi) > 1 and cpu > 0 \
            else np.full_like(kpi, np.nan)
        full = lambda v: np.concatenate([v, np.full(n_pts - len(v), np.nan)])
        built[name] = {"pressure": full(grid), "spend": full(spend),
                       "bare": full(kpi), "Efficiency": full(roi),
                       "Marginal Efficiency": full(marg)}
        usable = np.where(np.isfinite(roi), roi, -np.inf)
        di = int(np.argmax(usable))
        di = 0 if di <= 1 else di
        if kind == "native":
            nat = native[name]
            avg_k = native_window_total(name, 1.0) / n_i
            max_k = native_window_total(name, max_i / xbar_i) / n_i
            # slope against the own curve within +/-20% of current pressure
            errs = []
            for f in (0.8, 0.9, 1.0, 1.1, 1.2):
                x0, h = f * xbar_i, 0.01 * xbar_i
                own_m = (at(x0 + h) - at(x0 - h)) / (2 * h)
                nat_m = (native_window_total(name, (x0 + h) / xbar_i) -
                         native_window_total(name, (x0 - h) / xbar_i)) / (2 * h * n_i)
                if own_m:
                    errs.append((f, nat_m / own_m - 1.0))
            nat["slope_vs_own"] = errs
            nat["n_active"] = n_i
            nat["xbar"] = xbar_i
        else:
            avg_k, max_k = at(xbar_i) - zero, at(max_i) - zero
        summary[name] = {
            "Average": (xbar_i, xbar_i * cpu, avg_k),
            "Maximum": (max_i, max_i * cpu, max_k),
            "Diminishing Point": (float(grid[di]), float(spend[di]),
                                  float(kpi[di])),
        }
        increments[name] = avg_k * n_i          # window total at current spend

    pooled_incr = response(np.ones(len(names))) - response(np.zeros(len(names)))
    share = sum(increments.values()) / pooled_incr if pooled_incr else float("nan")
    out = {"built": built, "summary": summary, "gate_error": gate,
           "window": (pd.Timestamp(start), pd.Timestamp(period_end)),
           "increments_share_of_pooled": share, "kind": kind}
    if kind == "native":
        # How well the native curves add up to the group when all campaigns
        # scale together, over the range the group actually ran at.
        s_max = float(pooled_w.max()) / xbar
        gaps = []
        for sc in np.linspace(0.0, s_max, 41):
            group = response(np.full(len(names), sc))
            parts = sum(native_window_total(n, sc) for n in active)
            gaps.append((sc, parts - group))
        scale_g = max(abs(response(np.ones(len(names)))), 1e-12)
        for nv in native.values():
            nv.pop("H", None)
            nv.pop("W", None)
        out["native"] = native
        out["additivity_gap"] = max(abs(g) for _, g in gaps) / scale_g
        out["additivity_at_current"] = next(
            (g for sc, g in gaps if abs(sc - 1.0) < 1e-9), None)
        if out["additivity_at_current"] is None:
            group1 = pooled_incr
            out["additivity_at_current"] = 0.0
        out["additivity_at_current"] /= scale_g
    return out


def split_curves(
    sheet: pd.DataFrame,
    allocated: pd.DataFrame,
    data: pd.DataFrame,
    pooled_var: str,
    campaigns: dict[str, dict[str, str]],
    period_end: pd.Timestamp,
    date_col: str = "Date",
    model: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    headers = [str(c) for c in sheet.columns]
    gp, gs = global_columns(headers)
    blocks = parse_blocks(headers)
    sum_start = summary_column_start(headers)
    leading, trailing = partition_blocks(blocks, sum_start)
    names = [b.name for b in blocks]
    lead_names = [b.name for b in leading]
    if pooled_var not in names:
        raise ValueError(
            f"'{pooled_var}' not a channel in '{CURVES_SHEET}'. "
            f"Channels: {names[:8]}..."
        )
    idx = names.index(pooled_var)
    pooled = blocks[idx]
    # The first channel has no pressure/spend columns of its own: it uses the
    # sheet's shared pair (also the x-axis the summary rows fill). When it is
    # the one split, the first campaign inherits that pair and every other
    # campaign gets a full five-column block.
    first_borrows = pooled.borrows_global

    if pooled_var not in lead_names:
        raise ValueError(
            f"'{pooled_var}' sits after the summary columns in "
            f"'{CURVES_SHEET}' and so has no summary rows. Splitting a "
            f"trailing channel is not handled."
        )
    n_sum, n_curve = split_sections(sheet, len(leading))
    curve_rows = slice(n_sum, n_sum + n_curve)

    p_pool = sheet.loc[curve_rows, gp if first_borrows else
                       pooled.cols["pressure"]].astype(float).values
    r_pool = sheet.loc[curve_rows, pooled.cols["bare"]].astype(float).values
    valid = ~np.isnan(p_pool)
    n_pts = int(valid.sum())

    w = campaign_weights(allocated, list(campaigns))
    stats = last_12m_stats(data, campaigns, period_end, date_col=date_col)

    # ---- build each campaign's curve ------------------------------------
    # CURVE_MODE may be a single setting or one per variable.
    mode = (CURVE_MODE.get(pooled_var, CURVE_MODE.get("default", "shared_shape"))
            if isinstance(CURVE_MODE, dict) else CURVE_MODE)

    # If the model was built on spend, cost per unit is 1.00 for everyone and
    # there is nothing left to tell the campaigns apart. shared_shape then
    # gives five identical curves that also fail to add up -- the worst of
    # both. additive at least distinguishes them by size and reconciles.
    cpus = [float(stats.loc[n, "cpu"]) for n in campaigns]
    spend_based = all(abs(c - 1.0) < 1e-6 for c in cpus)
    if spend_based and mode == "shared_shape":
        print(f"    WARNING: cost per unit is 1.00 for every campaign, which "
              f"means this\n      variable was modelled on SPEND. Under "
              f"shared_shape the five curves\n      come out identical and "
              f"overlapping, and do not sum to the pooled\n      curve. "
              + _hint(f"Set CURVE_MODE to \"own_curve\" for '{pooled_var}' -- "
                      f"see config.py.",
                      "Choose 'Own curve per campaign' for this variable."))

    # A campaign that has not run in the last 12 months gets no curve.
    #
    # It still gets its spend and contribution -- what it did is history and
    # belongs in the file. But a response curve is a forward-looking statement
    # about what spending on it would return, and there is nothing recent to
    # base that on. Publishing one invites somebody to plan against a campaign
    # that has been dormant for a year.
    dormant = [n for n in campaigns
               if not bool(stats.loc[n, "active_in_window"])]
    active = [n for n in campaigns if n not in dormant]
    if dormant:
        print(f"    NO CURVE for {', '.join(dormant)} -- no activity in the "
              f"last 12 months.\n      Spend and contribution are still "
              f"written; only the curve is omitted.")
    if not active:
        raise ValueError(
            f"None of the campaigns for '{pooled_var}' ran in the last 12 "
            f"months, so none can be given a curve. The pooled variable itself "
            f"is dormant -- consider leaving it unsplit on this sheet.")

    zero_cost = [n for n in active if float(stats.loc[n, "cpu"]) <= 0]
    if zero_cost:
        raise ValueError(
            f"No cost per unit could be derived for: {', '.join(zero_cost)}.\n"
            f"These campaigns have no spend in the import file at all, so their "
            f"response curves would sit on an all-zero spend axis. Check the "
            + _hint("spend columns named in config.py.",
                    "spend columns chosen for them in step 2.")
        )

    # shared_shape: the pooled curve itself, with only the spend axis differing
    #   by cost per unit. Cheaper media reaches a given response for less
    #   spend, so on a chart of response against spend the cheap campaigns sit
    #   higher -- the ranking a reader expects, matching the efficiency figures.
    #
    # additive: the pooled curve scaled down by share of volume, so the five
    #   sum to the pooled curve. Correct for an unconstrained optimiser, but
    #   every campaign saturates almost at once and then sits at a plateau set
    #   by its volume, which reads as a performance ranking and is not one.
    if mode not in ("shared_shape", "additive", "own_curve", "native_curve"):
        raise ValueError(f"Unknown CURVE_MODE {mode!r} for '{pooled_var}'. "
                         f"Use native_curve, own_curve, shared_shape or additive.")
    own = None
    if mode in ("own_curve", "native_curve"):
        own = own_curves(pooled_var, active, model, p_pool, r_pool, valid,
                         data, campaigns, period_end, date_col,
                         kind="native" if mode == "native_curve" else "own")

    built = {}
    for name in (active if own is None else []):
        wi, cpu = float(w[name]), float(stats.loc[name, "cpu"])
        scale = 1.0 if mode == "shared_shape" else wi
        p = p_pool * scale
        r = r_pool * scale
        s = p * cpu
        with np.errstate(divide="ignore", invalid="ignore"):
            eff = np.where(s > 0, r / s, np.nan)
        marg = np.full_like(r, np.nan)
        if n_pts > 1:
            marg[valid] = np.gradient(r[valid], s[valid])
        built[name] = {"pressure": p, "spend": s, "bare": r,
                       "Efficiency": eff, "Marginal Efficiency": marg}

    # ---- summary rows ----------------------------------------------------
    n_ch = len(leading)
    lead_idx = lead_names.index(pooled_var)
    summary_new: dict[str, dict[str, tuple]] = {n: {} for n in active}
    if own is not None:
        built = own["built"]
        summary_new = own["summary"]
    for name in (active if own is None else []):
        wi, cpu = float(w[name]), float(stats.loc[name, "cpu"])
        c = built[name]
        pv, rv = c["pressure"][valid], c["bare"][valid]

        # Average: this campaign's own last-12m volume, read off its own curve.
        avg_p = float(stats.loc[name, "avg_volume"])
        summary_new[name]["Average"] = (
            avg_p, avg_p * cpu, float(np.interp(avg_p, pv, rv)))

        # Maximum and Diminishing Point are points ON the curve, so they
        # follow whichever curve was built: unchanged under shared_shape,
        # scaled by w_i under additive. The spend at those points always uses
        # the campaign's own cost.
        for label in ("Maximum", "Diminishing Point"):
            g = SUMMARY_COLS.index(label)
            row = g * n_ch + lead_idx
            p0 = float(sheet.at[row, gp])
            r0 = float(sheet.at[row, label])
            scale = 1.0 if mode == "shared_shape" else wi
            summary_new[name][label] = (p0 * scale, p0 * scale * cpu, r0 * scale)

    # ---- reassemble ------------------------------------------------------
    # New blocks copy the suffix spelling of an existing block, so the sheet
    # stays in one style whichever app version wrote it.
    styled = next((b for b in blocks if "pressure" in b.labels
                   and "spend" in b.labels), None)
    lab = {"pressure": " pressure", "spend": " spend",
           "Efficiency": " Efficiency",
           "Marginal Efficiency": " Marginal Efficiency"}
    if styled:
        lab.update(styled.labels)
    lab.pop("bare", None)
    # Where each campaign's five series go. The first campaign of a split
    # first channel writes its pressure/spend into the shared pair.
    col_for = {}
    for i, name in enumerate(active):
        col_for[name] = {
            "pressure": gp if (first_borrows and i == 0)
            else f"{name}{lab['pressure']}",
            "spend": gs if (first_borrows and i == 0)
            else f"{name}{lab['spend']}",
            "bare": name,
            "Efficiency": f"{name}{lab['Efficiency']}",
            "Marginal Efficiency": f"{name}{lab['Marginal Efficiency']}",
        }
    trigger = pooled.cols["bare"] if first_borrows else pooled.cols["pressure"]
    new_headers: list[str] = []
    for h in headers:
        if h in (pooled.cols.get(k) for k in
                 ("pressure", "spend", "bare", "Efficiency", "Marginal Efficiency")):
            if h == trigger:
                for i, name in enumerate(active):
                    subs = (("bare", "Efficiency", "Marginal Efficiency")
                            if first_borrows and i == 0 else
                            ("pressure", "spend", "bare", "Efficiency",
                             "Marginal Efficiency"))
                    new_headers += [col_for[name][k] for k in subs]
            continue
        new_headers.append(h)

    new_n_ch = n_ch - 1 + len(active)
    new_n_sum = 3 * new_n_ch
    out = pd.DataFrame(index=range(new_n_sum + n_curve),
                       columns=new_headers, dtype=object)
    out["Product"] = sheet["Product"].iloc[0]
    out["Market"] = sheet["Market"].iloc[0]

    new_lead = (lead_names[:lead_idx] + list(active)
                + lead_names[lead_idx + 1:])
    new_curve_rows = slice(new_n_sum, new_n_sum + n_curve)

    # Summary block: only the leading channels have rows here. Trailing
    # channels had none before and are left without, unchanged.
    for g, label in enumerate(SUMMARY_COLS):
        for i, ch in enumerate(new_lead):
            dst = g * new_n_ch + i
            if ch in active:
                p, s, r = summary_new[ch][label]
            else:
                src = g * n_ch + lead_names.index(ch)
                p = sheet.at[src, gp]
                s = sheet.at[src, gs]
                r = sheet.at[src, label]
            out.at[dst, gp] = p
            out.at[dst, gs] = s
            out.at[dst, label] = r

    # curves: untouched channels copied, new ones written
    for b in blocks:
        if b.name == pooled_var:
            continue
        for sub, col in b.cols.items():
            out.loc[new_curve_rows, col] = sheet.loc[curve_rows, col].values
    if blocks[0].borrows_global and not first_borrows:
        for col in (gp, gs):
            out.loc[new_curve_rows, col] = sheet.loc[curve_rows, col].values

    for name, c in built.items():
        for sub, col in col_for[name].items():
            out.loc[new_curve_rows, col] = c[sub]

    # ---- checks ----------------------------------------------------------
    if mode in ("own_curve", "native_curve"):
        resid = own["gate_error"]
    elif mode == "shared_shape":
        # By construction each campaign carries the pooled response, so the
        # check is that the shape was preserved, not that the parts add up.
        resid = float(np.nanmax([np.nanmax(np.abs(built[n]["bare"] - r_pool))
                                 for n in active]))
        if resid > 1e-9:
            raise ValueError(f"Shared-shape curves altered the response: "
                             f"{resid:.3e}")
    else:
        recombined = sum(built[n]["bare"] for n in active)
        resid = float(np.nanmax(np.abs(recombined - r_pool)))
        if resid > 1e-6 * max(1.0, float(np.nanmax(np.abs(r_pool)))):
            raise ValueError(f"Curves do not sum to the pooled curve: {resid:.3e}")

    report = {
        "sheet": CURVES_SHEET,
        "pooled_channel": pooled_var,
        "channel_position": idx + 1,
        "channels_with_summary_rows": f"{n_ch} -> {new_n_ch}",
        "channels_total": f"{len(blocks)} -> {len(blocks) - 1 + len(campaigns)}",
        "trailing_channels_no_summary": [b.name for b in trailing],
        "columns": f"{len(headers)} -> {len(new_headers)}",
        "summary_rows": f"{n_sum} -> {new_n_sum}",
        "curves_start_sheet_row": f"{n_sum + 2} -> {new_n_sum + 2}",
        "total_rows": f"{len(sheet)} -> {len(out)}",
        "curve_points": n_pts,
        "curve_mode": mode,
        "built_curves": built,
        "summary_points": summary_new,
        "campaigns_with_curves": list(active),
        "campaigns_without_curves": list(dormant),
        "modelled_on_spend": spend_based,
        "max_curve_residual": resid,
        "own_curve": ({k: v for k, v in own.items()
                       if k not in ("built", "summary")} if own else None),
        "weights": {k: round(float(v), 4) for k, v in w.items()},
        "cpu_last_12m": {k: round(float(stats.loc[k, "cpu"]), 6) for k in active},
        "cpu_fell_back_to_full_period": [
            k for k in active if bool(stats.loc[k, "cpu_from_full_period"])],
        "avg_volume_last_12m": {
            k: round(float(stats.loc[k, "avg_volume"]), 1) for k in active},
        "window_days": int(stats["days"].iloc[0]),
    }
    return out, report


# ==========================================================================
#  WRITING THE RESULT
# ==========================================================================

"""
Step 4 -- write the result to a new file.

The original is copied first, then only the sheets we actually changed are
rewritten in place. Every other sheet is left exactly as it was -- not read,
not parsed, not round-tripped. As further sheets get added to the pipeline
they join `updates`; nothing else here changes.

Assumes the output workbook contains no formulas.
"""

def _row_styles(ws, row_idx: int, n_cols: int) -> list[dict]:
    """Capture formatting from a template row so new rows match the sheet."""
    out = []
    # openpyxl deprecated the style .copy() methods but they still behave
    # correctly; the warning would otherwise print once per cell.
    warnings.filterwarnings("ignore", category=DeprecationWarning,
                            module="openpyxl")
    for c in range(1, n_cols + 1):
        cell = ws.cell(row=row_idx, column=c)
        out.append({
            "number_format": cell.number_format,
            "font": cell.font.copy(),
            "alignment": cell.alignment.copy(),
            "border": cell.border.copy(),
            "fill": cell.fill.copy(),
        })
    return out


PLATFORM_STRUCTURE_SHEET = "Structure"
ORIGINAL_STRUCTURE_SHEET = "Structure (original)"


def rebuild_platform_structure(wb, rows_by_pooled: dict) -> dict:
    """
    The platform's own 'Structure' sheet with the split campaigns in place of
    their combined variable. The original is kept, renamed
    'Structure (original)', right after it; the new sheet is a cell-for-cell
    copy (formatting, widths, colours) with each combined row replaced by one
    row per campaign, in the same place and the same style.

    Per campaign: normalized and standardized coefficients follow the
    platform's own definitions applied to the campaign's contribution series
    (share of the KPI total; standard deviation relative to the KPI's).
    Coefficients Actual, Response Curve and Alpha are the native
    curve's own parameters (what the platform would need to draw that curve);
    Decay, Lag and Category are the combined variable's; ROI and CPU are the
    campaign's, as on 'T structure'. SE, CI, t-Stat, p-Value and stars are
    left empty: the model never estimated the campaigns, so there is no
    uncertainty to report for them.
    """
    from copy import copy
    if PLATFORM_STRUCTURE_SHEET not in wb.sheetnames or not rows_by_pooled:
        return {"rebuilt": False}
    old = wb[PLATFORM_STRUCTURE_SHEET]
    idx = wb.sheetnames.index(PLATFORM_STRUCTURE_SHEET)
    new = wb.copy_worksheet(old)
    old.title = ORIGINAL_STRUCTURE_SHEET
    new.title = PLATFORM_STRUCTURE_SHEET
    wb._sheets.remove(new)
    wb._sheets.insert(idx, new)
    new.sheet_view.tabSelected = False

    # column positions from the sub-header row (the one holding 'Decay')
    hdr_row, cols = None, {}
    for r in range(1, min(new.max_row, 15) + 1):
        vals = [str(c.value).strip().casefold() if c.value is not None else ""
                for c in new[r]]
        if "decay" in vals and "alpha" in vals:
            hdr_row = r
            cols = {v: i + 1 for i, v in enumerate(vals) if v}
            break
    if hdr_row is None:
        return {"rebuilt": False, "reason": "no header row with Decay/Alpha"}
    c_actual = cols.get("actual")
    c_norm, c_std = cols.get("normalized"), cols.get("standardized")
    c_curve, c_alpha = cols.get("response curve"), cols.get("alpha")
    c_roi, c_cpu = cols.get("roi"), cols.get("cpu")
    keep = {cols.get(k) for k in ("decay", "lag", "category")} - {None}
    n_cols = new.max_column

    done = []
    for pooled, rows in rows_by_pooled.items():
        r0 = next((r for r in range(hdr_row + 1, new.max_row + 1)
                   if str(new.cell(r, 1).value or "").strip() == pooled.strip()),
                  None)
        if r0 is None or not rows:
            continue
        template = [new.cell(r0, c) for c in range(1, n_cols + 1)]

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return None
        pooled_norm = _f(new.cell(r0, c_norm).value) if c_norm else None
        pooled_std = _f(new.cell(r0, c_std).value) if c_std else None
        kept = {c: new.cell(r0, c).value for c in keep}
        styles = [(copy(t.font), copy(t.fill), copy(t.border), copy(t.alignment),
                   t.number_format, copy(t.protection)) for t in template]
        if len(rows) > 1:
            new.insert_rows(r0 + 1, len(rows) - 1)
        for i, row in enumerate(rows):
            r = r0 + i
            for c in range(1, n_cols + 1):
                cell = new.cell(r, c)
                f, fl, b, al, nf, pr = styles[c - 1]
                cell.font, cell.fill, cell.border = f, fl, b
                cell.alignment, cell.number_format, cell.protection = al, nf, pr
                cell.value = kept.get(c)
            new.cell(r, 1).value = row["name"]
            if c_norm and pooled_norm is not None and row.get("sum_share") is not None:
                new.cell(r, c_norm).value = pooled_norm * row["sum_share"]
            if c_std and pooled_std is not None and row.get("sd_share") is not None:
                new.cell(r, c_std).value = pooled_std * row["sd_share"]
            if c_actual:
                new.cell(r, c_actual).value = row.get("coefficient")
            if c_curve:
                new.cell(r, c_curve).value = row.get("curve_text")
            if c_alpha:
                new.cell(r, c_alpha).value = row.get("alpha")
            if c_roi:
                new.cell(r, c_roi).value = row.get("roi")
            if c_cpu:
                new.cell(r, c_cpu).value = row.get("cpu")
        done.append(pooled)
    return {"rebuilt": True, "variables": done}


def _platform_curve_format(ws, sheet_name: str) -> None:
    """
    The curve sheets cell for cell as the platform writes them, so whatever
    reads the platform's file reads ours the same way:

      * every cell of the table exists; an empty one is an empty text cell,
        not a missing cell (the platform writes the whole grid)
      * 'Response Curves': header cells wrap text, column A is 9.14 wide
      * 'T ROI curves': yellow sheet tab, panes frozen at C2
    """
    from openpyxl.styles import Alignment
    n_rows, n_cols = ws.max_row, ws.max_column
    for row in ws.iter_rows(min_row=2, max_row=n_rows, max_col=n_cols):
        for cell in row:
            if cell.value is None:
                cell.value = ""
    if sheet_name == CURVES_SHEET:
        for cell in ws[1]:
            cell.alignment = Alignment(wrap_text=True)
        ws.column_dimensions["A"].width = 9.140625
    else:
        ws.sheet_properties.tabColor = "FFFFFF00"
        ws.freeze_panes = "C2"


def _write_sheet(wb, sheet_name: str, df: pd.DataFrame,
                 style_from_row: int = 2, header_row: int = 1,
                 allow_column_change: bool = False) -> None:
    """Replace a sheet's data rows with df, keeping header and formatting."""
    ws = wb[sheet_name]
    n_cols = len(df.columns)
    old_cols = ws.max_column

    existing = [ws.cell(row=header_row, column=c).value
                for c in range(1, old_cols + 1)]
    existing = [str(h) if h is not None else None for h in existing]
    incoming = [str(c) for c in df.columns]

    # Compare leniently, but write back the EXACT header text. Some sheets have
    # headers with significant leading spaces (' pressure') that the downstream
    # environment matches on -- stripping them silently renames the column.
    def norm(v):
        return v.strip() if isinstance(v, str) else v

    if [norm(h) for h in existing] != [norm(c) for c in incoming]:
        if not allow_column_change:
            raise ValueError(
                f"Header mismatch on '{sheet_name}'.\n"
                f"  file: {existing}\n  data: {incoming}"
            )
        # Column set is meant to change (a pooled column became several).
        # Rewrite the header row and drop any now-surplus columns.
        for c in range(1, max(n_cols, old_cols) + 1):
            ws.cell(row=header_row, column=c,
                    value=incoming[c - 1] if c <= n_cols else None)
        if old_cols > n_cols:
            ws.delete_cols(n_cols + 1, old_cols - n_cols)

    first_data = header_row + 1
    styles = (_row_styles(ws, style_from_row, n_cols)
              if ws.max_row >= style_from_row else None)

    if ws.max_row >= first_data:
        ws.delete_rows(first_data, ws.max_row - first_data + 1)

    for r, (_, row) in enumerate(df.iterrows(), start=first_data):
        for c, col in enumerate(df.columns, start=1):
            v = row[col]
            if pd.isna(v):
                v = None
            elif isinstance(v, pd.Timestamp):
                v = v.to_pydatetime()
            elif hasattr(v, "item"):
                v = v.item()
            cell = ws.cell(row=r, column=c, value=v)
            if styles:
                s = styles[c - 1]
                cell.number_format = s["number_format"]
                cell.font = s["font"]
                cell.alignment = s["alignment"]
                cell.border = s["border"]
                cell.fill = s["fill"]

    # An autofilter left pointing at the old range hides rows or errors on open.
    if ws.auto_filter and ws.auto_filter.ref:
        ws.auto_filter.ref = (
            f"A{header_row}:{get_column_letter(n_cols)}"
            f"{first_data + len(df) - 1}"
        )


def write_output(
    source_path: str | Path,
    updates: dict[str, pd.DataFrame],
    dest_path: str | Path | None = None,
    suffix: str = "_split",
    column_change_sheets: set[str] | None = None,
    header_rows: dict[str, int] | None = None,
    new_sheets: dict[str, str | None] | None = None,
    platform_structure_rows: dict | None = None,
) -> Path:
    """
    source_path          : the untouched original output file
    updates              : {sheet name: replacement DataFrame}
    dest_path            : explicit destination, or None to derive from suffix
    column_change_sheets : sheets whose column set is expected to change
    header_rows          : {sheet name: 1-based header row}, default 1
    """
    column_change_sheets = column_change_sheets or set()
    header_rows = header_rows or {}
    source_path = Path(source_path)
    if dest_path is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M")
        dest_path = source_path.with_name(
            f"{source_path.stem}{suffix}_{stamp}{source_path.suffix}")
    dest_path = Path(dest_path)

    if dest_path.resolve() == source_path.resolve():
        raise ValueError("Destination is the source file. Refusing to overwrite.")

    # Build in a temporary file and only move it into place once every sheet
    # has been written and the result verified. Editing dest_path directly
    # leaves a byte-identical copy of the original behind if anything fails --
    # a file that looks like a successful result but contains no changes.
    tmp_path = dest_path.with_name(f".partial_{dest_path.name}")
    try:
        shutil.copy2(source_path, tmp_path)

        wb = load_workbook(tmp_path)
        # Sheets built from scratch (the curve sheets of an export without
        # them) are created next to where the platform puts them.
        for name, before in (new_sheets or {}).items():
            if name in wb.sheetnames:
                continue
            idx = (wb.sheetnames.index(before) if before in wb.sheetnames
                   else len(wb.sheetnames))
            ws = wb.create_sheet(name, idx)
            for c, h in enumerate(updates[name].columns, start=1):
                ws.cell(row=1, column=c, value=str(h))
            if name == ROI_CURVES_SHEET:
                ws.freeze_panes = "C2"             # as the platform writes it
        missing = [n for n in updates if n not in wb.sheetnames]
        if missing:
            raise ValueError(
                f"Sheet(s) not in {source_path.name}: {', '.join(missing)}.\n"
                f"Sheets present: {', '.join(wb.sheetnames)}"
            )
        for name, df in updates.items():
            _write_sheet(wb, name, df,
                         header_row=header_rows.get(name, 1),
                         allow_column_change=name in column_change_sheets)
            if name in (CURVES_SHEET, ROI_CURVES_SHEET):
                _platform_curve_format(wb[name], name)
        if platform_structure_rows:
            rebuild_platform_structure(wb, platform_structure_rows)
        wb.save(tmp_path)

        # The written file must actually differ from the source. If it does
        # not, the edits silently did nothing and shipping it would be worse
        # than failing.
        if tmp_path.stat().st_size == source_path.stat().st_size:
            check_o = pd.read_excel(source_path, sheet_name=None)
            check_n = pd.read_excel(tmp_path, sheet_name=None)
            if all(check_o[k].equals(check_n.get(k)) for k in check_o):
                raise ValueError(
                    "The written file is identical to the source -- no sheet "
                    "was actually modified. Nothing has been saved."
                )

        try:
            if dest_path.exists():
                dest_path.unlink()
            tmp_path.replace(dest_path)
        except PermissionError:
            raise PermissionError(
                f"Cannot write '{dest_path.name}' -- it is open in Excel (or "
                f"another program has it locked).\n"
                f"Close it and run the command again. Nothing has been changed."
            ) from None
    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return dest_path


def _trim_empty_tail(df: pd.DataFrame) -> pd.DataFrame:
    filled = np.where(df.notna().any(axis=1).to_numpy())[0]
    return df.iloc[: (filled[-1] + 1) if len(filled) else 0]


def verify_written(dest_path: Path, source_path: Path,
                   updates: dict[str, pd.DataFrame]) -> dict:
    """Read the written file back and confirm it matches what we intended."""
    written = pd.read_excel(dest_path, sheet_name=None)
    original = pd.read_excel(source_path, sheet_name=None)

    result = {"file": str(dest_path), "sheets": {}}
    rebuilt = ORIGINAL_STRUCTURE_SHEET in written and \
        ORIGINAL_STRUCTURE_SHEET not in original
    if rebuilt:
        original = dict(original)
        original[ORIGINAL_STRUCTURE_SHEET] = original.get(PLATFORM_STRUCTURE_SHEET)
    for name in written:
        entry = {"rows": len(written[name]), "columns": len(written[name].columns)}
        if rebuilt and name == PLATFORM_STRUCTURE_SHEET:
            entry["status"] = "rebuilt: campaigns in place of combined variables"
            result["sheets"][name] = entry
            continue
        if name in updates:
            exp = updates[name]
            entry["status"] = "modified"
            entry["row_match"] = len(written[name]) == len(exp)
            entry["header_match"] = (
                [str(c) for c in written[name].columns] == [str(c) for c in exp.columns])
        else:
            entry["status"] = "untouched"
            o = original.get(name)
            n = written[name]
            # An empty row at the very end can be counted in one file and
            # not the other (a formatted-but-empty row is not saved back).
            # It holds nothing, so compare without trailing empty rows.
            if o is not None:
                o, n = _trim_empty_tail(o), _trim_empty_tail(n)
            if o is None:
                entry["identical_to_source"] = False
            elif o.equals(n):
                entry["identical_to_source"] = True
            else:
                # Excel stores 15 significant digits; openpyxl writes Python's
                # full float repr, so an untouched value can come back as
                # 1140.079298630137 instead of 1140.0792986301371. Judge the
                # numbers, not their spelling.
                same = (o.shape == n.shape
                        and list(o.columns) == list(n.columns))
                if same:
                    for c in o.columns:
                        a, b = o[c], n[c]
                        if a.equals(b):
                            continue
                        an = pd.to_numeric(a, errors="coerce")
                        bn = pd.to_numeric(b, errors="coerce")
                        if an.notna().any() and np.allclose(
                                an.fillna(0), bn.fillna(0),
                                rtol=1e-12, atol=1e-12):
                            continue
                        same = False
                        break
                entry["identical_to_source"] = (
                    True if same else False)
                if same:
                    entry["note"] = "values match to 1e-12; float repr differs"
                else:
                    # pandas streams a sheet as its XML rows are stored, and
                    # some exports store rows in an order that reads back
                    # differently once the file is re-saved in order. Judge
                    # by the cells themselves before calling it changed.
                    src_name = (PLATFORM_STRUCTURE_SHEET
                                if name == ORIGINAL_STRUCTURE_SHEET else name)
                    if _cells_identical(source_path, dest_path, name, src_name):
                        entry["identical_to_source"] = True
                        entry["note"] = "cells identical; source rows stored out of order"
        result["sheets"][name] = entry
    return result


def _cells_identical(source_path, dest_path, sheet: str,
                     source_sheet: str | None = None) -> bool:
    from openpyxl import load_workbook
    try:
        a = load_workbook(source_path)[source_sheet or sheet]
        b = load_workbook(dest_path)[sheet]
    except Exception:                                 # noqa: BLE001
        return False
    ra = [list(r) for r in a.iter_rows(values_only=True)]
    rb = [list(r) for r in b.iter_rows(values_only=True)]
    if len(ra) != len(rb):
        return False
    for x, y in zip(ra, rb):
        for u, v in zip(x, y):
            if u == v:
                continue
            if isinstance(u, (int, float)) and isinstance(v, (int, float)) \
                    and np.isclose(u, v, rtol=1e-12, atol=1e-12):
                continue
            return False
    return True


# A hand adjustment covers a few days. Beyond these bounds it is something
# else, and rescaling would paper over it.
MAX_RESCALE_DAY_SHARE = 0.02      # 2% of the modelling period
MAX_RESCALE_VOLUME = 0.02         # 2% of total volume

COLUMN_CHANGE = {SPENDS_SHEET, CONTRIB_SHEET, CURVES_SHEET}  # column set changes
MAX_ACCEPTED_SHARE_IMPACT = 0.25   # percentage points, per campaign total
MAX_ACCEPTED_DAY_IMPACT = 5.0      # percent of contribution moving between days


COSTS_SHEET = "(COSTS DEF)"


def _num(v) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else f


def _ref_point(row: pd.Series) -> str | None:
    """The variable's reference point, if the structure sheet has that column."""
    for col in row.index:
        key = str(col).casefold().replace(".", "").replace(" ", "")
        if key in ("refpoint", "referencepoint"):
            v = row[col]
            if v is None or (isinstance(v, float) and np.isnan(v)) \
                    or not str(v).strip() or str(v).strip().casefold() == "none":
                return None
            return str(v).strip()
    return None


def load_cost_series(path) -> pd.Series | None:
    """The KPI's per-period unit value from '(COSTS DEF)', or None."""
    try:
        hdr = find_header_row(path, COSTS_SHEET)
        df = canon_cols(pd.read_excel(path, sheet_name=COSTS_SHEET, header=hdr))[0]
    except ValueError:
        return None
    vals = [c for c in df.columns
            if c not in ("Product", "Market", "Period", "Period name")]
    if not vals or "Period name" not in df.columns:
        return None
    out = pd.to_numeric(df[vals[0]], errors="coerce")
    out.index = pd.to_datetime(df["Period name"])
    return out[~out.index.duplicated()]


def cmd_split(args) -> int:
    """Run the split. Writes nothing unless --apply is given."""

    # A config with several "SPLITS = [...]" blocks silently keeps only the
    # last one: each assignment replaces the previous. Someone editing the file
    # to add a second variable will reasonably expect both to run. Catch it
    # here rather than let the run quietly do less than was asked.
    # This guard is about someone editing config.py by hand. When the split was
    # supplied programmatically -- by the app, say -- config.py's own SPLITS is
    # irrelevant and a stale one there must not block the run.
    try:
        import config as _cfg
        supplied = SPLITS is not getattr(_cfg, "SPLITS", None)
        n_assign = 1 if supplied else len(re.findall(
            r"^SPLITS\s*=", Path(_cfg.__file__).read_text(),
            flags=re.MULTILINE))
    except Exception:
        n_assign = 1               # never let the guard itself break a run
    if n_assign > 1:
        print(f"\nSTOPPED. config.py contains {n_assign} separate "
              f"'SPLITS = [...]' blocks.\n"
              f"Python keeps only the last one, so the other {n_assign - 1} "
              f"would be ignored\nand those variables would NOT be split.\n\n"
              f"Put every variable in ONE list instead:\n\n"
              f"    SPLITS = [\n"
              f"        Split(pooled=\"first variable\",  campaigns=[...]),\n"
              f"        Split(pooled=\"second variable\", campaigns=[...]),\n"
              f"    ]\n")
        return 1

    print(f"{len(SPLITS)} variable(s) to split: "
          f"{', '.join(s.pooled for s in SPLITS)}\n")

    inp = load_inputs(args.output_file, args.import_file)
    global KNOWN_VARIABLES
    KNOWN_VARIABLES = {str(v) for v in inp.model_variables} | \
        {c.name for sp in SPLITS for c in sp.campaigns}
    data = inp.data()
    date_col = inp.date_column()
    # Import files often carry a units row under the header ('Amount',
    # 'Spends'). It has no date, belongs to no day, and would otherwise be
    # reported as unreadable numbers in every column.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dated = pd.to_datetime(data[date_col], errors="coerce").notna()
    if (~dated).any():
        print(f"  {int((~dated).sum())} row(s) on '{DATA_SHEET}' have no date "
              f"(a units row, say) and are ignored")
        data = data[dated].reset_index(drop=True)
    try:
        data, made = add_composite_columns(data, COMPOSITE_COLUMNS)
    except ValueError as e:
        print(f"\nSTOPPED. {e}")
        return 1
    for name, parts in made.items():
        print(f"  '{name}' kept whole: the sum of {len(parts)} DATA column(s)")

    # Excel columns often arrive as text -- blanks typed as "-", numbers with a
    # comma decimal separator, stray notes in a cell. Summing those concatenates
    # strings instead of adding, so coerce and report anything that will not
    # convert rather than failing deep inside the maths.
    for split_cfg in SPLITS:
        for col in split_cfg.data_columns_used:
            if col not in data.columns:
                continue
            before = data[col]
            if pd.api.types.is_numeric_dtype(before):
                continue
            # Strip thousands separators and swap a comma decimal point for a
            # full stop. Done with plain replacements rather than a regex:
            # pandas' Arrow string engine rejects \u escapes, and a regex here
            # buys nothing.
            cleaned = before.astype(str)
            for ch in (" ", "\t", "\u00a0", "\u202f", "\u2009"):
                cleaned = cleaned.str.replace(ch, "", regex=False)
            after = pd.to_numeric(
                cleaned.str.replace(",", ".", regex=False), errors="coerce")
            bad = before[after.isna() & before.notna()]
            data[col] = after.fillna(0.0)
            note = ""
            if len(bad):
                sample = ", ".join(repr(str(x)) for x in bad.unique()[:4])
                note = (f" -- {len(bad)} value(s) could not be read as numbers, "
                        f"treated as 0: {sample}")
            print(f"  '{col}' was stored as text, converted to numbers{note}")
    print(f"Date column on '{inp.import_path.name}': '{date_col}'")

    structure = inp.structure
    # Read under canonical column names ('Period Name' -> 'Period name'); the
    # *_back maps restore each sheet's own header text before writing.
    datasheet = pd.read_excel(args.output_file, sheet_name=DATASHEET)
    datasheet.columns = [str(c).strip() for c in datasheet.columns]
    datasheet, ds_back = canon_cols(datasheet)
    hdr = find_header_row(args.output_file, SPENDS_SHEET)
    spends, sp_back = canon_cols(
        pd.read_excel(args.output_file, sheet_name=SPENDS_SHEET, header=hdr))
    hdr_ic = find_header_row(args.output_file, CONTRIB_SHEET)
    contribs, ic_back = canon_cols(
        pd.read_excel(args.output_file, sheet_name=CONTRIB_SHEET, header=hdr_ic))
    contribs_source = contribs.copy()
    # Newer exports may carry no curve sheets at all. Then there is nothing to
    # split there, which is said once rather than treated as an error.
    curves = (pd.read_excel(args.output_file, sheet_name=CURVES_SHEET)
              if CURVES_SHEET in inp.output_sheets else None)
    built_sheets: dict[str, str | None] = {}
    platform_rows: dict[str, list] = {}
    if curves is None and BUILD_MISSING_CURVES:
        sp_for_curves = canon_cols(pd.read_excel(
            args.output_file, sheet_name=SPENDS_SHEET,
            header=find_header_row(args.output_file, SPENDS_SHEET)))[0]
        ds_for_curves = canon_cols(pd.read_excel(args.output_file,
                                                 sheet_name=DATASHEET))[0]
        wide, tall, rb = build_curve_sheets(
            inp.structure, ds_for_curves, sp_for_curves,
            load_cost_series(args.output_file))
        if wide is None:
            print(f"NOTE: no '{CURVES_SHEET}' sheet in this file, and it could "
                  f"not be built: {rb['reason']}")
        else:
            curves = wide
            built_sheets[CURVES_SHEET] = STRUCTURE_SHEET
            print(f"BUILT '{CURVES_SHEET}' from the model: {len(rb['built'])} "
                  f"paid-media variable(s), the platform's own formula")
            if rb.get("unit_cost"):
                print(f"  NOTE: no '(COSTS DEF)' sheet, so the curves are in KPI "
                      f"units (a cost of 1 per period), as the platform draws "
                      f"them.")
            if ROI_CURVES_SHEET not in inp.output_sheets:
                built_sheets[ROI_CURVES_SHEET] = "Comments"   # platform order
                print(f"BUILT '{ROI_CURVES_SHEET}' from the same curves")
            for v, why in rb["skipped"].items():
                print(f"  WARNING: no curve for '{v}': {why}")
    elif curves is None:
        print(f"NOTE: no '{CURVES_SHEET}' sheet in this file, so no response "
              f"curves are split.")
    weekly, wk_back, hdr_wk = None, {}, 0
    if (WEEKLY_CONTRIB_SHEET in inp.output_sheets
            and WEEKLY_CONTRIB_SHEET not in UNTOUCHED):
        hdr_wk = find_header_row(args.output_file, WEEKLY_CONTRIB_SHEET)
        weekly, wk_back = canon_cols(pd.read_excel(
            args.output_file, sheet_name=WEEKLY_CONTRIB_SHEET, header=hdr_wk))
    roi_curves, roi_back = (
        canon_cols(pd.read_excel(args.output_file, sheet_name=ROI_CURVES_SHEET))
        if ROI_CURVES_SHEET not in UNTOUCHED
        and ROI_CURVES_SHEET in inp.output_sheets else (None, {}))
    if ROI_CURVES_SHEET in built_sheets:
        roi_curves = tall
    handled = {STRUCTURE_SHEET, DATASHEET, SPENDS_SHEET, CONTRIB_SHEET}
    if curves is not None:
        handled.add(CURVES_SHEET)
    if roi_curves is not None:
        handled.add(ROI_CURVES_SHEET)
    if weekly is not None:
        handled.add(WEEKLY_CONTRIB_SHEET)

    # The modelling period ends where the datasheet ends -- this drives the
    # last-12-month window used for the response curves.
    period_end = pd.to_datetime(datasheet["Period name"]).max()

    updates: dict[str, pd.DataFrame] = {}
    failures: list[str] = []
    accepted: list[tuple[str, str, float]] = []

    for split in SPLITS:
        print(f"\n{'#' * 72}\n# {split.pooled} -> {', '.join(split.names)}\n{'#' * 72}")
        split.validate(list(data.columns))
        hits = scan_workbook(args.output_file, split.pooled)
        print(scan_report(args.output_file, split.pooled,
                                handled=handled, skipped=UNTOUCHED))

        # A sheet that contains the variable but is neither handled nor
        # explicitly declared untouched is a sheet nobody has decided about.
        # Stop rather than leave a stale pooled variable behind in a file that
        # is about to be read by another system.
        skipped = [h.sheet for h in hits if h.orientation != "not_found"
                   and h.sheet not in handled and h.sheet in UNTOUCHED]
        if skipped:
            print(f"  '{split.pooled}' also appears on "
                  f"{len(skipped)} sheet(s) left untouched:")
            print(f"    {', '.join(skipped)}")
            print(f"  These are the modelling software's own working sheets. "
                  f"Nothing we use\n  downstream reads them, so they are "
                  f"carried through unchanged and still\n  name the combined "
                  f"variable." + _hint(" Listed in UNTOUCHED in config.py.", ""))

        undecided = [h.sheet for h in hits if h.orientation != "not_found"
                     and h.sheet not in handled and h.sheet not in UNTOUCHED]
        if undecided:
            print(f"\nSTOPPED. '{split.pooled}' appears on sheet(s) this script "
                  f"does not know about: {', '.join(undecided)}.")
            print(_hint("Either add handling for them, or add them to UNTOUCHED "
                        "to record that leaving them alone is deliberate.",
                        "The tool needs a decision about that sheet before it "
                        "can write a file. Send this to whoever maintains it."))
            return 1

        row = structure.set_index("Variable").loc[split.pooled]
        for needed in ("Decay", "Lag"):
            if needed not in row.index:
                print(f"\nSTOPPED. No '{needed}' column in "
                      f"'{STRUCTURE_SHEET}'.\nColumns found: "
                      f"{', '.join(map(str, row.index))}\n"
                      + _hint("Run:  python mmm_splitter.py headers "
                              "\"<your output file>\"",
                              "Send this to whoever maintains the tool."))
                return 1
        stated, bracketed, how_read = parse_decay(row["Decay"])
        if stated is None:
            print(f"\nSTOPPED. Could not read a Decay value for "
                  f"'{split.pooled}' in '{STRUCTURE_SHEET}'.\n"
                  f"  cell contains: {row['Decay']!r}  ({how_read})")
            return 1
        if not isinstance(row["Decay"], (int, float)):
            print(f"  Decay cell {row['Decay']!r} read as {how_read}")
            if bracketed is not None:
                print(f"    carryover truncated after {bracketed} days "
                      f"(the bracketed number)")
        # Convert to a retention rate, which is what the adstock needs.
        decay = stated if DECAY_IS_RETENTION else 1.0 - stated
        if not 0.0 <= decay < 1.0:
            print(f"\nSTOPPED. '{split.pooled}' states Decay={stated}, which "
                  f"gives a retention rate of {decay} -- not a valid carryover.\n"
                  f"With DECAY_IS_RETENTION="
                  f"{DECAY_IS_RETENTION}, that reading cannot be right.\n"
                  + _hint("Run:  python mmm_splitter.py check-decay "
                          "\"<output file>\"",
                          "Send this to whoever maintains the tool."))
            return 1
        lag = 0 if pd.isna(row["Lag"]) else int(row["Lag"])
        if not DECAY_IS_RETENTION:
            print(f"  Decay column reads {stated} as 'share that decays away' "
                  f"-> retention {decay:.4g}  (config: DECAY_IS_RETENTION=False)")

        block = (datasheet[datasheet["Variable"] == split.pooled]
                 .sort_values("Period name"))
        coef = _num(row.get("Coefficients actual"))
        curve_type = parse_curve(row.get("Response Curve"))
        alpha = _num(row.get("Alpha"))
        ref_point = _ref_point(row)
        if ref_point:
            print(f"  reference point: {ref_point}")

        # Recency decay ('R 10,0% (6)') is a tapered finite filter, not a
        # truncated geometric. Recover its exact taps from the model's own
        # contribution; fall back to the geometric only if that fails.
        kernel = bracketed
        if bracketed is not None:
            if ref_point:
                print(f"  WARNING: recency kernel not recovered -- with a "
                      f"reference point the contribution\n    cannot be "
                      f"inverted. Using a truncated geometric, which only "
                      f"approximates it.")
            elif coef is None:
                print(f"  WARNING: no 'Coefficients actual' for "
                      f"'{split.pooled}', so the recency kernel cannot be "
                      f"recovered.\n    Using a truncated geometric, which "
                      f"only approximates it.")
            else:
                rk, msg = recover_recency_kernel(
                    block["Raw"], block["Contribution"], coef, curve_type,
                    alpha, lag, int(bracketed), label=split.pooled)
                if rk is not None:
                    kernel = rk
                    print(f"  recency decay: recovered the exact "
                          f"{len(rk)}-tap kernel from the model's contribution "
                          f"({msg})\n    taps "
                          + " ".join(f"{t:.4f}" for t in rk.taps)
                          + f"\n    (a truncated geometric would be "
                          + " ".join(f"{decay ** i:.4f}"
                                     for i in range(len(rk.taps))) + ")")
                else:
                    print(f"  WARNING: recency kernel could not be recovered "
                          f"({msg}).\n    Using a truncated geometric, which "
                          f"only approximates it.")
        check = verify_transform(block["Raw"], block["Contribution"], decay, lag,
                                 max_lag=kernel, coefficient=coef,
                                 curve=curve_type, alpha=alpha)
        print(explain(check, split.pooled))
        if not check["passed"]:
            reason = ACCEPT_TRANSFORM_MISMATCH.get(split.pooled)
            best = check.get("best_fit") or {}
            day_impact = share_impact(
                block["Raw"], decay, lag, kernel,
                best.get("decay", decay), best.get("lag", lag), None
            ) if best else float("inf")

            # What actually reaches the output: how far any single campaign's
            # total share moves between the two parameter sets.
            try:
                craw = pd.DataFrame({
                    c.name: pd.to_numeric(data[c.raw], errors="coerce")
                             .fillna(0.0)
                    for c in split.campaigns})
                impact = campaign_share_impact(
                    craw, decay, lag, kernel,
                    best.get("decay", decay), best.get("lag", lag), None)
            except Exception:
                impact = float("inf")

            if reason is None:
                print(f"  (if the best fit were used instead: "
                      f"{day_impact:.3f}% of the allocation moves between days, "
                      f"and\n   the largest change to any campaign's total "
                      f"share is {impact:.3f} percentage points)")
                failures.append(split.pooled)
                continue

            # Accepted by explicit decision -- but only if the disagreement
            # genuinely does not move the allocation.
            if (impact > MAX_ACCEPTED_SHARE_IMPACT
                    or day_impact > MAX_ACCEPTED_DAY_IMPACT):
                print(f"\n  ACCEPT_TRANSFORM_MISMATCH lists this variable, but "
                      f"the disagreement is too large:\n"
                      f"    campaign share change {impact:.3f} pp "
                      f"(limit {MAX_ACCEPTED_SHARE_IMPACT})\n"
                      f"    daily movement {day_impact:.3f}% "
                      f"(limit {MAX_ACCEPTED_DAY_IMPACT}%)\n  Refusing.")
                failures.append(split.pooled)
                continue
            print(f"  ACCEPTED despite the failure, by explicit decision"
                  + _hint(" in config.py", " (override)") + f":\n"
                  f"    \"{reason}\"\n"
                  f"    verified: campaign shares move at most {impact:.3f} pp, "
                  f"and {day_impact:.3f}% of\n    contribution shifts between "
                  f"days -- both within limits")
            accepted.append((split.pooled, reason, impact))

        cmap = split.as_dict()
        try:
            datasheet, r2, allocated = split_datasheet(
                datasheet, data, split.pooled, cmap, decay, lag,
                data_date_col=date_col, max_lag=kernel,
                accept_raw_mismatch=ACCEPT_RAW_MISMATCH.get(split.pooled),
                smoothing=SMOOTHING.get(split.pooled),
                rescale_to_pooled=RESCALE_TO_POOLED.get(split.pooled),
                ref_point=ref_point)
        except ValueError as e:
            # The datasheet checks (sums, dates, carryover) are findings about
            # the files, not faults in the tool: report them as a stop.
            print(f"\nSTOPPED at '{split.pooled}'. Nothing was written.\n{e}")
            return 1
        scalable = detect_scalable(structure, datasheet)
        ratios = {c: f for c in ("ROI", "CPU")
                  if (f := detect_ratio(structure, datasheet, c))}
        structure, r3 = split_structure(
            structure, datasheet, split.pooled, split.names,
            detect_effectiveness(structure, datasheet),
            scalable=scalable, ratio_formulas=ratios)
        model_dates = pd.to_datetime(datasheet["Period name"])
        spends, r5 = split_spends(spends, data, split.pooled, cmap,
                                  data_date_col=date_col,
                                  model_period=(model_dates.min(),
                                                model_dates.max()),
                                  rescale_to_pooled=RESCALE_TO_POOLED.get(
                                      split.pooled))
        rs5 = r5.get("rescaled_to_pooled")
        if rs5:
            print(f"  '{SPENDS_SHEET}' rescaled to the pooled figure on "
                  f"{rs5['days']} day(s)")
            print(f"    ({rs5['first']:%Y-%m-%d} to {rs5['last']:%Y-%m-%d}), "
                  f"{rs5['volume_share']:.2%} of spend affected")
        out = r5.get("outside_model_period")
        if out:
            print(f"  NOTE: on {out['rows']} row(s) OUTSIDE the modelling "
                  f"period ({out['first']:%Y-%m-%d} to {out['last']:%Y-%m-%d}),")
            print(f"    the pooled spend column did not equal the sum of its "
                  f"campaigns.")
            print(f"    Those rows are rewritten from the campaign columns, so "
                  f"{out['difference']:,.2f}")
            print(f"    of spend that was in the pooled column is not carried "
                  f"over. The model")
            print(f"    never used those days, so the split is unaffected.")

        contribs, r6 = split_contributions(
            contribs, allocated, split.pooled, split.names)
        if weekly is not None and split.pooled in weekly.columns:
            if "week_cov" not in locals():
                week_cov = weekly_coverage(contribs_source, weekly)
                last = week_cov[-1][1]
                if last < pd.to_datetime(contribs_source[DATE_COL]).max():
                    print(f"  NOTE: '{WEEKLY_CONTRIB_SHEET}' stops at "
                          f"{last:%Y-%m-%d}; the export leaves the final "
                          f"day(s) out of it, so the split does too.")
            weekly, r6w = split_contributions(
                weekly, weekly_allocation(allocated, week_cov),
                split.pooled, split.names, sheet_name=WEEKLY_CONTRIB_SHEET)
            print(f"  '{WEEKLY_CONTRIB_SHEET}' {r6w['rows']} weeks, contribution "
                  f"{r6w['pooled_contribution_total']:,.2f} -> "
                  f"{r6w['split_contribution_total']:,.2f}  (max row diff "
                  f"{r6w['max_row_difference']})")
        mode = (CURVE_MODE.get(split.pooled, CURVE_MODE.get("default",
                                                             "shared_shape"))
                if isinstance(CURVE_MODE, dict) else CURVE_MODE)
        model = None
        if mode in ("own_curve", "native_curve"):
            if "cost_series" not in locals():
                cost_series = load_cost_series(args.output_file)
                if cost_series is None:
                    print(f"  NOTE: no '{COSTS_SHEET}' sheet, so the campaign "
                          f"curves use a cost of 1 per period: they are\n    in "
                          f"KPI units, not KPI value, as the platform draws them "
                          f"in that case.")
            model = {"coef": coef, "curve": curve_type, "alpha": alpha,
                     "model_dates": pd.to_datetime(
                         block["Period name"]).to_numpy(),
                     "retention": decay, "lag": lag, "kernel": kernel,
                     "panel": r2["campaign_panel"], "cost": cost_series}
        curve_names = ([b.name for b in parse_blocks([str(c) for c in curves.columns])]
                       if curves is not None else [])
        r7 = None
        if curves is None:
            pass
        elif split.pooled not in curve_names:
            print(f"  NOTE: '{split.pooled}' has no curve on '{CURVES_SHEET}' "
                  f"(the platform draws none for a\n    variable with no spend "
                  f"in the curve window), so neither do its campaigns.")
        else:
            try:
                curves, r7 = split_curves(curves, allocated, data, split.pooled,
                                          cmap, period_end, date_col=date_col,
                                          model=model)
            except OwnCurveGateError as e:
                print(f"\nSTOPPED. {e}")
                return 1

        # Rows for the platform's own 'Structure' sheet.
        st_idx = structure.set_index(structure["Variable"].astype(str).str.strip())
        nat = ((r7 or {}).get("own_curve") or {}).get("native") or {}
        prow = []
        for c in split.campaigns:
            nv = nat.get(c.name)
            pct = nv.get("percent") if nv else None
            r_ = st_idx.loc[c.name.strip()] if c.name.strip() in st_idx.index else None
            prow.append({
                "name": c.name,
                "coefficient": nv["coefficient"] if nv else None,
                "alpha": nv["alpha"] if (nv and nv["type"] != "linear") else None,
                "curve_text": (f"{nv['type']} ({pct * 100:.1f}%)".replace(".", ",")
                               if (nv and pct is not None) else None),
                "roi": (float(r_["ROI"]) if r_ is not None and "ROI" in r_.index
                        and pd.notna(r_["ROI"]) else None),
                "cpu": (float(r_["CPU"]) if r_ is not None and "CPU" in r_.index
                        and pd.notna(r_["CPU"]) else None),
            })
        # Normalized and standardized coefficients, from the campaign's own
        # contribution series. The platform defines them as
        #   normalized   = sum(contribution) / sum(actual KPI)
        #   standardized = coefficient x sd(transformed) / sd(actual KPI)
        #                = sd(contribution) / sd(actual KPI), signed
        # (both confirmed on every variable of a real export to 1e-14), so a
        # campaign's figure is the combined variable's figure scaled by its
        # share of the contribution total and of its standard deviation.
        pooled_c = block["Contribution"].astype(float).to_numpy()
        tot_p, sd_p = float(pooled_c.sum()), float(pooled_c.std(ddof=1))
        for rowp in prow:
            if rowp["name"] in allocated.columns:
                ci = allocated[rowp["name"]].astype(float).to_numpy()
                rowp["sum_share"] = float(ci.sum()) / tot_p if tot_p else None
                rowp["sd_share"] = float(ci.std(ddof=1)) / sd_p if sd_p else None
        platform_rows[split.pooled] = prow
        if not nat:
            print(f"  NOTE: the new '{PLATFORM_STRUCTURE_SHEET}' rows for these "
                  f"campaigns carry no coefficient, curve\n    or alpha: those "
                  f"come from native curves, and this variable uses "
                  f"{(r7 or {}).get('curve_mode', 'no curve')}.")

        # The tall version of the same curves, built from the same numbers.
        if roi_curves is not None and r7 is not None:
            roi_curves, r8 = split_roi_curves(
                roi_curves, r7["built_curves"], r7["summary_points"],
                split.pooled, r7["campaigns_with_curves"])
            print(f"  '{ROI_CURVES_SHEET}' {r8['rows']} rows; "
                  f"{r8['rows_replaced']} replaced by {r8['blocks_written']} "
                  f"block(s) of {r8['rows_per_block']}")

        print(f"\n  contribution {r2['contribution_pooled']:,.2f} -> "
              f"{r2['contribution_split']:,.2f}  "
              f"(residual {r2['max_daily_residual_rounded']:.1e})")
        print(f"  spend        {r5['pooled_spend_total']:,.2f} -> "
              f"{r5['split_spend_total']:,.2f}")
        rs = r2.get("rescaled_to_pooled")
        if rs:
            print(f"  RESCALED TO THE POOLED FIGURE on {rs['days']} day(s) "
                  f"({rs['first']:%Y-%m-%d} to {rs['last']:%Y-%m-%d}),")
            print(f"    by explicit decision"
                  + _hint(" in config.py", " (override)") + ":")
            print(f"    \"{rs['reason']}\"")
            print(f"    On those days the pooled figure is taken as what the "
                  f"model used, and")
            print(f"    shared across campaigns in proportion to what each "
                  f"actually ran.")
            print(f"    {rs['volume_share']:.2%} of total volume is affected.")
        rp = r2.get("reference_point")
        if rp and rp.get("offset") is not None:
            print(f"  NOTE: reference point '{rp['reference_point']}': a constant "
                  f"{rp['offset']:,.4f} per day (seen alone on\n    "
                  f"{rp['dark_days']} dark day(s)) is shared by each campaign's "
                  f"share of the rest; only the part\n    above it is split by "
                  f"daily carryover.")
        elif rp:
            print(f"  WARNING: reference point '{rp['reference_point']}' but no "
                  f"dark days, so its constant\n    offset cannot be separated "
                  f"and is split by daily carryover with the rest.")
        sm = r2.get("smoothing")
        if sm:
            print(f"  smoothing applied to each campaign: "
                  f"{sm.get('window')}-day rolling average"
                  f"{' (centred)' if sm.get('centred') else ''}, matching what "
                  f"was done\n    to the pooled variable before modelling")
        note = r2.get("raw_mismatch_accepted")
        if note:
            print(f"  RAW MISMATCH ACCEPTED, by explicit decision"
                  + _hint(" in config.py", " (override)") + ":")
            print(f"    \"{note['reason']}\"")
            print(f"    the pooled Raw differs from the campaign columns on "
                  f"{note['days_affected']} day(s)")
            print(f"    by {note['difference']:,.0f} "
                  f"({note['difference_pct']:.2f}% of the total)")
            print(f"    the written Raw values use the campaign columns, so "
                  f"the new file")
            print(f"    is corrected. Contribution is NOT corrected -- it came "
                  f"from the")
            print(f"    model as fitted and still contains that component's "
                  f"effect.")
        print(f"  raw          {r2['raw_pooled']:,.0f} -> {r2['raw_split']:,.0f}"
              + ("   (difference is the accepted mismatch)" if note else ""))
        print(f"  '{CONTRIB_SHEET}' contribution "
              f"{r6['pooled_contribution_total']:,.2f} -> "
              f"{r6['split_contribution_total']:,.2f}  "
              f"(col {r6['column_position']}, {r6['columns']} cols, "
              f"max row diff {r6['max_row_difference']})")
        if r7 is not None:
            print(f"  '{CURVES_SHEET}' {r7["channels_total"]} channels, "
                  f"{r7['columns']} cols, {r7['total_rows']} rows; curves move "
                  f"{r7['curves_start_sheet_row']}  (residual {r7['max_curve_residual']:.1e})")
            print(f"    curve mode: {r7['curve_mode']}" + {
                "shared_shape": "  (campaigns differ by cost only; they do NOT sum "
                                "to the pooled curve)",
                "additive": "  (campaigns sum to the pooled curve)",
                "native_curve": "  (platform-type curve per campaign, fitted to "
                            "its share when the whole group moves)",
            "own_curve": "  (each campaign scaled alone, others held at "
                             "actual; built from the model)",
            }.get(r7['curve_mode'], ""))
            oc = r7.get("own_curve")
            if oc:
                print(f"    own_curve gate: the pooled curve rebuilt from the model "
                      f"to {oc['gate_error']:.1e} (relative)")
                print(f"    curve window {oc['window'][0]:%Y-%m-%d} to "
                      f"{oc['window'][1]:%Y-%m-%d}")
                if oc.get("kind") == "native":
                    print(f"    native curves: fitted to each campaign's share when "
                          f"the whole group moves")
                    for nm, nv in oc["native"].items():
                        pct = nv.get("percent")
                        sl = nv.get("slope_vs_own") or []
                        lo = min((e for _, e in sl), default=float("nan"))
                        hi = max((e for _, e in sl), default=float("nan"))
                        at1 = next((e for f, e in sl if f == 1.0), float("nan"))
                        print(f"      {nm}: {nv['type']}"
                              + (f" ({pct * 100:.1f}%)".replace(".", ",")
                                 if pct is not None else "")
                              + f"\n        coefficient {nv['coefficient']:,.6g}, alpha "
                                f"{nv['alpha']:.6g}; fit to the joint-move curve within "
                                f"{nv['fit_error']:.1%}"
                              + f"\n        slope vs own curve at current {at1:+.0%}, "
                                f"within +/-20% of current pressure {lo:+.0%} to "
                                f"{hi:+.0%}")
                    print(f"    native curves add up to the group's response within "
                          f"{oc['additivity_gap']:.1%} when all campaigns move "
                          f"together\n      ({oc['additivity_at_current']:+.1%} at "
                          f"current spend)")
                sh = oc['increments_share_of_pooled']
                how = ("less: on a saturating curve they compete for the same "
                       "headroom" if sh < 0.995 else
                       "more: on the rising part of an S curve each one's carryover "
                       "lifts the others" if sh > 1.005 else
                       "about the same: they barely overlap in time")
                print(f"    NOTE: at current spend the campaigns' own increments add "
                      f"up to {sh:.1%} of the pooled\n      variable's response in "
                      f"the window ({how}).\n      That gap belongs to no single "
                      f"campaign, so these curves do not sum to the pooled one.")
            print(f"    last-12m window: {r7['window_days']} days ending "
                  f"{period_end:%Y-%m-%d}")
            if r7.get("cpu_fell_back_to_full_period"):
                print(f"    NOTE: no spend in the last 12 months for "
                      f"{', '.join(r7['cpu_fell_back_to_full_period'])} -- "
                      f"cost per unit taken from the whole period instead")
            print(f"    cost per unit: " +
                  "  ".join(f"{k.split('_')[-1]} {v:.5f}"
                            for k, v in r7['cpu_last_12m'].items()))
        print(f"  effectiveness formula: {r3['effectiveness_formula']}")
        if r3["scaled"]:
            print("  scaled per campaign: " + ", ".join(
                f"{k} (by {v})" for k, v in r3["scaled"].items()))
        if r3["not_recomputed"]:
            print(f"  WARNING: could not work out how "
                  f"{', '.join(r3['not_recomputed'])} is calculated in this "
                  f"file -- left blank rather than guessed")
        for col, how in r3["recomputed"].items():
            print(f"  {col} recomputed as {how}")
        if r3["campaign_roi"]:
            print(f"  ROI  pooled {r3['pooled_roi']}  ->  " + "  ".join(
                f"{k.split('_')[-1]} {v:.4f}" for k, v in r3["campaign_roi"].items()))
        elif r3["campaign_cpu"]:
            print("  CPU  " + "  ".join(
                f"{k.split('_')[-1]} {v}" for k, v in r3["campaign_cpu"].items()))
        else:
            print("  no ROI or CPU column on this sheet -- nothing to recompute")

    if failures:
        print("\n" + "=" * 72)
        print(f"STOPPED. Nothing was written.")
        print(f"Transform check failed for: {', '.join(failures)}")
        print()
        print("The decay/lag in 'T structure' do not reproduce the contribution")
        print("numbers in 'T datasheet variables'. Either:")
        print("  (a) the modelling software states decay differently than this")
        print("      script rebuilds it -- in which case the script needs adjusting,")
        print("      not the file; or")
        print("  (b) something is off in the model export.")
        print()
        print("The 'closest' figures above show what the contribution numbers")
        print("actually support. If that looks right, tell whoever maintains this")
        print("tool -- do not edit the file until the check passes.")
        print("=" * 72)
        return 1

    # Headers were stripped for matching on load ('  Decay' -> 'Decay').
    # Put the original text back so what is written, and what is verified,
    # both match the file exactly.
    structure = structure.rename(columns=inp.structure_headers)

    if SPLITS:
        updates = {STRUCTURE_SHEET: structure,
                   DATASHEET: datasheet.rename(columns=ds_back),
                   SPENDS_SHEET: spends.rename(columns=sp_back),
                   CONTRIB_SHEET: contribs.rename(columns=ic_back),
                   }
        if curves is not None:
            updates[CURVES_SHEET] = curves
        if roi_curves is not None:
            updates[ROI_CURVES_SHEET] = roi_curves.rename(columns=roi_back)
        if weekly is not None:
            updates[WEEKLY_CONTRIB_SHEET] = weekly.rename(columns=wk_back)
    else:
        # Nothing to split: the run exists to add the curve sheets the export
        # lacks, and every other sheet is left exactly as it was.
        updates = {}
        if CURVES_SHEET in built_sheets:
            updates[CURVES_SHEET] = curves
        if ROI_CURVES_SHEET in built_sheets:
            updates[ROI_CURVES_SHEET] = roi_curves
    if not updates:
        print("\nNothing to change: no variable to split, and no curve sheet "
              "to build.")
        return 0

    if accepted:
        print("\n" + "-" * 72)
        print("NOTE: the transform check was overridden for:")
        for name, reason, impact in accepted:
            print(f"  {name}\n    reason: {reason}\n    "
                  f"largest campaign share change: {impact:.3f} "
                  f"percentage points")
        print("-" * 72)

    if not args.apply:
        print("\nAll checks passed. " + _hint(
            "Re-run with --apply to write the file.",
            "Generate the file below when ready."))
        return 0

    try:
        dest = write_output(args.output_file, updates, dest_path=args.dest,
                            column_change_sheets=COLUMN_CHANGE
                            | {WEEKLY_CONTRIB_SHEET},
                            header_rows={SPENDS_SHEET: hdr + 1,
                                         CONTRIB_SHEET: hdr_ic + 1,
                                         WEEKLY_CONTRIB_SHEET: hdr_wk + 1},
                            new_sheets=built_sheets,
                            platform_structure_rows=platform_rows)
    except PermissionError as e:
        print(f"\nSTOPPED. {e}")
        return 1
    v = verify_written(dest, args.output_file, updates)
    print(f"\nWritten: {dest}")
    for name, info in v["sheets"].items():
        print(f"  {name:<28} {info}")
    return 0


ROI_CURVES_SHEET = "T ROI curves"
ROI_MARKERS = ["Average", "Maximum", "Diminishing Point"]


def split_roi_curves(sheet: pd.DataFrame, built: dict, summary: dict,
                     pooled_var: str, active: list[str],
                     key_col: str = "Variable") -> tuple[pd.DataFrame, dict]:
    """
    'T ROI curves' -- the same curves as 'Response Curves', stacked instead of
    side by side.

    Each channel occupies one block: three marker rows (Average, Maximum,
    Diminishing Point, carrying Dummy Size 3) followed by the curve itself
    (Dummy Size 1, Reference blank). 'x-axis' is spend and 'KPI' is response.

    The numbers are taken from what was already built for 'Response Curves'
    rather than recomputed, so the two sheets cannot drift apart. A campaign
    with no curve on that sheet gets no block here either.
    """
    cols = [str(c) for c in sheet.columns]
    if key_col not in cols:
        raise ValueError(
            f"No '{key_col}' column on '{ROI_CURVES_SHEET}'. "
            f"Columns: {', '.join(cols)}")

    idx = sheet.index[sheet[key_col].astype(str) == pooled_var]
    if len(idx) == 0:
        raise ValueError(f"'{pooled_var}' not found on '{ROI_CURVES_SHEET}'.")

    block = sheet.loc[idx]
    marker_rows = block[block["Reference"].astype(str).isin(ROI_MARKERS)]
    curve_rows = block[~block["Reference"].astype(str).isin(ROI_MARKERS)]
    marker_size = (marker_rows["Dummy Size"].iloc[0]
                   if "Dummy Size" in cols and len(marker_rows) else 3)
    curve_size = (curve_rows["Dummy Size"].iloc[0]
                  if "Dummy Size" in cols and len(curve_rows) else 1)

    new_blocks = []
    for name in active:
        c = built[name]
        valid = ~np.isnan(np.asarray(c["bare"], dtype=float))

        rows = []
        for label in ROI_MARKERS:
            if label not in summary.get(name, {}):
                continue
            _, spend_at, response_at = summary[name][label]
            rows.append({key_col: name, "x-axis": spend_at,
                         "KPI": response_at, "Reference": label,
                         "Dummy Size": marker_size})
        for sp, kpi in zip(np.asarray(c["spend"])[valid],
                           np.asarray(c["bare"])[valid]):
            rows.append({key_col: name, "x-axis": float(sp),
                         "KPI": float(kpi), "Reference": np.nan,
                         "Dummy Size": curve_size})

        nb = pd.DataFrame(rows)
        # Product, Market and anything else are copied from the pooled block.
        for col in cols:
            if col not in nb.columns:
                nb[col] = block[col].iloc[0]
        new_blocks.append(nb[cols])

    before = sheet.loc[: idx[0] - 1] if idx[0] > 0 else sheet.iloc[:0]
    after = sheet.loc[idx[-1] + 1:]
    parts = [before] + new_blocks + [after]
    rebuilt = pd.concat([p for p in parts if len(p)], ignore_index=True)[cols]

    return rebuilt, {
        "sheet": ROI_CURVES_SHEET,
        "pooled_variable": pooled_var,
        "rows": f"{len(sheet)} -> {len(rebuilt)}",
        "rows_replaced": len(block),
        "blocks_written": len(new_blocks),
        "rows_per_block": len(new_blocks[0]) if new_blocks else 0,
        "campaigns": list(active),
    }


# ==========================================================================
#  DIAGNOSTIC COMMANDS
# ==========================================================================

def cmd_headers(path: str, sheet: str | None = None) -> int:
    """Show exact column headers, revealing hidden characters."""
    sheet = sheet or STRUCTURE_SHEET
    df = pd.read_excel(path, sheet_name=sheet, nrows=0)

    print(f"\n'{sheet}' -- {len(df.columns)} columns\n")
    print(f"  {'#':>3}  {'what Excel shows':<34} {'what Python sees':<40} issue")
    print("  " + "-" * 92)

    for i, col in enumerate(df.columns, 1):
        raw = str(col)
        shown = raw.strip().replace("\n", " ")
        notes = []
        if raw != raw.strip():
            notes.append("extra space")
        if "\n" in raw or "\r" in raw:
            notes.append("line break")
        if "  " in raw.strip():
            notes.append("double space")
        if "\xa0" in raw:
            notes.append("non-breaking space")
        print(f"  {i:>3}  {shown:<34} {raw!r:<40} {', '.join(notes)}")

    print()
    wanted = ["Variable", "Decay", "Response Curve", "Alpha", "Lag",
              "Category", "ROI", "CPU", "Effectiveness"]
    lookup = {str(c).strip().lower(): str(c) for c in df.columns}
    print("  columns the tool needs:")
    for w in wanted:
        hit = lookup.get(w.lower())
        if hit is None:
            print(f"    {w:<20} NOT FOUND")
        elif hit == w:
            print(f"    {w:<20} ok")
        else:
            print(f"    {w:<20} found as {hit!r}  (will be matched loosely)")
    print()
    return 0


def cmd_check_decay(path: str) -> int:
    """Is the Decay column retention, or the share that decays away?"""
    def _best_fit(raw, con, lag_hint):
        """Search decay and lag for whatever actually reproduces the contribution."""
        best = (0.0, None, None)
        for L in range(0, 8):
            for dv in np.round(np.arange(0.0, 0.96, 0.05), 2):
                s = _score(raw, con, dv, L)
                if not np.isnan(s) and s > best[0]:
                    best = (s, dv, L)
        return best


    ts = pd.read_excel(path, sheet_name=STRUCTURE_SHEET)
    ts.columns = [str(c).strip() for c in ts.columns]
    ds = read_canon(path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]

    ts = ts.set_index("Variable")
    media = ts[ts["Decay"].notna()].index.tolist()

    print(f"\nTesting {len(media)} variables with a Decay value.\n")
    print(f"  {'variable':<32} {'stated':>6} {'lag':>4} {'as ret':>9} {'as dec':>9} "
          f"{'best fit':>18}  verdict")
    print("  " + "-" * 104)

    votes = {"retention": 0, "decay": 0, "neither": 0}
    rows = []

    for v in media:
        blk = ds[ds["Variable"] == v]
        if blk.empty or blk["Raw"].abs().sum() == 0:
            continue
        stated, window, _ = parse_decay(ts.loc[v, "Decay"])
        if stated is None:
            continue
        lag = 0 if pd.isna(ts.loc[v, "Lag"]) else int(ts.loc[v, "Lag"])
        raw = blk["Raw"].values.astype(float)
        con = blk["Contribution"].values.astype(float)

        as_ret = _score(raw, con, stated, lag, window)
        as_dec = _score(raw, con, 1.0 - stated, lag, window)

        ret_ok = not np.isnan(as_ret)
        dec_ok = not np.isnan(as_dec)

        if not ret_ok and dec_ok:
            verdict = "decay"          # stated value impossible as a retention rate
        elif ret_ok and not dec_ok:
            verdict = "retention"
        elif ret_ok and as_ret >= 0.999 and as_ret >= as_dec:
            verdict = "retention"
        elif dec_ok and as_dec >= 0.999:
            verdict = "decay"
        else:
            verdict = "neither"
        votes[verdict] += 1
        rows.append((v, stated, as_ret, as_dec, verdict, lag))
        fmt = lambda x: "  n/a" if np.isnan(x) else f"{x:.5f}"
        if verdict == "neither":
            bs, bd, bl = _best_fit(raw, con, lag)
            bf = f"d={bd} lag={bl} {bs:.4f}"
        else:
            bf = ""
        print(f"  {v[:31]:<32} {stated:>6.2f} {lag:>4} {fmt(as_ret):>9} {fmt(as_dec):>9} "
              f"{bf:>18}  {verdict}")

    print("  " + "-" * 84)
    print(f"  retention: {votes['retention']}   decay: {votes['decay']}   "
          f"neither: {votes['neither']}\n")

    impossible = [r for r in rows if np.isnan(r[2])]
    if impossible:
        print(f"  Note: {len(impossible)} variable(s) state a Decay of 1.0 or more,")
        print("  which cannot be a retention rate -- an effect that never decays.")
        print("  As 'share that decays away', 1.0 simply means no carryover.\n")

    # Variables where both readings give the same number carry no information.
    ambiguous = [r for r in rows if abs(r[1] - 0.5) < 1e-9]
    if ambiguous:
        print(f"  {len(ambiguous)} variable(s) state exactly 0.50, where both "
              f"readings are identical -- no evidence either way.\n")

    decided = [r for r in rows if r[4] in ("retention", "decay")
               and abs(r[1] - 0.5) >= 1e-9]
    n_dec = sum(1 for r in decided if r[4] == "decay")
    n_ret = len(decided) - n_dec
    lagged_fail = [r for r in rows if r[4] == "neither" and r[5] > 0]
    if lagged_fail:
        print(f"  {len(lagged_fail)} of the unclear variable(s) have Lag > 0.\n")

    total = len(decided)
    votes = {"decay": n_dec, "retention": n_ret, "neither": votes["neither"]}
    if n_dec == total and total > 0:
        print("  CONCLUSION: the file states DECAY (the share that disappears).")
        print("  Set  DECAY_IS_RETENTION = False  in config.py\n")
    elif votes["retention"] == total and total > 0:
        print("  CONCLUSION: the file states RETENTION (the share that carries over).")
        print("  Set  DECAY_IS_RETENTION = True  in config.py\n")
    else:
        print("  CONCLUSION: mixed or unclear -- do not split until this is resolved.")
        print("  Variables scoring under 0.999 either way may use a different")
        print("  adstock form entirely, or their 'Raw' may not be the modelled")
        print("  input. Send this table to whoever maintains the script.\n")
    return 0


def cmd_check_effectiveness(path: str) -> int:
    """How is the Effectiveness column calculated?"""
    ts = pd.read_excel(path, sheet_name=STRUCTURE_SHEET)
    ts.columns = [str(c).strip() for c in ts.columns]
    ds = read_canon(path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]

    if "Effectiveness" not in ts.columns:
        print("No 'Effectiveness' column found.")
        return 1

    ds["Period name"] = pd.to_datetime(ds["Period name"])
    period_end = ds["Period name"].max()
    w12 = window_start(ds["Period name"], period_end)

    totals = ds.groupby("Variable")[["Contribution", "Spend", "Raw"]].sum()
    rows = ts[ts["Effectiveness"].notna()][["Variable", "Effectiveness"]]


    def ratio_variants(v, e):
        """Effectiveness divided by Contribution/Raw over various day subsets."""
        b = ds[ds["Variable"] == v]
        out = {}

        def cr(frame, label):
            C, R = frame["Contribution"].sum(), frame["Raw"].sum()
            out[label] = e / (C / R) if R and C else np.nan

        cr(b, "all days")
        cr(b[b["Raw"] > 0], "active days")
        cr(b[b["Period name"] >= w12], "last 12m")
        cr(b[(b["Period name"] >= w12) & (b["Raw"] > 0)], "12m active")
        return out


    print(f"\n  Modelling period ends {period_end:%Y-%m-%d}; "
          f"last-12m window starts {w12:%Y-%m-%d}\n")
    print(f"  {'variable':<30} {'all days':>10} {'active':>10} "
          f"{'last 12m':>10} {'12m active':>11}")
    print("  " + "-" * 76)
    collected = {k: [] for k in ("all days", "active days", "last 12m", "12m active")}
    for _, row in rows.iterrows():
        v = row["Variable"]
        if v not in totals.index:
            continue
        d = ratio_variants(v, float(row["Effectiveness"]))
        for k in collected:
            collected[k].append(d[k])
        print(f"  {str(v)[:29]:<30} " + " ".join(
            f"{d[k]:>10.5f}" if np.isfinite(d[k]) else f"{'n/a':>10}"
            for k in ("all days", "active days", "last 12m", "12m active")))
    print("  " + "-" * 76)
    # A file can carry stale Effectiveness values for variables added after the
    # column was last calculated, so judge by how MANY variables agree on a
    # constant ratio, not by the spread across all of them -- a few outliers
    # would otherwise bury an overwhelming majority.
    best, best_rate = None, 0.0
    for k, vals in collected.items():
        a = np.array([x for x in vals if np.isfinite(x)])
        if len(a) < 2:
            continue
        ref = float(np.median(a))
        agree = int(np.sum(np.abs(a - ref) / abs(ref) < 0.002))
        rate = agree / len(a)
        print(f"  {k:<14} ratio {ref:>12.5f}   {agree}/{len(a)} agree ({rate:.0%})")
        if rate > best_rate:
            best, best_rate = (k, ref), rate
    if best_rate < 0.75:
        best = None
    print()
    if best:
        k, K = best
        print(f"  ({best_rate:.0%} of variables agree -- the rest are likely stale,")
        print("   calculated before those variables were added to the model.)")
        mult = "" if abs(K - 1) < 1e-3 else f"{K:.6g} * "
        print(f"  FOUND IT: Effectiveness = {mult}Contribution / Raw,")
        print(f"            summed over: {k}\n")
    else:
        print("  None of these is constant. Effectiveness is built from something")
        print("  outside the datasheet -- most likely a coefficient from")
        print("  'T structure' rather than the contribution totals.\n")
        print("  Next thing to try: divide Effectiveness by 'Coefficients actual'")
        print("  for each variable and see whether THAT is constant.\n")
    return 0

    print(f"\n  {len(rows)} variable(s) have an Effectiveness value.\n")
    print(f"  {'variable':<30} {'Effectiveness':>14} {'eff/(C/R)':>12} "
          f"{'eff/(C/S)':>12} {'eff/C':>12}")
    print("  " + "-" * 84)

    r1, r2, r3 = [], [], []
    for _, row in rows.iterrows():
        v = row["Variable"]
        if v not in totals.index:
            continue
        C, S, R = totals.loc[v, ["Contribution", "Spend", "Raw"]]
        e = float(row["Effectiveness"])
        a = e / (C / R) if R and C else np.nan
        b = e / (C / S) if S and C else np.nan
        c = e / C if C else np.nan
        r1.append(a); r2.append(b); r3.append(c)
        print(f"  {str(v)[:29]:<30} {e:>14.5f} {a:>12.5f} {b:>12.5f} {c:>12.6g}")

    print("  " + "-" * 84)


    def verdict(vals, label, formula):
        v = np.array([x for x in vals if np.isfinite(x)])
        if len(v) < 2:
            return
        spread = (v.max() - v.min()) / abs(v.mean()) if v.mean() else np.inf
        if spread < 0.001:
            print(f"\n  MATCH: {label} is constant at {v.mean():.6g}")
            print(f"  So Effectiveness = {formula.replace('K', f'{v.mean():.6g}')}")
        else:
            print(f"  {label:<12} varies {v.min():.5g} to {v.max():.5g} -- not it")


    verdict(r1, "eff/(C/R)", "K * Contribution / Raw")
    verdict(r2, "eff/(C/S)", "K * Contribution / Spend")
    verdict(r3, "eff/C", "K * Contribution")

    print("\n  If none is constant, Effectiveness uses something not in the")
    print("  datasheet -- a coefficient, or a per-period rather than total figure.")
    print("  Send this table over and it can be worked out from the numbers.\n")
    return 0


def cmd_check_coefficients(path: str) -> int:
    """How are the normalized and standardized coefficients built?"""
    ts = pd.read_excel(path, sheet_name=STRUCTURE_SHEET)
    ts.columns = [str(c).strip() for c in ts.columns]
    ds = read_canon(path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]
    ds["Period name"] = pd.to_datetime(ds["Period name"])

    end = ds["Period name"].max()
    w12 = window_start(ds["Period name"], end)
    last12 = ds[ds["Period name"] >= w12]

    tot = ds.groupby("Variable")[["Contribution", "Spend", "Raw"]].sum()
    sd = ds.groupby("Variable")[["Contribution", "Raw"]].std()
    t12 = last12.groupby("Variable")[["Contribution", "Spend", "Raw"]].sum()

    C_all = float(ds["Contribution"].sum())
    C_12 = float(last12["Contribution"].sum())

    TARGETS = [c for c in ("Coefficients normalized", "Coefficients standardized")
               if c in ts.columns]
    if not TARGETS:
        print("Neither coefficient column found.")
        return 1

    CANDIDATES = {
        "share of total contribution":       lambda v: tot.loc[v, "Contribution"] / C_all,
        "share of last-12m contribution":    lambda v: t12.loc[v, "Contribution"] / C_12,
        "contribution / raw":                lambda v: (tot.loc[v, "Contribution"] / tot.loc[v, "Raw"]
                                                     if tot.loc[v, "Raw"] else np.nan),
        "beta * sd(raw)":                    lambda v: sd.loc[v, "Raw"],
        "sd(contribution)":                  lambda v: sd.loc[v, "Contribution"],
        "mean daily contribution":           lambda v: tot.loc[v, "Contribution"] / len(ds[ds.Variable == v]),
    }

    for target in TARGETS:
        rows = ts[ts[target].notna()]
        print(f"\n{'=' * 78}\n{target}\n{'=' * 78}")
        print(f"  {'candidate':<34} {'ratio (median)':>16} {'agree':>12}")
        print("  " + "-" * 66)
        winner = None
        for label, fn in CANDIDATES.items():
            vals = []
            for _, row in rows.iterrows():
                v = row["Variable"]
                if v not in tot.index:
                    continue
                try:
                    d = float(fn(v))
                    e = float(row[target])
                except Exception:
                    continue
                if d and np.isfinite(d) and e:
                    vals.append(e / d)
            a = np.array([x for x in vals if np.isfinite(x)])
            if len(a) < 3:
                print(f"  {label:<34} {'too few':>16}")
                continue
            ref = float(np.median(a))
            agree = int(np.sum(np.abs(a - ref) / abs(ref) < 0.002))
            rate = agree / len(a)
            flag = "  <-- MATCH" if rate >= 0.75 else ""
            print(f"  {label:<34} {ref:>16.6g} {agree:>5}/{len(a)} "
                  f"({rate:>4.0%}){flag}")
            if rate >= 0.75 and winner is None:
                winner = (label, ref)
        if winner:
            label, K = winner
            mult = "" if abs(K - 1) < 1e-3 else f"{K:.6g} * "
            print(f"\n  => {target} = {mult}{label}")
        else:
            print(f"\n  => no match. {target} is probably derived from the fitted")
            print(f"     model rather than the contribution data, in which case")
            print(f"     copying it to each campaign is the honest choice.")

    print(f"\n{'=' * 78}")
    print("  Sample of the actual values, for reference:")
    cols = ["Variable"] + TARGETS + (["Coefficients actual"]
                                     if "Coefficients actual" in ts.columns else [])
    print(ts[ts[TARGETS[0]].notna()][cols].head(12).to_string(index=False))
    print()
    return 0


def cmd_check_curves(path: str, only: str | None = None) -> int:
    """Compare the split campaigns on volume against efficiency."""
    sheet = pd.read_excel(path, sheet_name=CURVES_SHEET)
    headers = [str(c) for c in sheet.columns]
    SUFFIXES = [" Marginal Efficiency", " Efficiency", " pressure", " spend"]
    SUMMARY = ["Average", "Maximum", "Diminishing Point"]


    def block_cols(name: str) -> dict:
        out = {}
        for h in headers:
            if h == name:
                out["bare"] = h
            else:
                for suf in SUFFIXES:
                    if h == name + suf:
                        out[suf.strip()] = h
        return out


    def curve_rows(cols) -> pd.DataFrame:
        """The curve section: rows where this block has values."""
        sub = sheet[[c for c in cols.values() if c in sheet.columns]].copy()
        return sub.dropna(how="all")


    splits = [s for s in SPLITS if only is None or s.pooled == only]
    if not splits:
        print(f"'{only}' is not in SPLITS in config.py")
        return 1

    for sp in splits:
        present = [n for n in sp.names if n in headers]
        if not present:
            print(f"\n'{sp.pooled}': none of its campaigns are on "
                  f"'{CURVES_SHEET}' -- was the file split?")
            continue

        print(f"\n{'=' * 86}\n{sp.pooled}\n{'=' * 86}")
        rows = []
        for name in present:
            cols = block_cols(name)
            if "bare" not in cols:
                continue
            c = curve_rows(cols).reset_index(drop=True)
            resp = pd.to_numeric(c[cols["bare"]], errors="coerce").values
            spend = (pd.to_numeric(c[cols["spend"]], errors="coerce").values
                     if "spend" in cols else None)
            press = (pd.to_numeric(c[cols["pressure"]], errors="coerce").values
                     if "pressure" in cols else None)
            eff = (pd.to_numeric(c[cols["Efficiency"]], errors="coerce").values
                   if "Efficiency" in cols else None)
            marg = (pd.to_numeric(c[cols["Marginal Efficiency"]],
                                  errors="coerce").values
                    if "Marginal Efficiency" in cols else None)

            live = spend > 0 if spend is not None else np.zeros_like(resp, bool)
            cpu = (np.nanmedian(spend[live] / press[live])
                   if press is not None and live.any() else np.nan)
            rows.append({
                "campaign": name,
                "max_response": np.nanmax(resp),
                "max_pressure": np.nanmax(press) if press is not None else np.nan,
                "cpu": cpu,
                "eff_mid": (np.nanmedian(eff[live]) if eff is not None
                            and live.any() else np.nan),
                "marg_mid": (np.nanmedian(marg[live]) if marg is not None
                             and live.any() else np.nan),
            })

        df = pd.DataFrame(rows)
        if df.empty:
            continue
        df["volume_share"] = df.max_response / df.max_response.sum() * 100

        print("\n  HEIGHT OF THE CURVE -- set by share of volume, not by how well")
        print("  the campaign worked. A short burst looks small however good it was.")
        print(f"\n  {'campaign':<34} {'max response':>14} {'share':>8} {'rank':>6}")
        print("  " + "-" * 66)
        for _, r in df.sort_values("max_response", ascending=False).iterrows():
            rank = int(df.max_response.rank(ascending=False)[r.name])
            print(f"  {r.campaign[:33]:<34} {r.max_response:>14,.2f} "
                  f"{r.volume_share:>7.1f}% {rank:>6}")

        print("\n  EFFICIENCY -- response per unit of spend. This is where a cheap")
        print("  campaign should look good, whatever its size.")
        print(f"\n  {'campaign':<34} {'cost per unit':>14} {'efficiency':>12} "
              f"{'rank':>6}")
        print("  " + "-" * 70)
        ranks = df.eff_mid.rank(ascending=False)
        for _, r in df.sort_values("eff_mid", ascending=False).iterrows():
            print(f"  {r.campaign[:33]:<34} {r.cpu:>14.5f} {r.eff_mid:>12,.2f} "
                  f"{int(ranks[r.name]):>6}")

        # Where the two disagree is the thing worth explaining to a client.
        vol_rank = df.max_response.rank(ascending=False)
        eff_rank = df.eff_mid.rank(ascending=False)
        df["gap"] = vol_rank - eff_rank
        flipped = df[df.gap.abs() >= 2]
        if len(flipped):
            print("\n  CAMPAIGNS THAT RANK DIFFERENTLY ON THE TWO MEASURES:")
            for _, r in flipped.iterrows():
                v, e = int(vol_rank[r.name]), int(eff_rank[r.name])
                direction = ("smaller than it is efficient" if v > e
                             else "bigger than it is efficient")
                print(f"      {r.campaign[:40]:<42} volume #{v}, efficiency #{e}"
                      f"  -- {direction}")
            print("\n  A campaign low on volume but high on efficiency did well per")
            print("  euro and simply ran less. That is the usual case for a sale")
            print("  burst, and it is not the model saying the campaign was weak.")
        else:
            print("\n  The two measures rank the campaigns the same way.")

    print("\n" + "=" * 86)
    print("  WHAT NONE OF THIS CAN TELL YOU")
    print("=" * 86)
    print("  All these curves share one shape, because the model estimated one")
    print("  saturation curve for the pooled variable. The split cannot say a")
    print("  sale click is worth more than a brand click -- the model never")
    print("  compared them. Differences here come from volume and from cost,")
    print("  not from campaign quality.")
    print()
    print("  If the difference between campaign types matters commercially, that")
    print("  is a case for modelling them separately, not for a better split.")
    print()
    return 0


def cmd_check_smoothing(out_path: str, imp_path: str, pooled: str) -> int:
    """Work out what smoothing was applied to a pooled variable."""







    cfg = next((s for s in SPLITS if s.pooled == pooled), None)
    if cfg is None:
        print(f"'{pooled}' is not in SPLITS in config.py")
        return 1

    ds = read_canon(out_path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]
    data = pd.read_excel(imp_path, sheet_name=DATA_SHEET)
    data.columns = [str(c).strip() for c in data.columns]
    data, _ = add_composite_columns(data, COMPOSITE_COLUMNS)

    block = ds[ds["Variable"] == pooled].copy()
    if block.empty:
        print(f"'{pooled}' not found in '{DATASHEET}'")
        return 1
    block["Period name"] = pd.to_datetime(block["Period name"])
    block = block.sort_values("Period name").reset_index(drop=True)

    date_col = None
    for c in data.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(data[c], errors="coerce")
        if parsed.notna().sum() >= 0.9 * len(data) and parsed.dropna().is_unique:
            date_col = c
            break
    if date_col is None:
        print("No date column found on the DATA sheet")
        return 1
    data[date_col] = pd.to_datetime(data[date_col])

    aligned = block[["Period name"]].merge(
        data, left_on="Period name", right_on=date_col, how="left")

    parts = np.zeros(len(block))
    for c in cfg.campaigns:
        if c.raw not in aligned.columns:
            print(f"'{c.raw}' is not a column on the DATA sheet")
            return 1
        parts = parts + pd.to_numeric(aligned[c.raw], errors="coerce").fillna(0.0).values

    pooled_raw = block["Raw"].astype(float).values
    diff = pooled_raw - parts
    off = np.abs(diff) > 0.5

    print(f"\n{'=' * 74}\n{pooled}\n{'=' * 74}")
    print(f"  days                     {len(block)}")
    print(f"  days that do not add up  {int(off.sum())} ({off.mean():.1%})")
    print(f"  pooled total             {pooled_raw.sum():,.2f}")
    print(f"  sum of campaigns         {parts.sum():,.2f}")
    print(f"  difference               {diff.sum():,.2f}")

    if not off.any():
        print("\n  These add up. No smoothing to find.\n")
        return 0

    # A rolling average leaves the TOTAL almost unchanged while moving values
    # between days. A missing component changes the total.
    rel = abs(diff.sum()) / pooled_raw.sum() if pooled_raw.sum() else 1
    print(f"\n  difference as a share of the total: {rel:.4%}")
    if rel > 0.01:
        print("  -> the totals differ by more than 1%, so this looks like a")
        print("     MISSING COMPONENT rather than smoothing. " + _hint(
            "Try check-sum.", "Try 'which columns are missing?'."))
    else:
        print("  -> the totals barely differ, so the same volume is present but")
        print("     spread differently across days. That is what smoothing does.")

    print(f"\n  testing rolling averages applied to the campaign sum...")
    print(f"      {'window':>8} {'centred':>9} {'worst day diff':>16} {'verdict':>12}")
    print("  " + "-" * 52)

    series = pd.Series(parts)
    best = None
    for window in range(2, 22):
        for centred in (False, True):
            sm = series.rolling(window, center=centred, min_periods=1).mean().values
            worst = float(np.nanmax(np.abs(sm - pooled_raw)))
            verdict = "MATCH" if worst < 0.5 else ""
            if window <= 12 or verdict:
                print(f"      {window:>8} {str(centred):>9} {worst:>16,.2f} "
                      f"{verdict:>12}")
            if best is None or worst < best[0]:
                best = (worst, window, centred)

    print("  " + "-" * 52)
    w, win, cen = best
    if w < 0.5:
        print(f"\n  FOUND IT: a {win}-day rolling average"
              f"{' (centred)' if cen else ''} reproduces the pooled series.")
        if INTERFACE == "app":
            print(f"\n  Enter {win} as this variable's smoothing window"
                  f"{' and tick Centred' if cen else ''} in step 2, then run "
                  f"the checks again.")
        else:
            print(f"\n  Add to config.py:")
            print(f"      SMOOTHING = {{")
            print(f"          \"{pooled}\": {{\"window\": {win}, "
                  f"\"centred\": {cen}}},")
        print(f"      }}")
        print(f"\n  The same smoothing is then applied to each campaign, which")
        print(f"  makes them sum to the pooled series exactly -- a rolling average")
        print(f"  is linear, so smoothing the parts and adding them is the same as")
        print(f"  smoothing the total.")
    else:
        print(f"\n  No plain rolling average reproduces it. Closest was a {win}-day"
              f"{' centred' if cen else ''} window,")
        print(f"  still {w:,.2f} out on the worst day.")
        print(f"\n  The smoothing may have been applied only to certain days, or")
        print(f"  by hand. In that case the campaign columns cannot be made to")
        print(f"  match by any single rule -- ask how the variable was built.")
        worst_i = int(np.argmax(np.abs(diff)))
        print(f"\n  worst day {block['Period name'].iloc[worst_i]:%Y-%m-%d}: "
              f"pooled {pooled_raw[worst_i]:,.2f}, campaigns {parts[worst_i]:,.2f}")
        print(f"  days affected run from "
              f"{block['Period name'][off].min():%Y-%m-%d} to "
              f"{block['Period name'][off].max():%Y-%m-%d}")
    print()
    return 0


def cmd_check_sum(out_path: str, imp_path: str, pooled: str) -> int:
    """Find what a pooled variable contains that the config does not."""
    cfg = next((s for s in SPLITS if s.pooled == pooled), None)
    if cfg is None:
        print(f"'{pooled}' is not in SPLITS in config.py")
        return 1

    ds = read_canon(out_path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]
    data = pd.read_excel(imp_path, sheet_name=DATA_SHEET)
    data.columns = [str(c).strip() for c in data.columns]
    data, _ = add_composite_columns(data, COMPOSITE_COLUMNS)

    block = ds[ds["Variable"] == pooled].copy()
    if block.empty:
        print(f"'{pooled}' not found in '{DATASHEET}'")
        return 1
    block["Period name"] = pd.to_datetime(block["Period name"])
    block = block.sort_values("Period name").reset_index(drop=True)

    # find the date column on DATA
    date_col = None
    for c in data.columns:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            parsed = pd.to_datetime(data[c], errors="coerce")
        if parsed.notna().sum() >= 0.9 * len(data) and parsed.dropna().is_unique:
            date_col = c
            break
    if date_col is None:
        print("No date column found on the DATA sheet")
        return 1
    data[date_col] = pd.to_datetime(data[date_col])

    aligned = block[["Period name"]].merge(
        data, left_on="Period name", right_on=date_col, how="left")


    def num(col):
        return pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).values


    pooled_raw = block["Raw"].astype(float).values
    parts = np.zeros_like(pooled_raw)
    for c in cfg.campaigns:
        if c.raw in aligned.columns:
            parts = parts + num(c.raw)
        else:
            print(f"  WARNING: '{c.raw}' is not a column on the DATA sheet")

    diff = pooled_raw - parts
    off = np.abs(diff) > 0.5

    print(f"\n{'=' * 74}\n{pooled}\n{'=' * 74}")
    print(f"  configured campaigns : {len(cfg.campaigns)}")
    print(f"  days                 : {len(block)}")
    print(f"  days that do not add up: {int(off.sum())} "
          f"({off.mean():.0%})")
    print(f"  pooled total         : {pooled_raw.sum():,.2f}")
    print(f"  sum of campaigns     : {parts.sum():,.2f}")
    print(f"  shortfall            : {diff.sum():,.2f} "
          f"({diff.sum() / pooled_raw.sum() if pooled_raw.sum() else 0:+.2%})")

    if not off.any():
        print("\n  These columns DO add up. Nothing missing.\n")
        return 0

    if (diff < -0.5).any():
        print(f"\n  NOTE: on {int((diff < -0.5).sum())} day(s) the campaigns sum to "
              f"MORE than the pooled variable,\n  so this is not simply a missing "
              f"component -- a configured column may not\n  belong to this pooled "
              f"variable at all.")

    # Search DATA for a column that explains the shortfall.
    print(f"\n  searching {len(data.columns)} DATA columns for the shortfall...")
    used = {c.raw for c in cfg.campaigns}
    best = []
    for col in data.columns:
        if col == date_col or col in used:
            continue
        try:
            v = pd.to_numeric(aligned[col], errors="coerce").fillna(0.0).values
        except Exception:
            continue
        if not np.isfinite(v).all() or np.allclose(v, 0):
            continue
        exact = np.abs(v - diff).max()
        if exact < 0.5:
            best.append((0.0, col, "EXACT MATCH"))
            continue
        resid = np.abs(v - diff)
        if resid.mean() < np.abs(diff).mean() * 0.25:
            best.append((resid.mean(), col, "close"))

    best.sort()
    if best:
        print("\n  columns that would account for the shortfall:")
        for score, col, tag in best[:6]:
            print(f"      {col:<44} {tag}")
        print(_hint("\n  Add the matching one to this Split in config.py as "
                    "another\n  Campaign(...) entry, with its own spend column.",
                    "\n  Add the matching one as another row in this variable's "
                    "table in step 2,\n  with its own spend column."))

        # If the pooled variable's NAME says a component is excluded but its Raw
        # includes it, then Raw is not what the model was fitted on -- and that
        # would also explain a transform check that never quite reaches 1.0.
        # Test it: does removing the component make the relationship exact?
        exact = [c for sc, c, tag in best if tag == "EXACT MATCH"]
        if exact:
            col = exact[0]
            ts = pd.read_excel(out_path, sheet_name=STRUCTURE_SHEET)
            ts.columns = [str(c).strip() for c in ts.columns]
            ts = ts.set_index("Variable")
            try:
                stated, window, _ = parse_decay(ts.loc[pooled, "Decay"])
                lag = 0 if pd.isna(ts.loc[pooled, "Lag"]) else int(ts.loc[pooled, "Lag"])
                ret = stated if DECAY_IS_RETENTION else 1.0 - stated
                con = block["Contribution"].astype(float).values
                with_it = _score(pooled_raw, con, ret, lag, window)
                without = _score(pooled_raw - num(col), con, ret, lag, window)
                print(f"\n  Does '{col}' belong in this variable at all?")
                print(f"      transform score using Raw as stored      "
                      f"{with_it:.6f}")
                print(f"      transform score with '{col}' removed  "
                      f"{without:.6f}")
                PASS = 0.999
                if without >= PASS > with_it:
                    print(f"\n      DECISIVE: removing it makes the relationship")
                    print(f"      exact, while keeping it does not. The stored Raw")
                    print(f"      includes a component the model was NOT fitted on.")
                    print(f"      Do not simply add it as a campaign -- ask how this")
                    print(f"      variable was built before splitting it.")
                elif with_it >= PASS > without:
                    print(f"\n      DECISIVE: it belongs here. Add it as another")
                    print(f"      Campaign(...) entry with its spend column.")
                elif without > with_it:
                    print(f"\n      Removing it fits better, though neither reaches")
                    print(f"      {PASS}. Suggestive that the stored Raw includes")
                    print(f"      something the model was not fitted on. Ask how the")
                    print(f"      variable was built.")
                else:
                    print(f"\n      Keeping it fits at least as well, so it probably")
                    print(f"      belongs. Add it as another Campaign(...) entry.")
            except Exception as _e:
                pass
    else:
        print("\n  No single column matches the shortfall. The pooled variable may")
        print("  be built from more than one extra component, or from something")
        print("  other than a plain sum of DATA columns.")
        worst = int(np.argmax(np.abs(diff)))
        print(f"\n  worst day {block['Period name'].iloc[worst]:%Y-%m-%d}: "
              f"pooled {pooled_raw[worst]:,.2f}, campaigns {parts[worst]:,.2f}, "
              f"missing {diff[worst]:,.2f}")
    print()
    return 0


def cmd_check_transform(path: str, only: str | None = None,
                        import_file: str | None = None) -> int:
    """Investigate a variable whose transform check fails."""
    ts = pd.read_excel(path, sheet_name=STRUCTURE_SHEET)
    ts.columns = [str(c).strip() for c in ts.columns]
    ds = read_canon(path, sheet_name=DATASHEET)
    ds.columns = [str(c).strip() for c in ds.columns]
    ts = ts.set_index("Variable")

    targets = [only] if only else [
        v for v in ts.index if pd.notna(ts.loc[v, "Decay"])]


    def fine_search(raw, con, lag_hint, window):
        """Best score over decay, lag and whether the window is applied."""
        best = (0.0, None, None, None)
        for use_window in ({None, window} if window else {None}):
            for L in range(0, 8):
                for d in np.round(np.arange(0.0, 0.99, 0.01), 2):
                    s = _score(raw, con, d, L, use_window)
                    if np.isfinite(s) and s > best[0]:
                        best = (s, d, L, use_window)
        return best


    for v in targets:
        if v not in ts.index:
            print(f"'{v}' not in {STRUCTURE_SHEET}")
            continue
        blk = ds[ds["Variable"] == v]
        if blk.empty:
            continue
        raw = blk["Raw"].values.astype(float)
        con = blk["Contribution"].values.astype(float)
        stated, window, how = parse_decay(ts.loc[v, "Decay"])
        if stated is None:
            continue
        lag = 0 if pd.isna(ts.loc[v, "Lag"]) else int(ts.loc[v, "Lag"])
        retention = stated if DECAY_IS_RETENTION else 1.0 - stated

        declared = _score(raw, con, retention, lag, window)
        if declared >= 0.999 and not only:
            continue

        print(f"\n{'=' * 74}\n{v}\n{'=' * 74}")
        print(f"  Decay cell        {ts.loc[v, 'Decay']!r}  ->  retention "
              f"{retention:.4g}, window {window}, lag {lag}")
        print(f"  score as declared {declared:.6f}")

        best, bd, bl, bw = fine_search(raw, con, lag, window)
        print(f"  best achievable   {best:.6f}  (retention {bd}, lag {bl}, "
              f"window {bw})")
        if best < 0.999:
            print("  -> no decay reproduces this variable. The problem is not the")
            print("     decay value; something else about the variable differs.")

        # Where does the relationship break?
        a = geometric_adstock(raw, retention, peak_lag=lag, max_lag=window)
        days = len(blk)
        zero_raw = int((raw == 0).sum())
        neg_raw = int((raw < 0).sum())
        neg_con = int((con < 0).sum())
        con_floor = float(np.nanmin(con))
        tied = int((con == con_floor).sum())
        dark_but_paid = int(((a <= 1e-12) & (np.abs(con) > 1e-9)).sum())

        print(f"\n  days                     {days}")
        print(f"  raw is zero on           {zero_raw} ({zero_raw / days:.0%})")
        if neg_raw:
            print(f"  raw is NEGATIVE on       {neg_raw}  <-- unusual")
        if neg_con:
            print(f"  contribution negative on {neg_con}")
        print(f"  contribution floor       {con_floor:,.4f} on {tied} day(s)")
        if dark_but_paid:
            print(f"  contribution non-zero with no carryover on {dark_but_paid} "
                  f"day(s)  <-- unexplained")

        # Is the contribution/adstock ratio stable? A single saturation curve
        # implies it varies smoothly with volume, not erratically.
        live = a > 0
        if live.sum() > 30:
            ratio = con[live] / a[live]
            print(f"  contribution / adstock   median {np.median(ratio):,.6g}, "
                  f"spread {np.percentile(ratio, 5):,.4g} to "
                  f"{np.percentile(ratio, 95):,.4g}")

        # Is the model applying saturation BEFORE adstock rather than after?
        # contribution = beta * adstock(f(raw)) instead of beta * f(adstock(raw)).
        # Both are common; they give the same shape of answer but a different
        # series, so rebuilding the wrong order caps the score below 1.
        alpha_col = ts.loc[v].get("Alpha")
        best_pre = (0.0, None)
        for a in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
            pre = np.power(np.maximum(raw, 0.0), a)
            sc = _score(pre, con, retention, lag, window)
            if sc > best_pre[0]:
                best_pre = (sc, a)
        print(f"\n  if saturation is applied BEFORE adstock:")
        print(f"      best {best_pre[0]:.6f} at power {best_pre[1]}"
              f"{'   <-- explains it' if best_pre[0] >= 0.999 else ''}")
        if pd.notna(alpha_col):
            sc = _score(np.power(np.maximum(raw, 0.0), float(alpha_col)),
                        con, retention, lag, window)
            print(f"      using the sheet's Alpha={alpha_col}: {sc:.6f}")

        # How much does any of this actually change the split? If the shares are
        # near-identical under the declared and best-fit parameters, the
        # disagreement is academic; if they move, it matters.
        print(f"\n  does it change the answer?")
        a_declared = geometric_adstock(raw, retention, peak_lag=lag, max_lag=window)
        a_best = geometric_adstock(raw, bd if bd is not None else retention,
                                   peak_lag=bl or 0, max_lag=bw)
        tot_d, tot_b = a_declared.sum(), a_best.sum()
        if tot_d > 0 and tot_b > 0:
            # daily share of the period total, under each parameter set
            sd, sb = a_declared / tot_d, a_best / tot_b
            moved = np.abs(sd - sb).sum() / 2 * 100
            print(f"      {moved:.3f}% of the allocation would move if the best "
                  f"fit were used")
            if moved < 1.0:
                print("      -> below the 1% limit: the split is practically the")
                print("         same either way. It can be accepted by listing the")
                print("         variable in ACCEPT_TRANSFORM_MISMATCH in config.py")
            else:
                print("      -> above the 1% limit: this WOULD change the split.")
                print("         Resolve it rather than overriding.")

        # The number that actually decides it: how much does each campaign's
        # TOTAL share change? Day-level movement partly cancels when summed per
        # campaign, and the per-campaign totals are what reach the output.
        cfg = next((sp for sp in SPLITS if sp.pooled == v), None)
        if cfg is not None:
            try:
                data = (pd.read_excel(import_file, sheet_name="DATA")
                        if import_file else None)
            except Exception:
                data = None
            if data is not None:
                data.columns = [str(c).strip() for c in data.columns]
                cr = pd.DataFrame({c.name: pd.to_numeric(data[c.raw], errors="coerce")
                                   .fillna(0.0) for c in cfg.campaigns})

                def shares(dec, lg, win):
                    ad = cr.apply(lambda col: geometric_adstock(
                        col.values, dec, peak_lag=lg, max_lag=win),
                        axis=0, result_type="broadcast")
                    t = ad.sum().sum()
                    return ad.sum() / t * 100 if t else None

                s_dec = shares(retention, lag, window)
                s_best = shares(bd if bd is not None else retention, bl or 0, bw)
                if s_dec is not None and s_best is not None:
                    print(f"\n  per-campaign share of the split, "
                          f"under each parameter set:")
                    print(f"      {'campaign':<34} {'declared':>9} {'best fit':>9} "
                          f"{'change':>8}")
                    worst = 0.0
                    for n in cfg.names:
                        d_, b_ = float(s_dec[n]), float(s_best[n])
                        worst = max(worst, abs(d_ - b_))
                        print(f"      {n[:33]:<34} {d_:>8.3f}% {b_:>8.3f}% "
                              f"{b_ - d_:>+7.3f}")
                    print(f"\n      largest change to any campaign's share: "
                          f"{worst:.3f} percentage points")
                    if worst < 0.25:
                        print("      -> the campaigns end up with the same numbers")
                        print("         either way. Safe to accept.")
                    else:
                        print("      -> this would visibly change what the client")
                        print("         sees. Resolve rather than override.")
            else:
                print("\n  (pass the import file as a third argument to see how "
                      "much\n   each campaign's share would actually change)")

        # Does the relationship hold within a single year but not across years?
        if "Period name" in blk.columns:
            d = pd.to_datetime(blk["Period name"])
            by_year = {}
            for y in sorted(d.dt.year.unique()):
                m = (d.dt.year == y).values
                if m.sum() > 60:
                    by_year[y] = _score(raw[m], con[m], retention, lag, window)
            if len(by_year) > 1:
                print("\n  score computed year by year:")
                for y, sc in by_year.items():
                    flag = "" if sc >= 0.999 else "   <-- breaks here"
                    print(f"      {y}   {sc:.6f}{flag}")
                print("  (a good score within each year but a poor one overall")
                print("   means the relationship shifts between years -- the")
                print("   variable may have been rescaled mid-model)")
    print()
    return 0


def cmd_compare(orig: str, new: str) -> int:
    """Show what changed between the original and the written result."""
    orig_path, new_path = Path(orig), Path(new)
    for p in (orig_path, new_path):
        if not p.exists():
            print(f"File not found: {p}")
            return 1
    if orig_path.resolve() == new_path.resolve():
        print("Both paths point at the same file.")
        return 1

    split = SPLITS[0]
    pooled, campaigns = split.pooled, split.names

    print(f"\n  original : {orig_path.name}  ({orig_path.stat().st_size:,} bytes)")
    print(f"  result   : {new_path.name}  ({new_path.stat().st_size:,} bytes)")
    if orig_path.stat().st_size == new_path.stat().st_size:
        print("  NOTE: identical file sizes -- suspicious")

    # Things openpyxl silently discards when it re-saves a workbook.
    wo, wn = load_workbook(orig_path), load_workbook(new_path)
    feat = []
    for label, fn in [
        ("charts", lambda w: sum(len(w[s]._charts) for s in w.sheetnames)),
        ("images", lambda w: sum(len(w[s]._images) for s in w.sheetnames)),
        ("conditional formatting",
         lambda w: sum(len(list(w[s].conditional_formatting)) for s in w.sheetnames)),
        ("defined names", lambda w: len(w.defined_names)),
        ("data validations",
         lambda w: sum(len(w[s].data_validations.dataValidation) for s in w.sheetnames)),
    ]:
        try:
            a, b = fn(wo), fn(wn)
        except Exception:
            continue
        if a or b:
            feat.append((label, a, b))
    if feat:
        print("\n  workbook features:")
        for label, a, b in feat:
            flag = "  <-- LOST" if b < a else ""
            print(f"      {label:<24} {a} -> {b}{flag}")

    orig = pd.read_excel(orig_path, sheet_name=None)
    new = pd.read_excel(new_path, sheet_name=None)

    print(f"\n  Looking for '{pooled}' to be gone and these to be present:")
    for c in campaigns:
        print(f"      {c}")

    print(f"\n  {'sheet':<30} {'rows':>15} {'cols':>11}  {'pooled':>7} {'new':>4}  status")
    print("  " + "-" * 88)

    changed = 0
    for name in orig:
        o = orig[name]
        n = new.get(name)
        if n is None:
            print(f"  {name:<30} {'MISSING from result':>32}")
            continue

        def is_texty(series):
            # pandas 3 gives string columns a 'str' dtype rather than 'object'
            return series.dtype == object or str(series.dtype).startswith("str")

        def has_pooled(df):
            if pooled in [str(c) for c in df.columns]:
                return True
            return any(df[c].astype(str).eq(pooled).any()
                       for c in df.columns if is_texty(df[c]))

        def n_new(df):
            cols = [str(c) for c in df.columns]
            found = sum(1 for c in campaigns if c in cols)
            if found:
                return found
            best = 0
            for c in df.columns:
                if is_texty(df[c]):
                    vals = set(df[c].astype(str).unique())
                    best = max(best, sum(1 for x in campaigns if x in vals))
            return best

        rows = f"{len(o):,} -> {len(n):,}"
        cols = f"{len(o.columns)} -> {len(n.columns)}"
        still = "YES" if has_pooled(n) else "gone"
        added = n_new(n)

        if len(o) != len(n) or len(o.columns) != len(n.columns):
            status = "CHANGED"
            changed += 1
        elif o.equals(n):
            status = "identical"
        else:
            status = "values differ"
            changed += 1

        print(f"  {name:<30} {rows:>15} {cols:>11}  {still:>7} {added:>4}  {status}")

    print("  " + "-" * 88)
    print(f"  {changed} sheet(s) changed\n")

    # Where a sheet we did not intend to touch shows differences, say exactly
    # what they are -- a dtype artefact of re-reading is harmless, an actual
    # value change is not.
    for name in orig:
        o, n = orig[name], new.get(name)
        if n is None or len(o) != len(n) or len(o.columns) != len(n.columns):
            continue
        if o.equals(n):
            continue
        diffs = []
        for c in o.columns:
            if c not in n.columns:
                continue
            a, b = o[c], n[c]
            if a.equals(b):
                continue
            try:
                an = pd.to_numeric(a, errors="coerce")
                bn = pd.to_numeric(b, errors="coerce")
                if an.notna().any() and np.allclose(an.fillna(0), bn.fillna(0),
                                                    rtol=1e-12, atol=1e-12):
                    diffs.append((c, f"same numbers: {a.dtype} -> {b.dtype}",
                                  (a.dropna().iloc[0], b.dropna().iloc[0])
                                  if a.notna().any() and b.notna().any() else None))
                    continue
                mask = ~((a.isna() & b.isna()) | (a.astype(str) == b.astype(str)))
                i = int(mask.idxmax()) if mask.any() else 0
                diffs.append((c, f"row {i + 2}", (a.iloc[i], b.iloc[i])))
            except Exception:
                diffs.append((c, "differs", None))
        if diffs:
            print(f"  '{name}' -- {len(diffs)} column(s) differ "
                  f"(values identical, storage type changed):"
                  if all("same numbers" in d[1] for d in diffs)
                  else f"  '{name}' -- {len(diffs)} column(s) differ:")
            for c, where, vals in diffs[:6]:
                extra = f"   {vals[0]!r} -> {vals[1]!r}" if vals else ""
                print(f"      {str(c)[:34]:<36} {where}{extra}")
            if len(diffs) > 6:
                print(f"      ... and {len(diffs) - 6} more")
            print()

    if changed == 0:
        print("  NOTHING CHANGED. Most likely you opened the original file rather")
        print("  than the new one, or --apply was not actually on the command.")
        print("  The result file is the one named like:")
        print("      <original name>_split_<date>_<time>.xlsx\n")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        prog="mmm_splitter",
        description="Split a pooled MMM media variable into its campaigns.")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("split", help="run the split")
    p.add_argument("output_file")
    p.add_argument("import_file")
    p.add_argument("--apply", action="store_true", help="write the new file")
    p.add_argument("--dest", default=None, help="name for the new file")

    p = sub.add_parser("headers", help="show exact column headers")
    p.add_argument("output_file")
    p.add_argument("sheet", nargs="?", default=None)

    for name, helptext in [
        ("check-decay", "is Decay retention or the share that decays away?"),
        ("check-effectiveness", "how is Effectiveness calculated?"),
        ("check-coefficients", "how are the coefficient columns built?"),
    ]:
        q = sub.add_parser(name, help=helptext)
        q.add_argument("output_file")

    p = sub.add_parser("check-transform",
                       help="why did a variable fail the transform check?")
    p.add_argument("output_file")
    p.add_argument("variable", nargs="?", default=None)
    p.add_argument("import_file", nargs="?", default=None,
                   help="optional: shows how much each campaign's share "
                        "would actually change")

    p = sub.add_parser("check-curves",
                       help="is a campaign small, or is it inefficient?")
    p.add_argument("result_file")
    p.add_argument("pooled", nargs="?", default=None)

    p = sub.add_parser("check-smoothing",
                       help="was the pooled variable smoothed before modelling?")
    p.add_argument("output_file")
    p.add_argument("import_file")
    p.add_argument("pooled")

    p = sub.add_parser("check-sum",
                       help="what is missing from a pooled variable's parts?")
    p.add_argument("output_file")
    p.add_argument("import_file")
    p.add_argument("pooled")

    p = sub.add_parser("compare", help="what changed between two files")
    p.add_argument("original")
    p.add_argument("result")

    args = ap.parse_args()
    try:
        return dispatch(args)
    except (FileNotFoundError, ValueError, PermissionError) as e:
        # These are the expected, explainable failures -- a wrong path, a
        # missing sheet, a locked file. Print the message rather than a
        # traceback; anything else still raises so real bugs stay visible.
        print(f"\nSTOPPED. {e}")
        return 1


def dispatch(args) -> int:
    if args.command == "split":
        return cmd_split(args)
    if args.command == "headers":
        return cmd_headers(args.output_file, args.sheet)
    if args.command == "check-decay":
        return cmd_check_decay(args.output_file)
    if args.command == "check-effectiveness":
        return cmd_check_effectiveness(args.output_file)
    if args.command == "check-coefficients":
        return cmd_check_coefficients(args.output_file)
    if args.command == "check-transform":
        return cmd_check_transform(args.output_file, args.variable,
                                   args.import_file)
    if args.command == "check-curves":
        return cmd_check_curves(args.result_file, args.pooled)
    if args.command == "check-smoothing":
        return cmd_check_smoothing(args.output_file, args.import_file,
                                   args.pooled)
    if args.command == "check-sum":
        return cmd_check_sum(args.output_file, args.import_file, args.pooled)
    if args.command == "compare":
        return cmd_compare(args.original, args.result)
    return 1


if __name__ == "__main__":
    sys.exit(main())