"""
The one place a split is defined. Everything else reads from here.

`raw`   the DATA column holding the metric the model was actually built on
        (impressions, clicks, GRPs -- whatever went into the equation)
`spend` the DATA column holding actual spend

Both are required: `raw` drives the contribution allocation, `spend` is written
as observed fact. A split is a list, so several pooled variables can be handled
in one run.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Campaign:
    name: str
    raw: str
    spend: str


@dataclass
class Split:
    pooled: str
    campaigns: list[Campaign]

    @property
    def names(self) -> list[str]:
        return [c.name for c in self.campaigns]

    def as_dict(self) -> dict[str, dict[str, str]]:
        """The shape the sheet-level functions expect."""
        return {c.name: {"raw": c.raw, "spend": c.spend} for c in self.campaigns}

    @property
    def data_columns_used(self) -> list[str]:
        return [v for c in self.campaigns for v in (c.raw, c.spend)]

    def validate(self, data_columns: list[str]) -> None:
        """Every named DATA column must exist, and names must not collide."""
        missing = [
            f"{c.name}.{k}={v}"
            for c in self.campaigns
            for k, v in (("raw", c.raw), ("spend", c.spend))
            if v not in data_columns
        ]
        if missing:
            raise ValueError(
                "DATA columns named in the config do not exist:\n  "
                + "\n  ".join(missing)
            )
        if len(set(self.names)) != len(self.names):
            raise ValueError(f"Duplicate campaign names in split '{self.pooled}'.")
        if self.pooled in self.names:
            raise ValueError(
                f"'{self.pooled}' is both the pooled variable and a replacement."
            )


# ===========================================================================
#  SECTION 1 -- SHEET NAMES
#  Change these only if a project uses different sheet names.
# ===========================================================================

STRUCTURE_SHEET = "T structure"               # output file

# Some model exports call it 'Structure' instead. The first name in this list
# that exists in the file AND has a 'Variable' column is used. They are not
# assumed to be the same sheet -- a 'Structure' sheet with a different layout
# is skipped rather than misread.
STRUCTURE_SHEET_ALTERNATIVES = ["Structure", "T Structure"]
DATASHEET       = "T datasheet variables"     # output file
SPENDS_SHEET    = "(SPENDS DEF)"              # output file
CONTRIB_SHEET   = "Individual contributions"  # output file
CURVES_SHEET    = "Response Curves"           # output file
DATA_SHEET      = "DATA"                      # import file

# The date column on the import file's DATA sheet. Leave as None to let the
# tool find it (it looks for a column of dates covering the modelling period).
# Set it explicitly if auto-detection picks the wrong one.
DATA_DATE_COL   = None

# How the modelling software states the Decay column.
#   True  -> the value is RETENTION: the share carrying over to the next day
#            (0.8 = 80% still there tomorrow)
#   False -> the value is DECAY: the share disappearing each day
#            (0.8 = only 20% carries over, so retention is 0.2)
# Getting this wrong makes every carryover wrong. Run
# check_decay_convention.py against a finished model to determine it.
DECAY_IS_RETENTION = False

# Variables whose transform check may be accepted despite failing.
#
# The check asks whether the decay in 'T structure' reproduces the model's own
# contribution. A failure usually means something real. But some variables
# score around 0.98 for reasons that have nothing to do with decay, and for
# those the choice of decay barely moves the split at all.
#
# Listing a variable here says: "we looked, and the disagreement does not
# change the answer." Give the reason -- it is written into the run output.
#
# This is NOT a way to silence the check. The tool still refuses to proceed if
# the disagreement would actually shift the allocation (see the "does it change
# the answer?" figure from  check-transform).
#
#     ACCEPT_TRANSFORM_MISMATCH = {
#         "M-Channel_Imp_Total":
#             "scores 0.98 for unrelated reasons; share impact 0.017pp",
#     }
ACCEPT_TRANSFORM_MISMATCH: dict[str, str] = {}


# Pooled variables whose Raw series is known to be wrong in the output file.
#
# Normally the campaign columns must sum to the pooled variable's Raw, and a
# mismatch means the config names the wrong columns. Occasionally the export
# itself is wrong -- a component included that should not be there.
#
# Listing a variable here says: "the campaign columns are right, the pooled
# Raw is not." The split proceeds on the campaign columns, and the written Raw
# values will differ from the original by the size of the error -- which is
# the point: the new file is corrected.
#
# The contribution figures are NOT corrected. They came from the model as
# fitted, so whatever effect the erroneous component drove stays in the total
# and is shared out across the campaigns. Only a refit can remove it.
#
#     ACCEPT_RAW_MISMATCH = {
#         "M-Channel_Imp_Total":
#             "Raw wrongly includes MMAI clicks (0.38%); confirmed with the "
#             "modeller that MMAI does not belong in this variable",
#     }
ACCEPT_RAW_MISMATCH: dict[str, str] = {}


# How the response curves for the new campaigns are built.
#
#   "shared_shape"  every campaign gets the pooled curve itself, and they
#                   differ only by cost per unit: a campaign with cheaper
#                   media reaches a given response for less spend, so it sits
#                   HIGHER on a chart of KPI against spend. This is what a
#                   reader expects, and the ranking follows cost efficiency.
#
#                   The catch: the five curves no longer sum to the pooled
#                   curve. Each implies it could reach the full pooled maximum
#                   alone, so an optimiser allowed to spend without limit
#                   across all of them would see far more headroom than the
#                   model supports. Safe when the optimiser caps each channel
#                   near historic spend, which is the normal setup.
#
#   "additive"      each campaign's curve is the pooled curve scaled down by
#                   its share of volume, so the five sum to the pooled curve
#                   exactly and no optimiser can over-allocate.
#
#                   The catch: on a chart every campaign saturates almost
#                   immediately and then sits at a plateau equal to its share
#                   of volume. A small campaign looks permanently worse than a
#                   large one at every spend level, which is not what the
#                   model says.
#
#   "own_curve"     each campaign's curve is built from the model itself:
#                   scale THAT campaign in the curve window, hold the others
#                   at what they actually ran, and read the KPI the model
#                   predicts (same formula as the platform's own curves). It
#                   starts at zero and is the increment the campaign adds.
#
#                   This is the only mode whose marginal ROI at current spend
#                   is the model's own answer: the model has one curve over
#                   the campaigns' COMBINED carryover, so a campaign's next
#                   unit lands where the whole variable already operates.
#                   shared_shape instead puts a small campaign low on the
#                   pooled curve, where it is still steep, and overstates its
#                   marginal ROI (4.4x on one synthetic test).
#
#                   The catch: the curves do not sum to the pooled curve. The
#                   gap is saturation the campaigns share, and the run reports
#                   its size. It needs '(COSTS DEF)' in the output file, and it
#                   first rebuilds the pooled curve from the model; if that is
#                   not exact to 1e-6 it stops rather than guess.
#
# The model only ever estimated one curve. These are three ways of
# presenting that single fact; own_curve is the one that answers "what would
# more spend on this campaign return, the others unchanged?"
#
#   "native_curve"  (the default) a platform-type curve per campaign -- the
#                   variable's own curve type, with its own height and one
#                   parameter -- fitted to the campaign's share of the response
#                   when the WHOLE GROUP moves together (day by day, with
#                   carry-over, by the same rule that splits the contribution).
#                   Starts at zero, adds up to the group when the campaigns
#                   move in proportion, and is exact on that kind of move. For
#                   moving money between campaigns of the same group it is
#                   conservative: slopes are understated above current spend
#                   and overstated below it. The run prints, per campaign, the
#                   fit, the slope difference to own_curve within +/-20% of
#                   current pressure, and how well the curves add up.
CURVE_MODE = "native_curve"


# When the model output has no 'Response Curves' / 'T ROI curves' sheets,
# build them from the model itself, with the platform's own formula (checked
# against a real export to 1e-11). Needs '(COSTS DEF)' and '(SPENDS DEF)' in
# the output file. With nothing to split, a run then only adds those sheets.
BUILD_MISSING_CURVES = True

# Steps in the grid of a BUILT curve: 350 gives 351 points from zero to twice
# the window maximum, the platform's default. (It can also write e.g. 20.)
# Only affects curve sheets the tool builds; existing sheets keep their own
# grid when split.
CURVE_STEPS = 350


# Campaign columns that are not on DATA themselves but the plain sum of DATA
# columns -- a composite inside a combined variable, kept whole as one
# campaign. The app fills this in; from the command line:
#
#     COMPOSITE_COLUMNS = {
#         "M-Channel_Inv_Campaign2": ["M-Channel_Inv_Part1", "M-Channel_Inv_Part2"],
#     }
COMPOSITE_COLUMNS: dict[str, list[str]] = {}

# CURVE_MODE can also be set per variable, which matters when a model mixes
# volume-based and spend-based pooled variables:
#
#     CURVE_MODE = {
#         "default": "shared_shape",
#         "M-Channel_Inv": "additive",
#     }
#
# Use "additive" for any variable the model was built on SPEND. Cost per unit
# is spend divided by the modelled metric, so when the metric IS spend that is
# 1.00 for every campaign -- shared_shape then produces five identical
# overlapping curves that also do not add up. The tool warns when it detects
# this.


# Smoothing applied to a pooled variable before it went into the model.
#
# Sometimes a variable is smoothed -- a rolling average over a few days -- to
# stop one campaign's spike dominating. The campaign columns in the import file
# are the raw figures, so they will not sum to the smoothed pooled series.
#
# A rolling average is linear, so applying the same smoothing to each campaign
# and adding them reproduces the smoothed pooled series exactly. Listing it
# here does that, and the sums then reconcile with no override needed.
#
# It also matters for the split itself: the model saw the smoothed series, so
# the effect of a spike was attributed spread across days. Splitting on the
# unsmoothed campaign data would hand that day's effect to the spiky campaign,
# which is the opposite of what the smoothing was for.
#
# Find the window with:
#     python mmm_splitter.py check-smoothing OUTPUT.xlsx IMPORT.xlsx "VARIABLE"
#
#     SMOOTHING = {
#         "M-Channel_Inv": {"window": 5, "centred": False},
#     }
SMOOTHING: dict[str, dict] = {}

# Pooled variables where a few days were adjusted by hand before modelling.
#
# A rolling window (above) can be reproduced exactly. A hand-edited cell -- a
# spike replaced with a typed average, say -- cannot be decomposed into
# campaigns by any rule, because there is no rule.
#
# Listing a variable here says: on the days where the campaign columns and the
# pooled figure disagree, the POOLED figure is what the model used, so take it
# as authoritative and share it across the campaigns in proportion to what they
# actually ran that day.
#
# Deliberately narrow. The tool refuses if the disagreement covers more than a
# handful of days or more than a couple of percent of volume -- beyond that it
# is not a hand adjustment, it is a missing component, and check-sum should be
# used instead.
#
#     RESCALE_TO_POOLED = {
#         "M-Channel_Inv": "5 days replaced with a typed average "
#                        "to stop one campaign's spike dominating",
#     }
RESCALE_TO_POOLED: dict[str, str] = {}


# Sheets our downstream tools do not read. The pooled variable may well appear
# on them -- they are part of the modelling software's own working -- but
# nothing we do afterwards uses them, so they are carried through untouched.
#
# If one of these ever starts being read downstream, take it off the list and
# the tool will stop until it is handled properly.
UNTOUCHED = {
    "T model vs. actual",
    "Statistics",
    "Model fit",
    # the modelling software's own working sheets
    "Summary",
    "Structure",
    "Composite data",
    "Transformed Data",
    "Individual total contributions",
    "Category contributions",
    "Category total contributions",
    "Raw data",
    "Model vs. Actual",
    "Comments",
    # hidden duplicates of the definition sheets
    "(HIDDEN SPENDS DEF)",
    "(HIDDEN COSTS DEF)",
    "(COSTS DEF)",
}


# ===========================================================================
#  SECTION 2 -- THE SPLIT
#  This is the part that changes for every job.
#
#  pooled   exact name of the variable being replaced, as written in the
#           output file
#  Campaign(name, raw, spend)
#           name   what the new variable will be called everywhere
#           raw    column in the import file's DATA sheet holding the metric
#                  the model was built on (impressions / clicks / GRPs)
#           spend  column in the import file's DATA sheet holding spend
#
#  Add as many Campaign lines as you need -- two, five, twelve.
#
#  To split SEVERAL pooled variables, add more Split(...) entries to this ONE
#  list. Do NOT write "SPLITS = [...]" a second time: Python keeps only the
#  last assignment and the earlier ones are silently ignored.
#
#      SPLITS = [
#          Split(pooled="first variable",  campaigns=[...]),
#          Split(pooled="second variable", campaigns=[...]),
#      ]
# ===========================================================================

SPLITS = [
    Split(
        pooled="<name of the combined variable, exactly as in the output file>",
        campaigns=[
            Campaign("<new variable name>",
                     "<DATA column with the modelled metric>",
                     "<DATA column with spend>"),
            Campaign("<new variable name>",
                     "<DATA column with the modelled metric>",
                     "<DATA column with spend>"),
        ],
    ),
]

# Leave the block above as it is if you are using the app -- it fills this in
# from your files and ignores whatever is written here. Only edit it if you
# intend to run the tool from the command line.