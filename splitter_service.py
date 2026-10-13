"""
Everything the hosted app needs that is not UI.

Three jobs:

1.  ISOLATION. mmm_splitter reads its settings from module globals and prints
    its findings. On a hosted server several people share one Python process,
    so two runs at once would overwrite each other's SPLITS and mix their
    output. Every call into the tool therefore goes through `run_isolated`,
    which holds one process-wide lock, resets every global to config.py's
    values, applies this run's settings, captures stdout, and resets again.

2.  WORKSPACE. The tool works on paths. Uploaded files are written to a fresh
    temporary folder for the duration of one call and the folder is deleted
    afterwards, whatever happens. Nothing is kept on the server's disk.

3.  PRE-FILL. Reads the model output's 'Composite data' sheet and turns
    plain-sum composites into campaign rows. For combined variables made
    outside the platform there is no formula, so it looks for a set of
    same-channel, same-metric DATA columns that sums exactly to the pooled
    Raw series. Both are suggestions; the tool's own checks still decide.
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import re
import shutil
import tempfile
import threading
import traceback
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config as CFG
import mmm_splitter as M

# Findings are worded for the app, not the command line.
M.INTERFACE = "app"

# Imported modules persist for the life of the server process (only the main
# script is re-executed on each interaction), so this is one lock per process.
_RUN_LOCK = threading.Lock()

# The globals mmm_splitter reads, and their values as config.py states them.
# Taken from config, NOT from mmm_splitter: the tool rewrites some of its own
# globals during a run (STRUCTURE_SHEET switches to an alternative name when a
# file lacks 'T structure'), and that must not leak into the next run.
_GLOBALS = ("SPLITS", "STRUCTURE_SHEET", "DECAY_IS_RETENTION", "CURVE_MODE",
            "ACCEPT_TRANSFORM_MISMATCH", "ACCEPT_RAW_MISMATCH", "SMOOTHING",
            "RESCALE_TO_POOLED", "COMPOSITE_COLUMNS", "SEGMENT_WINDOWS")
CONFIG_DEFAULTS = {k: copy.deepcopy(getattr(CFG, k, {})) for k in _GLOBALS}

# Per-job settings the app now asks for itself. If config.py still carries
# values for these they are ignored by the app, and the UI says so.
PER_JOB = ("ACCEPT_TRANSFORM_MISMATCH", "ACCEPT_RAW_MISMATCH", "SMOOTHING",
           "RESCALE_TO_POOLED", "COMPOSITE_COLUMNS", "SEGMENT_WINDOWS")

SPEND_TOKENS = ("Inv", "Spend", "Cost")
COMPOSITE_SHEET = "Composite data"


def _reset(settings: dict | None = None) -> None:
    for k, v in CONFIG_DEFAULTS.items():
        setattr(M, k, copy.deepcopy(v))
    for k, v in (settings or {}).items():
        setattr(M, k, v)


def _safe_name(name: str) -> str:
    return re.sub(r"[^\w.\- ]", "_", Path(name).name) or "file.xlsx"


@dataclass
class RunResult:
    code: int = 1
    output: str = ""
    crash: str | None = None
    value: object = None
    written: bytes | None = None


def run_isolated(files: dict[str, tuple[str, bytes]], settings: dict,
                 fn, keep: str | None = None) -> RunResult:
    """
    Write `files` ({role: (name, bytes)}) to a temporary folder, call
    fn(paths, folder) with the tool configured by `settings`, and clean up.

    `keep` names a file fn will create in the folder; its bytes are returned
    before the folder is removed.
    """
    res = RunResult()
    with _RUN_LOCK:
        folder = Path(tempfile.mkdtemp(prefix="vsplit_"))
        buf = io.StringIO()
        try:
            paths = {}
            for role, (name, data) in files.items():
                sub = folder / role
                sub.mkdir()
                p = sub / _safe_name(name)
                p.write_bytes(data)
                paths[role] = str(p)
            _reset(settings)
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                res.value = fn(paths, folder)
            res.code = res.value if isinstance(res.value, int) else 0
            if keep and (folder / keep).exists():
                res.written = (folder / keep).read_bytes()
        except SystemExit as e:                    # config import guard etc.
            res.code = int(e.code or 1)
        except Exception as e:                     # noqa: BLE001
            res.crash = (f"{type(e).__name__}: {e}\n\n"
                         + "".join(traceback.format_exc()))
        finally:
            _reset()
            shutil.rmtree(folder, ignore_errors=True)
        res.output = buf.getvalue()
    return res


class _Args:
    def __init__(self, output_file, import_file, apply=False, dest=None):
        self.output_file, self.import_file = output_file, import_file
        self.apply, self.dest = apply, dest


def split(files, settings, apply: bool, dest_name: str | None = None) -> RunResult:
    """The real thing: mmm_splitter.cmd_split, check-only or writing."""
    dest_name = _safe_name(dest_name) if dest_name else None

    def go(paths, folder):
        dest = str(folder / dest_name) if (apply and dest_name) else None
        return M.cmd_split(_Args(paths["output"], paths["import"], apply, dest))

    return run_isolated(files, settings, go, keep=dest_name if apply else None)


def diagnose(files, settings, which: str, pooled: str) -> RunResult:
    """The two diagnostics the app offers, for one variable: which DATA
    columns would close a gap in the sum, and whether a rolling average
    explains a day-level mismatch."""
    fn = {"sum": M.cmd_check_sum, "smoothing": M.cmd_check_smoothing}[which]
    return run_isolated(files, settings,
                        lambda p, _f: fn(p["output"], p["import"], pooled))


# ---------------------------------------------------------------------------
#  Naming helpers (same convention as the original app)
# ---------------------------------------------------------------------------

def split_name(variable: str) -> tuple[str, str]:
    """M-<channel>_<metric>_<descriptor> -> (channel, metric). Channels may
    contain spaces, so split on underscores only."""
    parts = str(variable).split("_")
    return (parts[0] if parts else ""), (parts[1] if len(parts) > 1 else "")


def _descriptor(variable: str) -> str:
    parts = str(variable).split("_", 2)
    return parts[2] if len(parts) > 2 else ""


def likely_columns(cols: list[str], pooled: str) -> list[str]:
    """Same channel and same metric as the pooled variable, pooled excluded."""
    ch, mt = split_name(pooled)
    return [c for c in cols if c != pooled and split_name(c) == (ch, mt)]


def guess_spend(raw: str, cols: list[str]) -> str:
    """The spend column for a metric column: same channel, a spend token in
    the metric slot, same descriptor. Only returned when exactly one fits."""
    ch, mt = split_name(raw)
    if mt in SPEND_TOKENS:
        return raw                       # modelled on spend
    desc = _descriptor(raw)
    hits = [c for c in cols if split_name(c)[0] == ch
            and split_name(c)[1] in SPEND_TOKENS and _descriptor(c) == desc]
    return hits[0] if len(hits) == 1 else ""


# ---------------------------------------------------------------------------
#  Composite data
# ---------------------------------------------------------------------------

def _blank(v) -> bool:
    if isinstance(v, str):
        return not v.strip()
    try:
        return bool(pd.isna(v))               # None, NaN and NaT alike
    except (TypeError, ValueError):
        return False


def read_composites(path: str) -> dict[str, dict]:
    """{name: {"formula": str, "symbols": {symbol: variable}}}.

    Layout as in the model export: names on row 0 from column 4, the formula
    on row 1, then 'symbol: variable' lines until the data block starts.
    """
    try:
        sheet = pd.read_excel(path, sheet_name=COMPOSITE_SHEET, header=None)
    except ValueError:
        return {}
    if sheet.empty or sheet.shape[1] <= 4:
        return {}
    first_data = sheet.shape[0]
    for i in range(sheet.shape[0]):
        cell = sheet.iloc[i, 3]
        if not _blank(cell) and not isinstance(cell, str):
            first_data = i
            break
    out = {}
    for col in range(4, sheet.shape[1]):
        name, formula = sheet.iloc[0, col], sheet.iloc[1, col]
        if _blank(name) or _blank(formula):
            continue
        symbols = {}
        for row in range(2, first_data):
            cell = sheet.iloc[row, col]
            if _blank(cell):
                continue
            sym, sep, var = str(cell).partition(":")
            if sep:
                symbols[sym.strip()] = var.strip()
        out[str(name).strip()] = {"formula": str(formula).strip(),
                                  "symbols": symbols}
    return out


def _strip_parens(e: str) -> str:
    while e.startswith("(") and e.endswith(")"):
        depth = 0
        for i, ch in enumerate(e):
            depth += ch == "("
            depth -= ch == ")"
            if depth == 0 and i < len(e) - 1:
                return e                 # '(a)+(b)' -- outer pair not matched
        e = e[1:-1]
    return e


def _plain_sum_terms(expr: str, symbols: dict) -> list[str] | None:
    e = _strip_parens(expr.replace(" ", ""))
    depth, terms, cur = 0, [], ""
    for ch in e:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "+" and depth == 0:
            terms.append(cur)
            cur = ""
        else:
            cur += ch
    terms.append(cur)
    terms = [_strip_parens(t) for t in terms]
    if not terms or any(t not in symbols for t in terms):
        return None
    return [symbols[t] for t in terms]


_SMOOTH = re.compile(r"^(bma|cma)\((.*),(\d+)\)$", re.IGNORECASE)


@dataclass
class Prefill:
    pooled: str
    source: str                       # "composite" | "sum-match" | "none"
    rows: list[dict] = field(default_factory=list)
    smoothing: dict | None = None
    notes: list[str] = field(default_factory=list)
    formula: str | None = None
    # columns that are not on DATA but are built as sums of DATA columns:
    # a nested composite kept whole as one campaign (and its spend)
    virtual: dict[str, list[str]] = field(default_factory=dict)
    # a platform split segment: the period its campaign columns are cut to
    # (None at an end that runs to the edge of the data)
    window: tuple | None = None
    suffix: str = ""


# ---------------------------------------------------------------------------
#  Split segments and the parts search
# ---------------------------------------------------------------------------

_SPLIT = re.compile(r"^(?P<base>.+?)(?P<suffix>\s*\(Split\d+\)|_Split\d+)\s*$")


def split_base(name: str) -> tuple[str | None, str]:
    """'M-X (Split1)' -> ('M-X', ' (Split1)'); 'M-X_Split2' -> ('M-X', '_Split2');
    (None, '') for a name that is not a platform split."""
    m = _SPLIT.match(str(name))
    return (m.group("base"), m.group("suffix")) if m else (None, "")


def segment_window(series: pd.Series | None) -> tuple | None:
    """
    The period a split segment covers, from its own Raw series: first to last
    non-zero day. Where the cut really falls inside an off stretch does not
    matter -- the variable is zero there either way. An end that touches the
    edge of the data is left open (None), so carry-over from before the model
    start still counts for a first segment.
    """
    if series is None or series.empty:
        return None
    v = pd.to_numeric(series, errors="coerce").fillna(0.0).sort_index()
    nz = v[v != 0]
    if nz.empty:
        return None
    lo, hi = nz.index.min(), nz.index.max()
    return (None if lo <= v.index.min() else lo,
            None if hi >= v.index.max() else hi)


def active_outside(col: str, data: pd.DataFrame | None, window) -> bool:
    """Whether a DATA column has any non-zero value outside a segment's
    period. Unknown (no data, no column, no window) counts as yes."""
    if data is None or not col or col not in data.columns or not window:
        return True
    lo, hi = window
    idx = data.index
    out = np.zeros(len(idx), dtype=bool)
    if lo is not None:
        out |= np.asarray(idx < pd.Timestamp(lo))
    if hi is not None:
        out |= np.asarray(idx > pd.Timestamp(hi))
    x = pd.to_numeric(data[col], errors="coerce").fillna(0.0).to_numpy()
    return bool((x[out] != 0).any())


def segment_name(raw: str, spend: str, data, window, suffix: str) -> str:
    """
    A split segment's campaign keeps the segment's suffix only where it is
    needed to tell the two apart: if the column also ran outside the
    segment's period, cutting it gives a different series, and it is named
    '<column> (Split1)'. If it ran only inside, the cut column IS the column,
    and it keeps its plain name.
    """
    if not suffix or not raw:
        return raw
    if active_outside(raw, data, window) or \
            (spend and spend != raw and active_outside(spend, data, window)):
        return f"{raw}{suffix}"
    return raw


def search_parts(target: pd.Series, data: pd.DataFrame, data_cols: list[str],
                 metric: str, window: tuple | None = None,
                 exclude: set | None = None) -> tuple[list[str] | None, str]:
    """
    The DATA columns that add up exactly to `target`, each used once.

    Not a search through combinations: one non-negative least-squares solve
    (milliseconds, also for dozens of candidates), then checks.
      candidates  media columns of the same metric type, active inside the
                  window, never above the target (parts of a plain sum cannot be)
      accepted    only if the solve reproduces the target exactly, every weight
                  is 0 or 1, at least two columns are used, and NO other exact
                  solution exists -- tested by re-solving without each chosen
                  column in turn. Anything else returns None and a reason; the
                  user fills the table in.
    """
    from scipy.optimize import nnls
    y = pd.to_numeric(target, errors="coerce").fillna(0.0).sort_index()
    if window:
        lo, hi = window
        keep = pd.Series(True, index=y.index)
        if lo is not None:
            keep &= y.index >= pd.Timestamp(lo)
        if hi is not None:
            keep &= y.index <= pd.Timestamp(hi)
        y = y[keep]
    if y.empty or not (y != 0).any() or data is None:
        return None, "no activity to match"
    X = data.reindex(y.index)
    tol = 1e-6 * float(np.abs(y).max()) + 1e-9
    cands, alone = [], []
    for c in data_cols:
        if c in (exclude or set()) or not str(c).startswith("M-") \
                or split_name(c)[1] != metric or c not in X.columns:
            continue
        x = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        if not (x > tol).any() or (x > y + tol).any() or (x < -tol).any():
            continue
        if float(np.abs(x - y).max()) <= tol:
            alone.append(c)                       # the variable itself, renamed
            continue
        cands.append(c)
    if len(cands) < 2:
        return None, (f"matches {alone[0]!r} on its own, which is the variable "
                      f"itself rather than its parts" if alone else
                      f"only {len(cands)} '{metric}' column(s) fit inside it")
    A = X[cands].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy()
    b = y.to_numpy()

    def solve(cols_idx):
        w, _ = nnls(A[:, cols_idx], b)
        exact = float(np.abs(A[:, cols_idx] @ w - b).max()) <= tol
        return w, exact

    w, exact = solve(list(range(len(cands))))
    if not exact:
        return None, (f"no set of the {len(cands)} '{metric}' columns that fit "
                      f"inside it adds up to it exactly")
    if not all(abs(v) < 1e-6 or abs(v - 1) < 1e-6 for v in w):
        return None, "only a weighted combination fits, not a plain sum"
    chosen = [i for i, v in enumerate(w) if v > 0.5]
    if len(chosen) < 2:
        return None, "only one column fits"
    for j in chosen:
        rest = [i for i in range(len(cands)) if i != j]
        if rest and solve(rest)[1]:
            return None, (f"more than one set of columns adds up to it exactly "
                          f"(for example without {cands[j]!r}); choose the parts "
                          f"by hand")
    return [cands[i] for i in chosen], (f"the only exact sum among {len(cands)} "
                                        f"'{metric}' columns")


def _spend_name(part: str, taken) -> str:
    ch, mt = split_name(part)
    if mt in SPEND_TOKENS:
        return part
    alt = part.replace(f"_{mt}_", "_Inv_", 1) if mt else ""
    return alt if alt and alt != part and alt not in taken else f"{part} spend"


def from_composite(pooled: str, composites: dict, data_cols: list[str],
                   _depth: int = 0, model_raw_cols=(),
                   expand_nested: bool = False) -> Prefill:
    """
    Campaign rows from a composite's formula. A part that is itself a
    composite is kept WHOLE by default -- one campaign, built as the sum of
    its DATA columns -- because that is usually the level the composite was
    made at. expand_nested=True takes it apart down to DATA columns.
    """
    spec = composites[pooled]
    pf = Prefill(pooled, "composite", formula=spec["formula"])
    expr = spec["formula"].replace(" ", "")

    m = _SMOOTH.match(expr)
    if m:
        kind, inner, n = m.group(1).lower(), m.group(2), int(m.group(3))
        if kind == "cma" and n % 2 == 0:
            pf.notes.append(
                f"Formula is an even-width centred average (cma, {n}). The "
                f"platform weights its two end points by half; the tool's "
                f"rolling mean does not, so smoothing was NOT pre-filled. "
                f"Use 'Was it smoothed before modelling?' under Diagnose this "
                f"variable before splitting this one.")
        else:
            pf.smoothing = {"window": n, "centred": kind == "cma"}
        expr = inner

    parts = _plain_sum_terms(expr, spec["symbols"])
    if parts is None:
        pf.source = "none"
        pf.notes.append(
            f"Formula `{spec['formula']}` is not a plain sum of its parts "
            f"(weights, subtraction, shifts or functions). The tool splits "
            f"plain sums only, so nothing was pre-filled.")
        return pf

    for part in parts:
        if part in data_cols:
            pf.rows.append({"name": part, "raw": part,
                            "spend": guess_spend(part, data_cols)})
        elif part in composites and _depth < 5:
            sub = from_composite(part, composites, data_cols, _depth + 1,
                                 model_raw_cols, expand_nested=True)
            resolved = (sub.source == "composite" and not sub.smoothing
                        and sub.rows and all(r["raw"] for r in sub.rows))
            if resolved and not expand_nested:
                taken = set(data_cols) | set(composites)
                row = {"name": part, "raw": part, "spend": ""}
                pf.virtual[part] = [r["raw"] for r in sub.rows]
                if all(r["spend"] for r in sub.rows):
                    sp = _spend_name(part, taken)
                    row["spend"] = sp
                    if sp != part:
                        pf.virtual[sp] = [r["spend"] for r in sub.rows]
                pf.rows.append(row)
                pf.notes.append(f"'{part}' is itself a composite and is kept "
                                f"whole: one campaign, the sum of its "
                                f"{len(sub.rows)} DATA columns.")
            elif resolved:
                pf.rows += sub.rows
                pf.virtual.update(sub.virtual)
                pf.notes.append(f"'{part}' is itself a composite and was "
                                f"expanded into its {len(sub.rows)} parts.")
                pf.notes += [n for n in sub.notes
                             if not n.startswith("No unambiguous spend")]
            else:
                pf.rows.append({"name": part, "raw": "", "spend": ""})
                pf.notes.append(f"'{part}' is a composite that cannot be "
                                f"expanded into DATA columns; fill it in by hand.")
        elif part.strip() in {c.strip() for c in model_raw_cols}:
            pf.rows.append({"name": part, "raw": "", "spend": ""})
            pf.notes.append(f"'{part}' is in the model output's 'Raw data' but "
                            f"not on the import file's DATA sheet -- this import "
                            f"file is probably not the one the model was built "
                            f"from. Use that one.")
        else:
            pf.rows.append({"name": part, "raw": "", "spend": ""})
            pf.notes.append(f"'{part}' is not a column on the import file's "
                            f"DATA sheet (a platform split such as _Split2 "
                            f"never is); fill it in by hand.")
    missing = [r["name"] for r in pf.rows if r["raw"] and not r["spend"]]
    if missing:
        pf.notes.append(f"No unambiguous spend column for: {', '.join(missing)}.")
    return pf


def from_sum_match(pooled: str, pooled_raw: pd.Series | None,
                   data: pd.DataFrame | None, data_cols: list[str],
                   window: tuple | None = None, suffix: str = "") -> Prefill:
    """Find the same-metric DATA columns that add up exactly to the pooled
    Raw series (see search_parts). For a split segment, only inside its
    window, and the campaigns are named with the segment's suffix."""
    pf = Prefill(pooled, "none", window=window, suffix=suffix)
    if pooled_raw is None or data is None:
        return pf
    metric = split_name(split_base(pooled)[0] or pooled)[1]
    parts, why = search_parts(pooled_raw, data, data_cols, metric, window,
                              exclude={pooled})
    if parts is None:
        pf.notes.append(f"No parts found automatically: {why}.")
        return pf
    pf.source = "sum-match"
    pf.rows = [{"name": c, "raw": c,
                "spend": guess_spend(c, data_cols)} for c in parts]
    pf.notes.append(f"Parts found automatically: {why}.")
    missing = [r["raw"] for r in pf.rows if not r["spend"]]
    if missing:
        pf.notes.append(f"No unambiguous spend column for: {', '.join(missing)}.")
    return pf

# ---------------------------------------------------------------------------
#  Inspection: everything the UI needs from the two files, in one locked call
# ---------------------------------------------------------------------------

@dataclass
class Inspection:
    variables: list[str]
    structure_sheet: str
    data_cols: list[str]
    composites: dict
    date_col: str | None
    pooled_raw: dict[str, pd.Series]          # variable -> Raw, indexed by date
    data: pd.DataFrame | None                 # DATA, indexed by date
    notes: list[str] = field(default_factory=list)
    model_raw_cols: list[str] = field(default_factory=list)
    has_curves: bool = True                   # 'Response Curves' present
    can_build_curves: bool = False            # spends to build them
    has_costs: bool = True                    # '(COSTS DEF)' present


def inspect(files) -> RunResult:
    def go(paths, _folder):
        out, imp = paths["output"], paths["import"]
        inp = M.load_inputs(out, imp)
        notes = []
        try:
            date_col = inp.date_column()
            data = inp.data()
            data.index = pd.to_datetime(data[date_col], errors="coerce")
            data = data[data.index.notna() & ~data.index.duplicated()]
        except Exception as e:                       # noqa: BLE001
            date_col, data = None, None
            notes.append(f"Could not read DATA by date ({e}); sum-matching "
                         f"pre-fill is off.")
        pooled_raw = {}
        try:
            ds = M.read_canon(out, sheet_name=M.DATASHEET)
            ds.columns = [str(c).strip() for c in ds.columns]
            ds["Period name"] = pd.to_datetime(ds["Period name"], errors="coerce")
            for var, g in ds.groupby(ds["Variable"].astype(str).str.strip()):
                pooled_raw[var] = pd.to_numeric(g.set_index("Period name")["Raw"],
                                                errors="coerce")
        except Exception as e:                       # noqa: BLE001
            notes.append(f"Could not read '{M.DATASHEET}' ({e}); sum-matching "
                         f"pre-fill is off.")
        try:
            raw_cols = [str(c).strip() for c in
                        pd.read_excel(out, sheet_name="Raw data", nrows=0).columns]
        except Exception:                            # noqa: BLE001
            raw_cols = []
        return Inspection(
            variables=inp.model_variables, structure_sheet=M.STRUCTURE_SHEET,
            data_cols=list(inp.data_columns), composites=read_composites(out),
            date_col=date_col, pooled_raw=pooled_raw, data=data, notes=notes,
            model_raw_cols=raw_cols,
            has_curves=M.CURVES_SHEET in inp.output_sheets,
            can_build_curves=(M.SPENDS_SHEET in inp.output_sheets
                              and bool(M.BUILD_MISSING_CURVES)),
            has_costs=M.COSTS_SHEET in inp.output_sheets)

    return run_isolated(files, {}, go)


def prefill(pooled: str, ins: Inspection, expand_nested: bool = False) -> Prefill:
    base, suffix = split_base(pooled)
    if base is not None and pooled not in ins.data_cols:
        return _prefill_segment(pooled, base, suffix, ins, expand_nested)
    return _prefill_plain(pooled, ins, expand_nested)


def _prefill_segment(pooled, base, suffix, ins, expand_nested) -> Prefill:
    """
    A platform split segment ('… (Split1)', '…_Split2'). Its campaigns are its
    parts cut to the segment's period; the cut is set whether or not the
    parts are found, so it also applies to rows filled in by hand.
    """
    window = segment_window(ins.pooled_raw.get(pooled))
    if base in ins.composites:
        pf = from_composite(base, ins.composites, ins.data_cols,
                            model_raw_cols=ins.model_raw_cols,
                            expand_nested=expand_nested)
        pf.pooled, pf.window, pf.suffix = pooled, window, suffix
        if pf.source != "none":
            _name_segment_rows(pf, ins.data)
            return pf
    pf = from_sum_match(pooled, ins.pooled_raw.get(pooled), ins.data,
                        ins.data_cols, window=window, suffix=suffix)
    _name_segment_rows(pf, ins.data)
    pf.notes.insert(0, f"'{pooled}' is a split segment of '{base}'. Its campaign "
                       f"columns are cut to the segment's period.")
    if pf.source == "none":
        pf.notes.append("Fill in the parts of the combined variable by hand -- "
                        "the full columns; the tool cuts them to the segment.")
    return pf


def _name_segment_rows(pf: Prefill, data) -> None:
    """Segment suffix only where a campaign would otherwise be
    indistinguishable from its full column (see segment_name)."""
    for r in pf.rows:
        r["name"] = (segment_name(r["raw"], r["spend"], data, pf.window, pf.suffix)
                     if r.get("raw") else f"{r['name']}{pf.suffix}")


def _prefill_plain(pooled: str, ins: Inspection, expand_nested: bool) -> Prefill:
    if pooled in ins.composites:
        pf = from_composite(pooled, ins.composites, ins.data_cols,
                            model_raw_cols=ins.model_raw_cols,
                            expand_nested=expand_nested)
        if pf.source != "none":
            return pf
        alt = from_sum_match(pooled, ins.pooled_raw.get(pooled), ins.data,
                             ins.data_cols)
        if alt.source != "none":
            alt.notes = pf.notes + ["A set of DATA columns that sums exactly "
                                    "to it was found instead."] + alt.notes
            alt.formula = pf.formula
            return alt
        return pf
    pf = from_sum_match(pooled, ins.pooled_raw.get(pooled), ins.data,
                        ins.data_cols)
    if pf.source == "none" and not pf.notes:
        pf.notes.append("Not in 'Composite data'. Combined outside the "
                        "platform -- fill the campaigns in by hand.")
    return pf
