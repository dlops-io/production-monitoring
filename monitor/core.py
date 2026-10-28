"""Loading, windowing and reporting. This file is given to you — no TODOs in here.

Read it once before you start writing probes: `Finding`, `window()` and `labels_known_by()`
are the three things every probe depends on, and the last one is where most of the honesty
of this tutorial lives.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "data" / "production_log.csv"
PROFILE_PATH = ROOT / "reference" / "training_profile.json"

# Columns a drift test should watch. Note what is *not* here: `model_version` changes every
# time you deploy, so alerting on it would be alerting on your own release process.
CONTINUOUS = ["blur_score", "brightness", "saturation", "confidence", "latency_ms"]
CATEGORICAL = ["image_source", "region", "pred_label"]

# Columns worth slicing accuracy by. `model_version` belongs here, not above.
SEGMENTS = ["image_source", "region", "model_version"]

# Which of the three monitoring layers each column speaks to.
LAYER = {
    "blur_score": "Data Quality",
    "brightness": "Data Quality",
    "saturation": "Data Quality",
    "image_source": "Data Quality",
    "region": "Data Quality",
    "confidence": "Model Performance",
    "pred_label": "Model Performance",
    "latency_ms": "System Health",
}

# The "golden window": the first 30 days of production, which we treat as the reference
# distribution for drift. A drift test always needs a stored reference — either the
# training set, or a window of production traffic from when the system was known healthy.
GOLDEN_START = datetime(2026, 8, 5)
GOLDEN_END = datetime(2026, 9, 4)     # exclusive


@dataclass
class Finding:
    """One thing a probe noticed.

    `layer` says which part of the system the finding is about: the inputs ("Data
    Quality"), the model's behaviour ("Model Performance"), or the service around it
    ("System Health"). The lecture's boundary table calls these Input, Model and Serving.
    """

    probe: str
    layer: str            # "Data Quality" | "Model Performance" | "System Health"
    severity: str         # "info" | "warn" | "critical"
    subject: str          # the feature, segment or artifact this is about
    detail: str           # one line a human can act on
    value: float | None = None

    def __str__(self) -> str:
        mark = {"info": "  ", "warn": "! ", "critical": "!!"}[self.severity]
        return f"{mark} [{self.layer:<17}] {self.subject:<28} {self.detail}"


def ks_statistic(reference: pd.Series, current: pd.Series) -> tuple[float, float]:
    """Two-sample Kolmogorov-Smirnov test: returns (D, p_value).

    D = sup_x |F1(x) - F2(x)| — the largest vertical gap between the two empirical CDFs.
    D is 0 when the two samples come from the same distribution and 1 when they do not
    overlap at all. scipy computes it; you decide what to do about it.
    """
    d, p = stats.ks_2samp(reference.dropna(), current.dropna())
    return float(d), float(p)


def ks_critical(n: int, m: int, alpha: float = 0.01) -> float:
    """D_critical ≈ c(α)·√((n+m)/(n·m)): the largest D you'd expect from two samples of
    the *same* distribution, at significance α. If D exceeds it, the samples differ.
    Notice it shrinks as n and m grow — with enough data, any difference is "significant".
    """
    c = {0.10: 1.22, 0.05: 1.36, 0.01: 1.63}[alpha]
    return float(c * np.sqrt((n + m) / (n * m)))


def psi(reference: pd.Series, current: pd.Series, eps: float = 1e-4) -> float:
    """Population Stability Index — the categorical counterpart to a KS test.

    KS only works on continuous columns. For `region` or `image_source` you compare the
    category mix instead: Σ (b−a)·ln(b/a) over categories. Rules of thumb: <0.1 stable,
    0.1–0.25 moderate shift, >0.25 major shift.
    """
    cats = sorted(set(reference.dropna()) | set(current.dropna()))
    a = reference.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    b = current.value_counts(normalize=True).reindex(cats).fillna(0) + eps
    return float(((b - a) * np.log(b / a)).sum())


def load_log(path: Path | str = LOG_PATH) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["ts", "label_ts"])
    return df.sort_values("ts").reset_index(drop=True)


def load_profile(path: Path | str = PROFILE_PATH) -> dict:
    return json.loads(Path(path).read_text())


def window(df: pd.DataFrame, as_of: datetime, days: int) -> pd.DataFrame:
    """The last `days` of prediction events, as of `as_of`."""
    start = as_of - timedelta(days=days)
    return df[(df["ts"] > start) & (df["ts"] <= as_of)]


def golden_window(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["ts"] >= GOLDEN_START) & (df["ts"] < GOLDEN_END)]


def labels_known_by(df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
    """Events whose confirmed label had actually *arrived* by `as_of`.

    This is the whole difficulty of production evaluation in one function. A confirmed
    label sitting in the CSV with `label_ts` next Tuesday is not something you know today,
    and filtering on it would let you 'detect' failures using information you don't have.
    Every accuracy number in this repo goes through here.
    """
    return df[df["label_ts"].notna() & (df["label_ts"] <= as_of)]


def accuracy(frame: pd.DataFrame) -> float | None:
    if frame.empty:
        return None
    return float((frame["pred_label"] == frame["confirmed_label"]).mean())


def segment_table(
    frame: pd.DataFrame, by: str, min_n: int = 40
) -> pd.DataFrame:
    """Accuracy and volume per value of `by` — the 'run it per segment' drill."""
    rows = []
    for value, grp in frame.groupby(by, dropna=False):
        if len(grp) < min_n:
            continue
        rows.append({by: value, "n": len(grp), "accuracy": accuracy(grp)})
    out = pd.DataFrame(rows)
    return out.sort_values("accuracy") if not out.empty else out


def print_findings(findings: list[Finding], header: str) -> None:
    print(f"\n{header}")
    print("-" * len(header))
    if not findings:
        print("  (all probes green)")
        return
    order = {"critical": 0, "warn": 1, "info": 2}
    for f in sorted(findings, key=lambda f: (order[f.severity], f.probe)):
        print(f)
