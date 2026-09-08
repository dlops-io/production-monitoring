"""Reference solution for monitor/probes.py.

⚠️  Try each probe yourself first. The point of the tutorial is the moment your own probe
    fires on a day you didn't expect — that doesn't happen if you copy this in.

To run the tutorial against this file instead of your own:
    python -m monitor.cli check --solution
"""

from __future__ import annotations

import pandas as pd

from monitor.core import (
    CATEGORICAL,
    CONTINUOUS,
    LAYER,
    SEGMENTS,
    Finding,
    accuracy,
    ks_critical,
    ks_statistic,
    psi,
    segment_table,
)

# ---------------------------------------------------------------------------
# Probe 1 — Data Quality: are the inputs broken?
# ---------------------------------------------------------------------------


def check_data_quality(win: pd.DataFrame, profile: dict) -> list[Finding]:
    out: list[Finding] = []

    for feat, spec in profile["features"].items():
        rate = float(win[feat].isna().mean())
        baseline = spec.get("null_rate", 0.0)
        threshold = max(3 * baseline, 0.02)
        if rate > threshold:
            out.append(
                Finding(
                    "data_quality",
                    "Data Quality",
                    "critical" if rate > 0.05 else "warn",
                    feat,
                    f"null rate {rate:.1%} (training baseline {baseline:.1%})",
                    rate,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Probe 2 — Feature & output drift: has the distribution moved?
# ---------------------------------------------------------------------------


def check_feature_drift(
    win: pd.DataFrame, golden: pd.DataFrame, profile: dict
) -> list[Finding]:
    out: list[Finding] = []

    for feat in CONTINUOUS:
        ref = golden[feat].dropna()
        cur = win[feat].dropna()
        if len(cur) < 50 or len(ref) < 50:
            continue
        d, p = ks_statistic(ref, cur)
        crit = ks_critical(len(ref), len(cur))
        # D must clear both the critical value *and* a floor, so that huge samples don't
        # flag differences too small for anyone to act on.
        if d > crit and d > 0.10:
            out.append(
                Finding(
                    "feature_drift",
                    LAYER[feat],
                    "critical" if d > 0.25 else "warn",
                    feat,
                    f"KS D={d:.3f} (D_crit={crit:.3f}, p={p:.2g})",
                    d,
                )
            )

    for feat in CATEGORICAL:
        value = psi(golden[feat], win[feat])
        if value > 0.10:
            out.append(
                Finding(
                    "feature_drift",
                    LAYER[feat],
                    "critical" if value > 0.25 else "warn",
                    feat,
                    f"PSI={value:.3f} (>0.25 is a major shift in the traffic mix)",
                    value,
                )
            )
    return out


# ---------------------------------------------------------------------------
# Probe 3 — Serving parity: does production match how the model was trained?
# ---------------------------------------------------------------------------


def check_serving_parity(win: pd.DataFrame, profile: dict) -> list[Finding]:
    out: list[Finding] = []

    trained_px = profile["preprocessing"]["resize_px"]
    served = win["resize_px"].value_counts()
    for px, n in served.items():
        if int(px) != int(trained_px):
            share = n / len(win)
            out.append(
                Finding(
                    "serving_parity",
                    "Model Performance",
                    "critical",
                    "preprocessing.resize_px",
                    f"serving at {int(px)}px, model trained at {int(trained_px)}px "
                    f"({share:.0%} of traffic)",
                    float(share),
                )
            )

    for feat, spec in profile["features"].items():
        cur = win[feat].dropna()
        if cur.empty:
            continue
        z = abs(cur.mean() - spec["mean"]) / spec["std"]
        if z > 0.5:
            out.append(
                Finding(
                    "serving_parity",
                    "Data Quality",
                    "critical" if z > 1.5 else "warn",
                    feat,
                    f"mean {cur.mean():.3f} is {z:.1f}σ from the training mean "
                    f"{spec['mean']:.3f}",
                    float(z),
                )
            )
    return out


# ---------------------------------------------------------------------------
# Probe 4 — Model performance: is the judgment degrading?
# ---------------------------------------------------------------------------


def check_performance(
    labeled: pd.DataFrame, profile: dict, segment_by: tuple[str, ...] = tuple(SEGMENTS)
) -> list[Finding]:
    out: list[Finding] = []
    offline = profile["offline_metrics"]["accuracy"]

    overall = accuracy(labeled)
    if overall is None:
        return [
            Finding(
                "performance", "Model Performance", "info", "accuracy",
                "no confirmed labels have arrived for this window yet",
            )
        ]

    # A confirmed label the model was never trained to emit: the label space itself has
    # moved. The model cannot get these right — no amount of retraining on the old classes
    # will help, because the classes are new.
    unseen = set(labeled["confirmed_label"].dropna()) - set(profile["label_space"])
    if unseen:
        share = float(labeled["confirmed_label"].isin(unseen).mean())
        out.append(
            Finding(
                "performance",
                "Data Quality",
                "critical",
                "label_space",
                f"{share:.1%} of confirmed labels are outside the training label space "
                f"({', '.join(sorted(unseen))})",
                share,
            )
        )

    gap = offline - overall
    if gap > 0.05:
        out.append(
            Finding(
                "performance",
                "Model Performance",
                "critical" if gap > 0.10 else "warn",
                "accuracy (all traffic)",
                f"{overall:.1%} on confirmed labels vs {offline:.1%} offline "
                f"({gap*100:.1f} pts below) — n={len(labeled)}",
                overall,
            )
        )

    # The reason to segment: an aggregate number is an average over the healthy majority.
    # Compare the worst slice against `overall`, not against `offline` — once global accuracy
    # has fallen, *every* slice is below the offline number, and comparing to offline just
    # restates the global regression once per segment. What you want to know is which slice
    # is dragging the average down.
    for by in segment_by:
        table = segment_table(labeled, by)
        if table.empty or len(table) < 2:
            continue
        worst = table.iloc[0]
        if overall - worst["accuracy"] > 0.10:
            out.append(
                Finding(
                    "performance",
                    "Model Performance",
                    "critical",
                    f"{by}={worst[by]}",
                    f"accuracy {worst['accuracy']:.1%} vs {overall:.1%} overall "
                    f"(n={int(worst['n'])})",
                    float(worst["accuracy"]),
                )
            )
    return out
