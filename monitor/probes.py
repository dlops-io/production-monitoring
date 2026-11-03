"""The four probes. This is the only file you edit.
I think Patrick broke everything !!!!

Every probe takes a slice of the production log and returns a list of `Finding`s — zero
findings means "green". Right now every probe returns nothing, which is why the monitor
reports all-clear on a system that is quietly broken.

Helpers you already have (from monitor.core, import them as you need them):

    ks_statistic(reference, current) -> (D, p)   two-sample KS test
    ks_critical(n, m, alpha=0.01)    -> D_crit    the largest D chance alone would produce
    psi(reference, current)          -> float     categorical drift, >0.25 is a major shift
    accuracy(frame)                  -> float     share of predictions matching confirmed_label
    segment_table(frame, by)         -> DataFrame accuracy and n per value of `by`
    CONTINUOUS, CATEGORICAL, SEGMENTS, LAYER      which columns are which

A Finding is:

    Finding(probe="data_quality", layer="Data Quality", severity="warn",
            subject="brightness", detail="null rate 4.8% (baseline 0.4%)", value=0.048)

`severity` is one of "info", "warn", "critical". Choose deliberately: a critical finding is
one you would page a human for at 2am.
"""

from __future__ import annotations

import pandas as pd

from monitor.core import (  # noqa: F401  (imported for you to use)
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


def check_data_quality(win: pd.DataFrame, profile: dict) -> list[Finding]:
    """Layer 1 — Data Quality: are the inputs broken, before the model even runs?

    TODO (Step 3)
      For each feature in `profile["features"]`, compare its null rate in `win` against the
      `null_rate` the training profile recorded. Fire when it is meaningfully higher.

      Two decisions are yours, and they are the whole exercise:
        - What counts as "meaningfully higher"? A fixed 2%? A multiple of the baseline?
          Both? A feature whose training null rate is 0.4% behaving at 5% is a different
          story from one that was always 30% null.
        - When is it `warn` and when is it `critical`?
    """
    return []


def check_feature_drift(
    win: pd.DataFrame, golden: pd.DataFrame, profile: dict
) -> list[Finding]:
    """Layer 1/2 — has the distribution moved away from the reference window?

    `golden` is the first 30 days of production: the reference distribution. `win` is now.

    TODO (Step 4)
      - For each column in CONTINUOUS: run `ks_statistic(golden[col], win[col])` and compare
        D against `ks_critical(len(ref), len(cur))`. Fire when it clears the critical value.
      - For each column in CATEGORICAL: use `psi()` instead. KS is for continuous data; a
        column like `region` needs the categorical test.
      - Use `LAYER[col]` for the finding's layer.

      Watch out for two things:
        - With ~1000 events a window, statistical significance is cheap. D=0.03 will clear
          D_crit and mean nothing anyone can act on. Consider a floor on D as well.
        - `confidence` is in CONTINUOUS on purpose. It is the model's *output*, not its
          input. Ask yourself what it means when the inputs look stable and the output
          distribution moves.
    """
    return []


def check_serving_parity(win: pd.DataFrame, profile: dict) -> list[Finding]:
    """Does production actually match how the model was trained?

    Not a drift test. Drift is the world changing; this is *us* changing something and the
    model not being told — "training-serving skew". It is one of the most common causes of
    production model failures, and no distribution test on the input features will find
    it, because the input features are fine. Only a comparison against what was true at
    training time can.

    TODO (Step 5)
      - Compare `win["resize_px"]` against `profile["preprocessing"]["resize_px"]`. Any
        traffic served at a different size than the model was trained at is a bug, and it
        is `critical` even at small volume.
      - Then compare each feature's mean in `win` against the training mean in
        `profile["features"]`, in units of the training std. More than ~0.5σ deserves a
        look; 1.5σ is not a coincidence.
    """
    return []


def check_performance(labeled: pd.DataFrame, profile: dict) -> list[Finding]:
    """Layer 2 — Model Performance: is the model's judgment degrading?

    `labeled` only contains events whose confirmed label had *arrived* by the as-of date
    (see `labels_known_by` in core.py). That is deliberate. You cannot grade yourself on
    labels that have not come back yet.

    TODO (Step 6)
      - Compute overall `accuracy(labeled)` and compare it to
        `profile["offline_metrics"]["accuracy"]`. Fire on a real gap.
      - Then do it again per segment, using `segment_table(labeled, by)` for each `by` in
        SEGMENTS, and fire on the worst slice. This is the step that turns "accuracy is
        down" into "accuracy is down *for these requests*", which is the difference between
        a mystery and a fix.
        Careful what you compare the slice against. Once overall accuracy has fallen, every
        slice is below the offline number, and you will emit one near-duplicate finding per
        segment. What you actually want to surface is the slice dragging the average down.
      - Handle the empty case: early in a window there may be no arrived labels at all.
        Returning a single `severity="info"` finding that says so is more honest than
        returning green.

    Stretch: `profile["label_space"]` lists the classes the model was trained on. What does
    it mean if `labeled["confirmed_label"]` contains something that isn't in that list?
    """
    return []
