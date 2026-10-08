"""monitor/core.py is given to students, so it had better be right.

The one test to read is `test_labels_known_by_excludes_labels_from_the_future`: it is the
whole honesty of the tutorial in six lines.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest

from monitor.core import (
    Finding,
    accuracy,
    ks_critical,
    ks_statistic,
    labels_known_by,
    psi,
    segment_table,
    window,
)

# --- the statistics ----------------------------------------------------------------------


def test_ks_statistic_is_near_zero_for_same_distribution():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(size=5000))
    b = pd.Series(rng.normal(size=5000))
    d, p = ks_statistic(a, b)
    assert d < 0.05
    assert p > 0.01


def test_ks_statistic_detects_a_shift():
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0.0, 1.0, size=2000))
    b = pd.Series(rng.normal(1.0, 1.0, size=2000))
    d, p = ks_statistic(a, b)
    assert d > 0.3
    assert p < 1e-10


def test_ks_statistic_ignores_nulls():
    a = pd.Series([1.0, 2.0, 3.0, np.nan])
    b = pd.Series([1.0, 2.0, 3.0])
    d, _ = ks_statistic(a, b)
    assert d == 0.0


def test_ks_critical_formula_and_that_it_shrinks_with_sample_size():
    assert ks_critical(100, 100, alpha=0.05) == pytest.approx(1.36 * np.sqrt(200 / 10_000))
    assert ks_critical(4000, 1000) < ks_critical(400, 100) < ks_critical(40, 10)


def test_psi_is_zero_for_identical_mix_and_large_for_a_new_category():
    ref = pd.Series(["na"] * 60 + ["eu"] * 40)
    assert psi(ref, ref) == pytest.approx(0.0, abs=1e-6)
    cur = pd.Series(["na"] * 45 + ["eu"] * 30 + ["nordics"] * 25)
    assert psi(ref, cur) > 0.25  # a category that never existed in the reference


# --- windowing ----------------------------------------------------------------------------


def _frame(ts, label_ts=None):
    return pd.DataFrame(
        {
            "ts": pd.to_datetime(ts, format="mixed"),
            "label_ts": pd.to_datetime(label_ts if label_ts is not None else [None] * len(ts), format="mixed"),
            "pred_label": ["a"] * len(ts),
            "confirmed_label": ["a"] * len(ts),
        }
    )


def test_window_is_right_closed_and_left_open():
    df = _frame(["2026-09-01", "2026-09-02 12:00", "2026-09-04", "2026-09-05"])
    out = window(df, datetime(2026, 9, 4), days=2)
    assert list(out["ts"].dt.strftime("%Y-%m-%d %H:%M")) == ["2026-09-02 12:00", "2026-09-04 00:00"]


def test_labels_known_by_excludes_labels_from_the_future():
    # Three predictions on 1 Sep. Their labels arrive on 5 Sep, 20 Sep, and never.
    df = _frame(
        ["2026-09-01", "2026-09-01", "2026-09-01"],
        ["2026-09-05", "2026-09-20", None],
    )
    known = labels_known_by(df, datetime(2026, 9, 10))
    # On 10 Sep only the first label exists. The second is in the CSV, but you don't have it yet.
    assert len(known) == 1
    assert known["label_ts"].iloc[0] == pd.Timestamp("2026-09-05")
    # Filtering on `confirmed_label.notna()` instead would have returned all three.
    assert df["confirmed_label"].notna().sum() == 3


def test_accuracy_is_none_on_empty_and_a_fraction_otherwise():
    assert accuracy(pd.DataFrame(columns=["pred_label", "confirmed_label"])) is None
    df = pd.DataFrame({"pred_label": ["a", "a", "b", "b"], "confirmed_label": ["a", "b", "b", "b"]})
    assert accuracy(df) == 0.75


def test_segment_table_drops_small_slices_and_sorts_worst_first():
    df = pd.DataFrame(
        {
            "seg": ["x"] * 50 + ["y"] * 50 + ["tiny"] * 10,
            "pred_label": ["a"] * 110,
            "confirmed_label": ["a"] * 45 + ["b"] * 5 + ["a"] * 20 + ["b"] * 30 + ["a"] * 10,
        }
    )
    table = segment_table(df, "seg", min_n=40)
    assert list(table["seg"]) == ["y", "x"]  # worst first
    assert "tiny" not in set(table["seg"])


def test_finding_renders_with_a_severity_mark():
    f = Finding("p", "Data Quality", "critical", "brightness", "null rate 4.8%", 0.048)
    assert str(f).startswith("!! [Data Quality")
    assert str(Finding("p", "Data Quality", "warn", "s", "d")).startswith("! ")
