"""Run this after every probe you write:  pytest tests/test_probes_contract.py

It does not check your thresholds — those are yours. It checks that each probe keeps the
contract the CLI depends on: it returns a list of `Finding`s with a valid severity and
layer, and it never crashes on an early window where no labels have arrived yet.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from monitor import probes
from monitor.core import Finding, labels_known_by, window

SEVERITIES = {"info", "warn", "critical"}
LAYERS = {"Data Quality", "Model Performance", "System Health"}


def _check(findings):
    assert isinstance(findings, list), "a probe returns a list (empty list = green)"
    for f in findings:
        assert isinstance(f, Finding), f"{f!r} is not a Finding"
        assert f.severity in SEVERITIES, f"severity {f.severity!r} — use info / warn / critical"
        assert f.layer in LAYERS, f"layer {f.layer!r} — use one of {sorted(LAYERS)}"
        assert f.probe and f.subject and f.detail, "probe, subject and detail must be non-empty"


def test_check_data_quality_contract(win, profile):
    _check(probes.check_data_quality(win, profile))


def test_check_feature_drift_contract(win, golden, profile):
    _check(probes.check_feature_drift(win, golden, profile))


def test_check_serving_parity_contract(win, profile):
    _check(probes.check_serving_parity(win, profile))


def test_check_performance_contract(labeled, profile):
    _check(probes.check_performance(labeled, profile))


def test_check_performance_survives_a_window_with_no_labels(df, profile):
    """Day 6 of production: nothing has been labelled yet. `replay` hits this every run."""
    early = datetime(2026, 8, 10)
    nothing = labels_known_by(window(df, early, 28), early)
    assert nothing.empty
    _check(probes.check_performance(nothing, profile))


@pytest.mark.parametrize("name", ["check_data_quality", "check_feature_drift",
                                  "check_serving_parity", "check_performance"])
def test_probe_exists(name):
    assert callable(getattr(probes, name, None)), f"monitor/probes.py must define {name}()"
