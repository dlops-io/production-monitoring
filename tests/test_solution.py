"""The reference solution must find exactly the incident the README describes.

These are the numbers quoted in the README's "what you should see" toggles. If a threshold
in answers/probes_solution.py changes, this tells you which quoted output to update.
"""

from __future__ import annotations

from datetime import datetime

import pytest

from answers import probes_solution as sol
from monitor.cli import run_all
from monitor.core import labels_known_by, window


def _subjects(findings):
    return {(f.probe, f.subject): f for f in findings}


def test_data_quality_finds_only_brightness(win, profile):
    out = _subjects(sol.check_data_quality(win, profile))
    assert set(out) == {("data_quality", "brightness")}
    assert out[("data_quality", "brightness")].severity == "warn"
    assert out[("data_quality", "brightness")].value == pytest.approx(0.048, abs=0.001)


def test_feature_drift_finds_five_including_the_three_traps(win, golden, profile):
    out = _subjects(sol.check_feature_drift(win, golden, profile))
    assert {s for _, s in out} == {"blur_score", "confidence", "latency_ms", "image_source", "region"}
    assert out[("feature_drift", "latency_ms")].layer == "System Health"  # drifted *better*
    assert out[("feature_drift", "confidence")].value > 0.5  # the model's output moved most
    assert all(f.severity == "critical" for f in out.values())


def test_serving_parity_catches_the_resize_skew_on_all_traffic(win, profile):
    out = _subjects(sol.check_serving_parity(win, profile))
    skew = out[("serving_parity", "preprocessing.resize_px")]
    assert skew.severity == "critical"
    assert skew.value == pytest.approx(1.0)
    assert "224px" in skew.detail and "256px" in skew.detail


def test_serving_parity_during_the_rollout(df, profile):
    before = window(df, datetime(2026, 9, 24), 7)
    during = window(df, datetime(2026, 9, 26), 7)
    assert ("serving_parity", "preprocessing.resize_px") not in _subjects(sol.check_serving_parity(before, profile))
    share = _subjects(sol.check_serving_parity(during, profile))[("serving_parity", "preprocessing.resize_px")].value
    assert 0.25 < share < 0.40  # caught mid-rollout, at roughly a third of traffic


def test_performance_finds_the_gap_the_slices_and_the_label_space(labeled, profile):
    out = _subjects(sol.check_performance(labeled, profile))
    assert set(out) == {
        ("performance", "label_space"),
        ("performance", "accuracy (all traffic)"),
        ("performance", "image_source=partner_uploader"),
        ("performance", "region=nordics"),
    }
    assert out[("performance", "accuracy (all traffic)")].value == pytest.approx(0.678, abs=0.001)
    assert out[("performance", "image_source=partner_uploader")].value == pytest.approx(0.528, abs=0.001)
    assert out[("performance", "region=nordics")].value == pytest.approx(0.355, abs=0.001)
    for name in ("danablu", "gammelost", "herrgard", "vasterbotten"):
        assert name in out[("performance", "label_space")].detail


def test_performance_is_honest_when_no_labels_have_arrived(df, profile):
    early = datetime(2026, 8, 10)
    out = sol.check_performance(labels_known_by(window(df, early, 28), early), profile)
    assert len(out) == 1 and out[0].severity == "info"


def test_all_four_probes_together_match_the_full_report(df, profile, as_of):
    findings = run_all(df, profile, as_of, 7, 28, sol)
    assert len(findings) == 12
    assert sum(f.severity == "critical" for f in findings) == 10
    assert sum(f.severity == "warn" for f in findings) == 2
