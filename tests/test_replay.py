"""The replay is the payoff of the tutorial: each planted failure is caught one day after it
starts, by the probe built to see it. Pin those dates so the story can't drift silently."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest

from answers import probes_solution as sol
from monitor.cli import INCIDENT_DAY, run_all


@pytest.fixture(scope="module")
def first_alerts(df, profile):
    """Re-implements the loop in cmd_replay so the test doesn't depend on printed output."""
    start = df["ts"].min().to_pydatetime() + timedelta(days=8)
    end = df["ts"].max().to_pydatetime()
    streak, first = {}, {}
    day = start
    while day <= end:
        fired = set()
        for f in run_all(df, profile, day, 7, 28, sol):
            if f.severity == "info":
                continue
            key = (f.probe, f.subject)
            fired.add(key)
            streak[key] = streak.get(key, 0) + 1
            if streak[key] == 2 and key not in first:
                first[key] = (day - timedelta(days=1)).date()
        for key in list(streak):
            if key not in fired:
                streak[key] = 0
        day += timedelta(days=1)
    return first


def test_the_humans_noticed_on_the_day_after_the_log_ends(df):
    assert INCIDENT_DAY == datetime(2026, 11, 3)
    assert df["ts"].max().date() == (INCIDENT_DAY - timedelta(days=1)).date()


def test_each_planted_failure_is_caught_one_day_after_it_starts(first_alerts):
    # Failure 1: partner uploader goes live 4 Sep -> categorical drift fires 5 Sep.
    assert first_alerts[("feature_drift", "image_source")] == date(2026, 9, 5)
    # Failure 2: v3.2 ships at 224px on 24 Sep -> parity fires 25 Sep.
    assert first_alerts[("serving_parity", "preprocessing.resize_px")] == date(2026, 9, 25)
    # Failure 3: Nordics launch 8 Oct -> region mix drifts by 13 Oct; label space 21 Oct.
    assert first_alerts[("feature_drift", "region")] == date(2026, 10, 13)
    assert first_alerts[("performance", "label_space")] == date(2026, 10, 21)


def test_ground_truth_is_the_last_signal_to_arrive(first_alerts):
    """Segmented accuracy on the newest failure fires four days before the humans did."""
    nordics = first_alerts[("performance", "region=nordics")]
    assert (INCIDENT_DAY.date() - nordics).days == 4
    # ...while the earliest signal was 59 days ahead.
    earliest = min(first_alerts.values())
    assert (INCIDENT_DAY.date() - earliest).days == 59


def test_thirteen_distinct_alerts_in_total(first_alerts):
    assert len(first_alerts) == 13
