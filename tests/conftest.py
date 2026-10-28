"""Shared fixtures. Everything here is read-only and fast — the whole suite runs in seconds."""

from __future__ import annotations

from datetime import datetime

import pytest

from monitor.core import golden_window, labels_known_by, load_log, load_profile, window


@pytest.fixture(scope="session")
def df():
    return load_log()


@pytest.fixture(scope="session")
def as_of(df) -> datetime:
    """The CLI's default "today": the last timestamp in the log (2026-11-02, late evening)."""
    return df["ts"].max().to_pydatetime()


@pytest.fixture(scope="session")
def profile():
    return load_profile()


@pytest.fixture(scope="session")
def win(df, as_of):
    """The 7-day drift window as of the incident."""
    return window(df, as_of, 7)


@pytest.fixture(scope="session")
def golden(df):
    return golden_window(df)


@pytest.fixture(scope="session")
def labeled(df, as_of):
    """The 28-day performance window, restricted to labels that had arrived."""
    return labels_known_by(window(df, as_of, 28), as_of)
