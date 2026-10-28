"""The committed data is the incident. If any of this changes, the README's numbers are wrong."""

from __future__ import annotations

import pandas as pd

from monitor.core import CATEGORICAL, CONTINUOUS, LAYER, SEGMENTS

COLUMNS = [
    "event_id", "ts", "model_version", "image_source", "region", "resize_px",
    "blur_score", "brightness", "saturation",
    "pred_label", "confidence", "latency_ms",
    "label_ts", "confirmed_label",
]


def test_log_schema_and_size(df):
    assert list(df.columns) == COLUMNS
    assert len(df) == 12_322
    assert df["event_id"].is_unique


def test_log_covers_ninety_days_ending_on_the_incident(df):
    assert df["ts"].min().date() == pd.Timestamp("2026-08-05").date()
    assert df["ts"].max().date() == pd.Timestamp("2026-11-02").date()
    assert df["ts"].is_monotonic_increasing


def test_labels_are_sparse_and_late(df):
    labeled = df["confirmed_label"].notna()
    assert labeled.sum() == 3_663
    assert 0.25 < labeled.mean() < 0.35
    delay = (df["label_ts"] - df["ts"]).dropna()
    assert (delay > pd.Timedelta(0)).all(), "a label can never arrive before its prediction"
    assert pd.Timedelta(days=10) < delay.median() < pd.Timedelta(days=13)
    # Labels only exist where a label timestamp exists, and vice versa.
    assert (df["label_ts"].notna() == labeled).all()


def test_every_monitored_column_exists_and_has_a_layer(df):
    for col in CONTINUOUS + CATEGORICAL:
        assert col in df.columns
        assert col in LAYER
    for col in SEGMENTS:
        assert col in df.columns
    assert "model_version" not in CONTINUOUS + CATEGORICAL, "alerting on your own release process"


def test_training_profile_has_what_the_probes_need(profile, df):
    assert profile["preprocessing"]["resize_px"] == 256
    assert set(profile["features"]) == {"blur_score", "brightness", "saturation"}
    for spec in profile["features"].values():
        assert {"mean", "std", "null_rate"} <= set(spec)
    assert len(profile["label_space"]) == 10
    assert profile["offline_metrics"]["accuracy"] == 0.93
    # The model can only emit classes it was trained on.
    assert set(df["pred_label"]) <= set(profile["label_space"])
