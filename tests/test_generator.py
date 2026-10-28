"""The data is generated with a fixed seed. Regenerating it must reproduce the committed
files byte for byte — otherwise every number in the README is up for grabs."""

from __future__ import annotations

from pathlib import Path

from answers import make_production_log

ROOT = Path(__file__).resolve().parent.parent


def test_generator_reproduces_the_committed_data_exactly(tmp_path):
    make_production_log.main(tmp_path)
    assert (tmp_path / "data" / "production_log.csv").read_bytes() == (
        ROOT / "data" / "production_log.csv"
    ).read_bytes()
    assert (tmp_path / "reference" / "training_profile.json").read_bytes() == (
        ROOT / "reference" / "training_profile.json"
    ).read_bytes()
