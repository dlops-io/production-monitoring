"""Every finding and replay line quoted in the README must be what the CLI actually prints.

The README quotes real output on purpose — students compare their screen against it. This
test runs the reference solution and checks that each quoted finding line exists in the
real output (whitespace-normalised, so column alignment can change freely).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text()


def _norm(line: str) -> str:
    return re.sub(r"\s+", " ", line).strip()


def _run(*args: str) -> str:
    out = subprocess.run(
        [sys.executable, "-m", "monitor.cli", *args, "--solution"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    return "\n".join(_norm(l) for l in out.splitlines())


@pytest.fixture(scope="module")
def check_output():
    return _run("check")


@pytest.fixture(scope="module")
def replay_output():
    return _run("replay")


def test_readme_finding_lines_are_real(check_output):
    quoted = [_norm(l) for l in README.splitlines() if re.match(r"^\s*!(!| ) \[", l)]
    assert len(quoted) >= 10, "the README should quote the probe findings"
    for line in quoted:
        assert line in check_output, f"README quotes a finding the CLI does not print:\n  {line}"


def test_readme_replay_lines_are_real(replay_output):
    quoted = [_norm(l) for l in README.splitlines() if re.match(r"^\s*2026-\d\d-\d\d\s+\d+d\s+\S+ / ", l)]
    assert len(quoted) == 13
    for line in quoted:
        assert line in replay_output, f"README quotes a replay row the CLI does not print:\n  {line}"


def test_readme_headline_numbers_are_real(check_output):
    for line in ["as of 2026-11-02", "drift window last 7d 951 events",
                 "performance window last 28d 717 of those events have a confirmed label so far (19%)",
                 "accuracy on those 67.8%"]:
        assert line in check_output
        assert _norm(line).split(" ")[-1] in README
