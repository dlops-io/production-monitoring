"""The monitor's command line. Given to you — you only edit monitor/probes.py.

    python -m monitor.cli check                       # today's report
    python -m monitor.cli check --as-of 2026-09-10    # what you would have seen that day
    python -m monitor.cli segments --by image_source  # accuracy per slice
    python -m monitor.cli replay                     # when would each probe have fired?

Add --solution to any command to run answers/probes_solution.py instead of your own probes.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta

from monitor.core import (
    SEGMENTS,
    Finding,
    accuracy,
    golden_window,
    labels_known_by,
    load_log,
    load_profile,
    print_findings,
    segment_table,
    window,
)

# The day the humans finally noticed, in the Slack message that opens the tutorial.
INCIDENT_DAY = datetime(2026, 11, 3)


def get_probes(use_solution: bool):
    if use_solution:
        from answers import probes_solution as p
    else:
        from monitor import probes as p
    return p


def run_all(df, profile, as_of, win_days, perf_days, probes) -> list[Finding]:
    win = window(df, as_of, win_days)
    golden = golden_window(df)
    labeled = labels_known_by(window(df, as_of, perf_days), as_of)

    findings: list[Finding] = []
    findings += probes.check_data_quality(win, profile)
    findings += probes.check_feature_drift(win, golden, profile)
    findings += probes.check_serving_parity(win, profile)
    findings += probes.check_performance(labeled, profile)
    return findings


def cmd_check(args) -> None:
    df, profile = load_log(), load_profile()
    as_of = args.as_of or df["ts"].max().to_pydatetime()
    probes = get_probes(args.solution)

    win = window(df, as_of, args.window)
    labeled = labels_known_by(window(df, as_of, args.perf_window), as_of)
    acc = accuracy(labeled)

    print(f"\nas of {as_of:%Y-%m-%d}")
    print(f"  drift window      last {args.window}d   {len(win):,} events")
    print(
        f"  performance window last {args.perf_window}d  "
        f"{len(labeled):,} of those events have a confirmed label so far "
        f"({len(labeled)/max(len(window(df, as_of, args.perf_window)),1):.0%})"
    )
    print(f"  accuracy on those  {acc:.1%}" if acc is not None else "  accuracy   n/a")

    findings = run_all(df, profile, as_of, args.window, args.perf_window, probes)
    print_findings(findings, "probe results")
    print()


def cmd_segments(args) -> None:
    df, profile = load_log(), load_profile()
    as_of = args.as_of or df["ts"].max().to_pydatetime()
    labeled = labels_known_by(window(df, as_of, args.window), as_of)

    print(f"\naccuracy by {args.by} — last {args.window}d as of {as_of:%Y-%m-%d}")
    print(f"overall: {accuracy(labeled):.1%}  (n={len(labeled):,})\n")
    table = segment_table(labeled, args.by)
    if table.empty:
        print("  not enough labelled events in this window")
        return
    for _, row in table.iterrows():
        bar = "█" * int(row["accuracy"] * 40)
        print(f"  {str(row[args.by]):<18} n={int(row['n']):>5}  {row['accuracy']:>6.1%}  {bar}")
    print()


def cmd_confidence(args) -> None:
    """Weekly mean confidence per segment — a performance signal that needs no labels.

    This is the idea behind NannyML's CBPE, stripped to its simplest form: when a model
    starts meeting inputs it was never trained on, its own confidence sags *before* any
    ground truth arrives to prove it wrong.
    """
    df = load_log()
    df = df[df["ts"] > df["ts"].max() - timedelta(days=args.days)]
    df = df.assign(week=df["ts"].dt.to_period("W").dt.start_time)

    print(f"\nmean confidence by {args.by}, per week (last {args.days}d)")
    print("  low% = share of predictions under 0.70\n")
    for value, grp in df.groupby(args.by):
        print(f"  {args.by}={value}")
        for wk, g in grp.groupby("week"):
            if len(g) < 30:
                continue
            low = float((g["confidence"] < 0.70).mean())
            bar = "█" * int(g["confidence"].mean() * 40)
            print(
                f"    {wk:%Y-%m-%d}  n={len(g):>4}  mean={g['confidence'].mean():.3f}  "
                f"low={low:>5.1%}  {bar}"
            )
        print()


def cmd_replay(args) -> None:
    """Walk the log day by day and record when each probe would first have alerted.

    Requiring `--consecutive` windows before alerting is the multiple-testing discipline
    from the lecture: running ~15 tests a day guarantees the occasional false positive, so
    a single significant result is a reason to look, not a reason to page anyone.
    """
    df, profile = load_log(), load_profile()
    probes = get_probes(args.solution)

    start = df["ts"].min().to_pydatetime() + timedelta(days=args.window + 1)
    end = df["ts"].max().to_pydatetime()

    streak: dict[tuple[str, str], int] = {}
    first_alert: dict[tuple[str, str], datetime] = {}
    day = start
    while day <= end:
        fired = set()
        for f in run_all(df, profile, day, args.window, args.perf_window, probes):
            if f.severity == "info":
                continue
            key = (f.probe, f.subject)
            fired.add(key)
            streak[key] = streak.get(key, 0) + 1
            if streak[key] == args.consecutive and key not in first_alert:
                first_alert[key] = day - timedelta(days=args.consecutive - 1)
        for key in list(streak):
            if key not in fired:
                streak[key] = 0
        day += timedelta(days=1)

    print(f"\nreplay — {args.window}d window, alert after {args.consecutive} consecutive days\n")
    print(f"  {'first alert':<14} {'days early':>10}  probe / subject")
    print(f"  {'-'*14} {'-'*10}  {'-'*44}")
    for (probe, subject), when in sorted(first_alert.items(), key=lambda kv: kv[1]):
        early = (INCIDENT_DAY - when).days
        print(f"  {when:%Y-%m-%d}     {early:>7}d   {probe} / {subject}")
    print(f"\n  the humans noticed on {INCIDENT_DAY:%Y-%m-%d}.\n")


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def main() -> None:
    ap = argparse.ArgumentParser(prog="monitor", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add_common(parser):
        parser.add_argument(
            "--solution", action="store_true", help="run answers/probes_solution.py"
        )

    c = sub.add_parser("check", help="run every probe once")
    c.add_argument("--as-of", type=parse_date, default=None)
    c.add_argument("--window", type=int, default=7)
    c.add_argument("--perf-window", type=int, default=28)
    add_common(c)
    c.set_defaults(func=cmd_check)

    s = sub.add_parser("segments", help="accuracy per slice of traffic")
    s.add_argument("--by", default="image_source", choices=SEGMENTS)
    s.add_argument("--as-of", type=parse_date, default=None)
    s.add_argument("--window", type=int, default=28)
    add_common(s)
    s.set_defaults(func=cmd_segments)

    f = sub.add_parser("confidence", help="label-free performance signal, per segment")
    f.add_argument("--by", default="region", choices=SEGMENTS)
    f.add_argument("--days", type=int, default=56)
    add_common(f)
    f.set_defaults(func=cmd_confidence)

    r = sub.add_parser("replay", help="when would each probe first have alerted?")
    r.add_argument("--window", type=int, default=7)
    r.add_argument("--perf-window", type=int, default=28)
    r.add_argument("--consecutive", type=int, default=2)
    add_common(r)
    r.set_defaults(func=cmd_replay)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
