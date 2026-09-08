"""Generate data/production_log.csv — the 90 days of prediction events used by the tutorial.

⚠️  SPOILER. This file is the answer key for the incident: it says, in code, exactly what
    broke and when. Read it *after* you have run your own probes, not before.

The log is deterministic (fixed seed), so everyone in the class sees the same incident and
the README can quote exact numbers.

Run from the repo root:  python answers/make_production_log.py
"""

import csv
import json
import random
from datetime import date, datetime, timedelta
from pathlib import Path

SEED = 215
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "production_log.csv"

DAY_ONE = date(2026, 8, 5)
N_DAYS = 90
EVENTS_PER_DAY = 150

# ---------------------------------------------------------------------------
# The three planted failures
# ---------------------------------------------------------------------------
# Day 31 (2026-09-04): a partner cheesemonger's uploader goes live. Its images are
#   recompressed and stripped of EXIF, so blur_score collapses and brightness goes null.
#   -> a DATA QUALITY failure. Loud on null-rate, visible on KS.
PARTNER_START = 31
PARTNER_SHARE = 0.30

# Day 51 (2026-09-24): v3.2 ships. Serving resizes to 224px; the model was trained on 256.
#   -> a TRAINING-SERVING SKEW failure. Invisible to every drift test (the *image features*
#      are unchanged), invisible to accuracy for weeks (labels lag), and it makes the
#      latency dashboard look *better*. Only a parity check against the training profile
#      catches it on day one.
V32_START = 51
SKEW_ACC_MULT = 0.86
SKEW_CONF_PENALTY = 0.08
SKEW_LATENCY_MULT = 0.78  # smaller images are faster — the deploy looked like a win

# Day 65 (2026-10-08): formaggio.me launches in the Nordics. Real users start photographing
#   varieties that are not in the training label space at all.
#   -> CONCEPT DRIFT. Input features look perfectly normal. The only early signal is the
#      model's own confidence distribution; confirmed labels arrive ~3 weeks later.
NORDICS_START = 65
NORDICS_RAMP_DAYS = 12
NORDICS_MAX_SHARE = 0.25
NORDICS_ACC = 0.55
NORDICS_CONF_PENALTY = 0.10

TRAINED_LABELS = [
    "halloumi", "gouda", "comte", "brie", "manchego",
    "pecorino", "gruyere", "cheddar", "roquefort", "feta",
]
NORDIC_LABELS = ["vasterbotten", "gammelost", "danablu", "herrgard"]

# who confuses with whom — so wrong predictions look plausible, not random
CONFUSABLE = {
    "halloumi": "feta", "feta": "halloumi",
    "gouda": "cheddar", "cheddar": "gouda",
    "comte": "gruyere", "gruyere": "comte",
    "manchego": "pecorino", "pecorino": "manchego",
    "brie": "roquefort", "roquefort": "brie",
    # nordic varieties the model has never seen collapse onto lookalikes
    "vasterbotten": "gruyere", "gammelost": "roquefort",
    "danablu": "roquefort", "herrgard": "gouda",
}

BASE_ACCURACY = 0.91
PARTNER_ACCURACY = 0.62
LABEL_COVERAGE = 0.35          # share of events that ever get a confirmed label
LABEL_DELAY_DAYS = (8, 14)     # how long a user takes to confirm


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def nordics_share(day: int) -> float:
    if day < NORDICS_START:
        return 0.0
    ramp = min(1.0, (day - NORDICS_START + 1) / NORDICS_RAMP_DAYS)
    return NORDICS_MAX_SHARE * ramp


def main() -> None:
    rng = random.Random(SEED)
    rows = []
    last_ts = datetime.combine(DAY_ONE + timedelta(days=N_DAYS - 1), datetime.min.time())

    for day in range(1, N_DAYS + 1):
        d = DAY_ONE + timedelta(days=day - 1)
        weekend = d.weekday() >= 5
        n = int(EVENTS_PER_DAY * (0.72 if weekend else 1.0) * rng.uniform(0.9, 1.1))

        for _ in range(n):
            ts = datetime.combine(d, datetime.min.time()) + timedelta(
                seconds=rng.randrange(86400)
            )

            partner = day >= PARTNER_START and rng.random() < PARTNER_SHARE
            source = "partner_uploader" if partner else "app"

            if not partner and rng.random() < nordics_share(day):
                region = "nordics"
            else:
                region = "na" if rng.random() < 0.6 else "eu"

            v32 = day >= V32_START
            model_version = "v3.2" if v32 else "v3.1"
            resize_px = 224 if v32 else 256

            # ---- serving-side image features -------------------------------
            if partner:
                blur = clamp(rng.gauss(0.44, 0.10))
                brightness = None if rng.random() < 0.12 else clamp(rng.gauss(0.51, 0.12))
                saturation = clamp(rng.gauss(0.46, 0.10))
            else:
                blur = clamp(rng.gauss(0.72, 0.08))
                brightness = None if rng.random() < 0.004 else clamp(rng.gauss(0.55, 0.10))
                saturation = clamp(rng.gauss(0.48, 0.09))

            # ---- what the model gets right ---------------------------------
            if region == "nordics":
                true_label = rng.choice(NORDIC_LABELS + TRAINED_LABELS[:3])
                p = NORDICS_ACC if true_label in NORDIC_LABELS else BASE_ACCURACY
            else:
                true_label = rng.choice(TRAINED_LABELS)
                p = PARTNER_ACCURACY if partner else BASE_ACCURACY

            if v32:
                p *= SKEW_ACC_MULT
            correct = rng.random() < p
            pred_label = true_label if correct else CONFUSABLE.get(
                true_label, rng.choice(TRAINED_LABELS)
            )
            if pred_label not in TRAINED_LABELS:      # the model can only emit trained classes
                pred_label = CONFUSABLE.get(pred_label, "gouda")
            if not correct and pred_label == true_label:
                pred_label = rng.choice([l for l in TRAINED_LABELS if l != true_label])

            # ---- confidence: the leading indicator -------------------------
            conf = rng.gauss(0.88, 0.06) if correct else rng.gauss(0.62, 0.11)
            if v32:
                conf -= SKEW_CONF_PENALTY
            if region == "nordics" and true_label in NORDIC_LABELS:
                conf -= NORDICS_CONF_PENALTY
            confidence = clamp(conf, 0.05, 0.999)

            latency = rng.lognormvariate(5.15, 0.30)
            if v32:
                latency *= SKEW_LATENCY_MULT

            # ---- delayed ground truth --------------------------------------
            label_ts = ""
            confirmed_label = ""
            if rng.random() < LABEL_COVERAGE:
                lt = ts + timedelta(
                    days=rng.randint(*LABEL_DELAY_DAYS), seconds=rng.randrange(86400)
                )
                if lt <= last_ts:                      # labels from the last ~2 weeks
                    label_ts = lt.isoformat(timespec="seconds")   # have not arrived yet
                    confirmed_label = true_label

            rows.append(
                {
                    "event_id": f"evt_{len(rows):06d}",
                    "ts": ts.isoformat(timespec="seconds"),
                    "model_version": model_version,
                    "image_source": source,
                    "region": region,
                    "resize_px": resize_px,
                    "blur_score": round(blur, 4),
                    "brightness": "" if brightness is None else round(brightness, 4),
                    "saturation": round(saturation, 4),
                    "pred_label": pred_label,
                    "confidence": round(confidence, 4),
                    "latency_ms": round(latency, 1),
                    "label_ts": label_ts,
                    "confirmed_label": confirmed_label,
                }
            )

    rows.sort(key=lambda r: r["ts"])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    labeled = sum(1 for r in rows if r["confirmed_label"])
    print(f"wrote {OUT.relative_to(ROOT)}  ({len(rows):,} events, {labeled:,} confirmed labels)")

    # the training profile the parity probe compares against
    profile = {
        "model_version_trained": "v3.1",
        "training_snapshot": "2026-07-15",
        "preprocessing": {"resize_px": 256, "color_space": "RGB"},
        "features": {
            "blur_score": {"mean": 0.72, "std": 0.08, "null_rate": 0.004},
            "brightness": {"mean": 0.55, "std": 0.10, "null_rate": 0.004},
            "saturation": {"mean": 0.48, "std": 0.09, "null_rate": 0.0},
        },
        "label_space": TRAINED_LABELS,
        "offline_metrics": {"accuracy": 0.93, "macro_f1": 0.91},
    }
    prof_path = ROOT / "reference" / "training_profile.json"
    prof_path.parent.mkdir(parents=True, exist_ok=True)
    prof_path.write_text(json.dumps(profile, indent=2) + "\n")
    print(f"wrote {prof_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
