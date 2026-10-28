# Production Monitoring — The Incident Nobody Noticed

**AC215 · Lecture 17 — Evaluating and Monitoring ML Systems in Production**

A cheese-identification model has been serving `formaggio.me` for three months. It passed
every offline test. It is now getting a third of its predictions wrong, and nobody knows
why. Your job is to find out — and then to build the monitor that would have caught it
weeks earlier.

In the lecture you worked four incidents from the outside, with the evidence handed to you
one slide at a time. Today you are inside one. The log is on your disk, the probes are
empty, and the protocol is the same:

```
OBSERVE  →  LOCALIZE  →  INSTRUMENT  →  DECIDE  →  VERIFY
```

---

## What you'll build

Four monitoring probes, wired into a small CLI, running against 90 days of prediction
events from a deployed model:

| Probe | Question it answers | Boundary (lecture) |
|---|---|---|
| `check_data_quality` | Are the inputs broken before the model even runs? | Input |
| `check_feature_drift` | Has the distribution moved away from the reference window? | Input / Model |
| `check_serving_parity` | Does production match how the model was trained? | Serving |
| `check_performance` | Is the model's judgment actually degrading? | Model / Outcome |

By the end you will have replayed all 90 days through your own probes and found the exact
date each failure became visible. There is more than one failure.

**You will not train a model, provision a cloud resource, or wait for anything.** Every
command in this tutorial runs in under three seconds.

## What you'll learn

- Why "accuracy is down" is a feeling, not a finding — and how to turn it into one.
- How to evaluate honestly when labels are sparse and late (the inner-join trap, in pandas).
- The difference between **drift** (the world changed) and **skew** (you changed something),
  and why no distribution test can see the second one.
- Why an alert fires only after sustained evidence, and what "critical" should mean.
- How to develop a Python tool **inside a container** with your editor on the outside.

## Prerequisites

Complete **[Tutorial 0 - Setup & Installs](https://github.com/dlops-io/ac215-setup)** first.
It covers everything this tutorial assumes: VS Code, Git, Docker Desktop, and `uv`.

> You don't need GCP credentials (a service account) for this tutorial. No bucket, no
> project, no key file. Everything is local and the data is committed to the repo. Step 9
> covers where each piece goes when you move it to the cloud for Milestone 3.

## Contents

✍️ = you write code here. ▶️ = you run something and read it.

| | Step | Protocol | You produce |
|---|---|---|---|
| ▶️ | 0. [The incident](#0-the-incident) | OBSERVE | a written guess, in `probes.py` |
| ▶️ | 1. [Get the monitor running — in a container](#1-get-the-monitor-running--in-a-container) | | a running container |
| ▶️ | 2. [Look at what you actually have](#2-look-at-what-you-actually-have) | OBSERVE | — |
| ✍️ | 3. [Probe 1 — Data quality](#3-probe-1--data-quality) | INSTRUMENT | `check_data_quality()` |
| ✍️ | 4. [Probe 2 — Feature drift](#4-probe-2--feature-drift) | INSTRUMENT | `check_feature_drift()` |
| ✍️ | 5. [Probe 3 — Serving parity](#5-probe-3--serving-parity) | INSTRUMENT | `check_serving_parity()` |
| ✍️ | 6. [Probe 4 — Model performance](#6-probe-4--model-performance) | LOCALIZE | `check_performance()` |
| ▶️ | 7. [The signal that needs no labels](#7-the-signal-that-needs-no-labels) | LOCALIZE | — |
| ▶️ | 8. [Replay: how early could you have known?](#8-replay-how-early-could-you-have-known) | VERIFY | the failure → signal table, filled in |
| ▶️ | 9. [Wire it into your own project](#9-wire-it-into-your-own-project) | DECIDE | — |

> [!WARNING]
> **Steps 3–6 do not work if you only read them.** Each one asks you to replace a
> `return []` in `monitor/probes.py` and run the monitor before you continue. The expected
> output is hidden behind a *"what you should see"* toggle in each step — open it **after**
> your own probe fires, not before. If you reach Step 8 without having written four
> functions, `replay` will print nothing and the ending lands flat.

---

## 0. The incident

> **#formaggio-eng** — Monday, 3 November, 9:14 AM
>
> **@pavlos:** the cheese classifier is a disaster. accuracy on confirmed user labels was
> **92%** when we launched in August. last 28 days it's **68%**. offline eval still says
> 93%. nobody changed the model. what happened?
>
> **@you:** …

**Before you run anything, write down your guess.** One sentence, in a comment at the top of
`monitor/probes.py`. You watched a fraud model lose money while every dashboard stayed
green. You watched a camera report `NORMAL, 0.91` at a frozen screen. You have a prior.
Commit to it.

> [!TIP]
> **Commit before the next slide.** The lecture made you pick A, B, C or D before it showed
> you any evidence, and it did that for a reason: the most expensive habit in production ML
> is forming a diagnosis *while* reading the dashboard, because you will find evidence for
> whatever you already suspect. A guess written down first is the only one that can turn
> out to be wrong — and finding out it was wrong is the point.

Keep your guess. Step 8 asks you to score it.

## 1. Get the monitor running — in a container

```bash
git clone https://github.com/dlops-io/production-monitoring.git
cd production-monitoring
```

Everything in this tutorial happens **inside a container**, the way every component of your
project will. The `Dockerfile` is the same shape as the one you wrote for the Mega
Pipeline: a slim Python image, `uv`, a non-root user, and a shell that drops you into the
virtual environment. You don't need to touch it, but read it once — it is twenty lines.

```bash
./docker-shell.sh
```

That builds the image (a minute or two the first time; a few seconds after that) and drops you
into a shell at `/app`, with the venv already active. Your prompt changes to something like
`app@3f2c…:/app$`. From here on, **every command in this tutorial is typed inside that
shell.**

```bash
python -m monitor.cli check
```

You should see:

```
as of 2026-11-02
  drift window      last 7d   951 events
  performance window last 28d  717 of those events have a confirmed label so far (19%)
  accuracy on those  67.8%

probe results
-------------
  (all probes green)
```

There it is. **Accuracy is 68% and every probe is green** — because every probe currently
returns an empty list. This is not a contrived starting point; it is what most deployed
models actually have. Logging is on, a dashboard exists, and nothing in it is watching.

> *A green dashboard says its checks passed. Production evaluation asks whether you checked
> the right thing.* — last slide of the lecture. Today you build the right things.

### How the development loop works

Open the repo in VS Code **on your laptop**, as usual. Leave the container shell open in a
terminal. When you edit `monitor/probes.py` in Step 3 and rerun `python -m monitor.cli check`
inside the container, your change is already there. No rebuild, no copy.

Look at `docker-shell.sh` to see why: the repo is bind-mounted over `/app` with
`-v "$(pwd)":/app`, exactly like the Mega Pipeline. The container runs the code; your editor
owns the files.

> [!TIP]
> **🤔 One question to answer before you move on.** The `Dockerfile` puts the virtual
> environment at `/home/app/.venv`, not at `/app/.venv` where `uv` would normally put it.
> Why? *Hint: what is sitting on top of `/app` while the container runs?* If you can answer
> that, you understand bind mounts. (The answer is in the Troubleshooting table at the end.)

> [!NOTE]
> **Windows:** run `./docker-shell.sh` from **Git Bash**, not `cmd` or PowerShell. If you
> see `the input device is not a TTY`, prefix with `winpty`. Both gotchas are explained in
> the [Mega Pipeline README](https://github.com/dlops-io/mega-pipeline#notes-for-running-on-windows).

<details>
<summary><b>No Docker on this machine?</b> Run it with <code>uv</code> directly.</summary>

```bash
uv sync
source .venv/bin/activate        # Windows (Git Bash): source .venv/Scripts/activate
python -m monitor.cli check
```

Every command after this point is identical. You lose nothing today — but your Milestone 3
monitor will run in a container, so come back and do it the container way when you can.

</details>

## 2. Look at what you actually have

Two files matter.

**`data/production_log.csv`** — one row per prediction the deployed model served, 12,322 of
them over 90 days:

```
event_id, ts, model_version, image_source, region, resize_px,
blur_score, brightness, saturation,          ← features computed at serving time
pred_label, confidence, latency_ms,          ← what the model did
label_ts, confirmed_label                    ← ground truth, when it eventually arrives
```

This is the lecture's `prediction_event` and `outcome_event` in one table, joined on
`event_id`. Notice the last two columns: not just *what* the true label was, but *when* it
arrived.

**`reference/training_profile.json`** — what was true when the model was trained: the
preprocessing config, per-feature mean / std / null rate, the label space, and the offline
metrics it shipped with. Twenty lines of JSON, written once at deploy time. Remember it.

Now the part that makes production evaluation hard. Run this:

```bash
python -c "
import pandas as pd
df = pd.read_csv('data/production_log.csv', parse_dates=['ts','label_ts'])
print('events                ', len(df))
print('with confirmed label  ', df.confirmed_label.notna().sum())
print('median label delay    ', (df.label_ts - df.ts).median())
"
```

```
events                 12322
with confirmed label   3663
median label delay     11 days 11:07:05
```

**Only 30% of predictions ever get a label, and those arrive 11 days late.** You are being
asked to detect a failure using data where the answer key is mostly missing and always
stale. In the lecture, chargebacks took 15–45 days. Same problem, smaller numbers.

> [!TIP]
> **The inner-join trap, in pandas.** Remember the query that said v8 was fine? It joined
> predictions to outcomes and silently dropped every prediction whose label hadn't arrived.
> This repo has the same trap in a different shape. Look at `labels_known_by(df, as_of)` in
> `monitor/core.py`: it keeps only rows whose `label_ts` had *already happened* by the as-of
> date. Every accuracy number in this repo goes through it.
>
> Without it, when you replay September in Step 8 you would be grading September's
> predictions with labels that arrived in October — labels from the future. The replay would
> look brilliant and prove nothing. Same discipline as the lecture's label-complete cohort:
> **only evaluate what you could actually have known at the time.**

Two reference points, for two different questions:

| Reference | Where it comes from | Answers |
|---|---|---|
| **Golden window** | first 30 days of the log (`golden_window()`) | "has the world changed since we were healthy?" |
| **Training profile** | `reference/training_profile.json` | "does production match what we trained on?" |

Those are not the same question, and Step 5 is where the difference bites.

**Before Step 3 — open `monitor/core.py` and skim four things.** You never edit this file, but
every probe you're about to write is built out of it, so five minutes here saves you twenty
later:

| In `core.py` | Why you care |
|---|---|
| `Finding` | the dataclass every probe returns. Note the `severity` field. |
| `labels_known_by()` | the twelve lines above. Read the docstring. |
| `ks_statistic` / `ks_critical` / `psi` | the statistics, already written. Step 4 uses all three. |
| `CONTINUOUS`, `CATEGORICAL`, `SEGMENTS`, `LAYER` | which columns are which — and the comment explaining why `model_version` is in one list and not the other. |

Then run the test suite once, so you know what green looks like:

```bash
pytest
```

All of it passes right now. The tests check the code you were *given* and the contract your
probes must keep (return a list of `Finding`s with a valid severity). They do not check your
thresholds — those are yours to defend.

## 3. Probe 1 — Data quality

> [!IMPORTANT]
> **🛑 YOUR TURN — write this before reading on**
>
> **File:** `monitor/probes.py` · **Function:** `check_data_quality(win, profile)`
>
> It currently says `return []`. That is why Step 1 printed all-green.
>
> **The job:** for each feature in `profile["features"]`, compare its null rate in `win`
> against the `null_rate` the training profile recorded, and return a `Finding` when it is
> meaningfully higher. The full TODO is in the docstring in the file — read that, not this.
>
> **The two decisions that *are* the exercise:** what counts as "meaningfully higher", and
> when is it `warn` versus `critical`. There is no single right answer and the reference
> solution is not the only defensible one.
>
> **🤔 Predict before you run:** three features, one week of traffic. How many findings do
> you expect? Write the number down.
>
> **Then run:**
> ```bash
> python -m monitor.cli check
> pytest tests/test_probes_contract.py
> ```
> You are done with this step when the report contains exactly one finding and the contract
> tests pass. Then open the toggle below and compare.

<details>
<summary><b>▶ What you should see</b> (open after your probe fires)</summary>

```
!  [Data Quality     ] brightness    null rate 4.8% (training baseline 0.4%)
```

If you got more than one finding, your threshold is too tight — check what it's saying about
features that were *already* null in training. If you got none, it's too loose, or you're
comparing against the wrong number.

</details>

One feature is going null twelve times more often than it did in training. Nothing has
errored. No exception was ever raised, no request failed, and the model happily produced a
prediction for every one of those rows.

> [!TIP]
> **Why a multiple of the baseline, not a flat number?** A feature that was 0.4% null in
> training and is 4.8% null now has changed by more than a factor of ten. A feature that was
> always 30% null sitting at 32% has not changed at all. A single global threshold like
> "alert above 5% null" would have stayed silent here *and* screamed forever about the second
> feature. Thresholds relative to a recorded baseline are why the training profile exists.

> [!TIP]
> **Warn or critical?** One rule, and it is not statistical: *critical is what you would wake
> a human for at 2am.* If you wouldn't get out of bed, it's a warn. The lecture's DECIDE step
> is exactly this — pre-commit the threshold, the action and the owner — and teams that skip
> it end up with everything critical, which is identical to nothing critical.

**🤔 Before moving on:** is a 4.8% null rate on one feature enough to move accuracy by 24
points? Do the arithmetic in your head. It is one finding, not the answer.

## 4. Probe 2 — Feature drift

A quick word on the two statistics you are about to use, both already written in `core.py`:

- **KS test** (`ks_statistic`), for continuous columns. Stack the two samples' cumulative
  distributions on top of each other; `D` is the biggest vertical gap between them. 0 means
  identical, 1 means no overlap. `ks_critical(n, m)` tells you the largest `D` that chance
  alone would produce at your sample sizes.
- **PSI** (`psi`), for categorical columns. KS needs an ordering ("less than") and `region`
  has none, so PSI compares the category *mix* instead. Rule of thumb: above 0.25 is a major
  shift.

> [!IMPORTANT]
> **🛑 YOUR TURN**
>
> **File:** `monitor/probes.py` · **Function:** `check_feature_drift(win, golden, profile)`
>
> **The job:** `golden` is the reference distribution, `win` is now. Loop the `CONTINUOUS`
> columns through `ks_statistic()` and compare `D` against `ks_critical()`; loop the
> `CATEGORICAL` ones through `psi()`. Use `LAYER[col]` for each finding's layer.
>
> **You are not writing statistics, you are writing judgment.** The KS test is one line of
> scipy and it's already done. What's yours: deciding when a significant result is
> *actionable*.
>
> **🤔 Predict before you run:** eight columns are tested. How many will clear the critical
> value? Which one do you think will move the most?
>
> **Then run:**
> ```bash
> python -m monitor.cli check
> ```
> Expect around five findings. Before you open the toggle: three of them are traps, so read
> your own output first and see if you can spot which.

<details>
<summary><b>▶ What you should see</b> (open after your probe fires)</summary>

```
!! [Data Quality     ] blur_score      KS D=0.263 (D_crit=0.059, p=1.6e-47)
!! [Model Performance] confidence      KS D=0.552 (D_crit=0.059, p=2.3e-218)
!! [System Health    ] latency_ms      KS D=0.320 (D_crit=0.059, p=1.7e-70)
!! [Data Quality     ] image_source    PSI=2.424 (>0.25 is a major shift in the traffic mix)
!! [Data Quality     ] region          PSI=1.198 (>0.25 is a major shift in the traffic mix)
```

Getting a dozen findings instead of five? You're firing on statistical significance alone —
see the third tip below. Getting `model_version` as a drift finding? Check what's in
`CATEGORICAL` versus `SEGMENTS` in `core.py`, and why.

</details>

Five alarms. Read them carefully, because three of them are traps.

> [!TIP]
> **`latency_ms` drifted — and got *faster*.** D=0.32 with a p-value of 1.7e-70 is a
> screaming statistical result about an unambiguously *good* change. A KS test detects
> difference; it has no idea which direction is bad. If you page someone on "significant
> drift" alone, this is the alert that teaches your team to ignore your alerts. Direction
> and impact are your job, not the test's.

> [!TIP]
> **`confidence` is the model's output, not its input.** It is in `CONTINUOUS` deliberately.
> When input features look stable and the output distribution moves, the world hasn't
> changed shape — the model's *relationship* to the world has. That is concept drift, and
> this is the cheapest way to see it: you already log confidence, and it needs no ground
> truth at all. Hold that thought until Step 7.

> [!TIP]
> **`p=2.3e-218` is not 218 times more urgent than `p=0.04`.** With ~950 events per window
> against a 4,400-event reference, `D_crit` is 0.059 — so a 6% distributional wobble is
> "significant". Significance gets cheaper the more traffic you have, which is why your
> probe wants a floor on `D` as well as the critical value. And it is why the lecture's
> alert rules all said *"for 10 minutes"* or *"for 30 minutes"*: run 15 tests a day and you
> get a false positive most weeks by chance. `replay` in Step 8 requires two consecutive
> days for exactly this reason.

**What you know so far:** the traffic mix changed, one image feature degraded, and the model
is less sure of itself. What you still don't know is whether any of that explains a 24-point
accuracy drop. Two of these five alarms have nothing to do with it.

## 5. Probe 3 — Serving parity

This one is not a drift test, and the distinction is the most useful thing in this tutorial.

> [!IMPORTANT]
> **🛑 YOUR TURN**
>
> **File:** `monitor/probes.py` · **Function:** `check_serving_parity(win, profile)`
>
> **The job:** two comparisons, both against `reference/training_profile.json` rather than
> against any window of production.
> 1. `win["resize_px"]` versus `profile["preprocessing"]["resize_px"]`. Any traffic served at
>    a size the model wasn't trained at is a bug — `critical` even at small volume.
> 2. Each feature's mean in `win` versus the training mean in `profile["features"]`, measured
>    in units of the training std.
>
> **🤔 Predict before you run:** you already ran a drift test over these same features in
> Step 4. Do you expect this probe to find anything *new*? Commit to yes or no.
>
> **Then run:**
> ```bash
> python -m monitor.cli check
> ```

<details>
<summary><b>▶ What you should see</b> (open after your probe fires)</summary>

```
!! [Model Performance] preprocessing.resize_px   serving at 224px, model trained at 256px (100% of traffic)
!  [Data Quality     ] blur_score                mean 0.640 is 1.0σ from the training mean 0.720
```

</details>

**Every single request is being preprocessed differently from how the model was trained.**

Now go back and look: **no drift test could have found this.** The image features are
unchanged, because the images are unchanged. The *pipeline* changed. A KS test compares
production against production; it is structurally blind to production having drifted away
from *training*. Nothing in three months of logs contains the number 256. That number exists
in exactly one place — the training profile — and without that file this bug is invisible
forever.

This is **training-serving skew**: not the world changing, but *you* changing something and
the model not being told. It is one of the most common causes of production model failures,
and the reason feature stores exist — one feature definition, computed once, used by both
the training and the serving path, so the two cannot disagree.

> [!TIP]
> **Every value passed its contract.** Remember the lecture's point about data validation:
> every field in the fraud model's batch was present and in range, and the system was still
> broken, because the failure lived in what the values *meant*. Same here. `resize_px=224` is
> a perfectly valid integer. A schema check would wave it through. Only a check against
> *what the model expects* catches it — which is the lecture's line about the edge probes:
> **no probe substitutes for another.**

> [!TIP]
> **Find the day it started.** `--as-of` re-runs any check as it would have looked on a past
> date:
> ```bash
> python -m monitor.cli check --as-of 2026-09-24   # no resize_px finding
> python -m monitor.cli check --as-of 2026-09-26   # serving at 224px (32% of traffic)
> ```
> Narrow it down. Then look at which `model_version` the log carries either side of that
> date. A parity break with a hard start date and a version boundary is a deploy, not a
> drift — and it means someone's change is sitting in a commit you can read.
>
> The share is worth noticing too: 32% of traffic, climbing to 100%. Your probe caught this
> *during* the rollout — which is the whole argument for a canary release from the lecture:
> a small real cohort, pre-committed stop rules, fast rollback. But only because it alerts
> on *any* nonzero share. A threshold like "fire above 50% of traffic" would have politely
> waited for the rollout to finish before mentioning it.

> [!TIP]
> **Notice what the system-health dashboard said that week.** The same deploy made p50
> latency drop by about 20%, because smaller images are faster to process. On a standard
> latency-and-error-rate dashboard, that deploy looked like a straight win, and it stayed
> looking like a win for a month, because the labels that would have contradicted it had not
> arrived yet. Somebody probably got congratulated for it.

## 6. Probe 4 — Model performance

> [!IMPORTANT]
> **🛑 YOUR TURN — last one**
>
> **File:** `monitor/probes.py` · **Function:** `check_performance(labeled, profile)`
>
> **The job:** overall accuracy first, then per segment. This is the lecture's LOCALIZE step —
> geography, then merchant age — done in code.
> 1. `accuracy(labeled)` against `profile["offline_metrics"]["accuracy"]`.
> 2. Then again per segment: `segment_table(labeled, by)` for each `by` in `SEGMENTS`, firing
>    on the worst slice. Careful what you compare the slice *against* — the docstring warns
>    you about the trap, and it's a real one.
> 3. Handle the empty case. Early in a window no labels have arrived at all, and an `info`
>    finding saying so is more honest than green.
> 4. Stretch, and it's the best finding in the tutorial: what does it mean if
>    `labeled["confirmed_label"]` contains something that isn't in `profile["label_space"]`?
>
> `labeled` only holds events whose label had *already arrived* — see `labels_known_by()`
> from Step 2. You are grading yourself on a third of your traffic, eleven days stale.
>
> **🤔 Predict before you run:** which segment do you expect to be worst, and by how much?
>
> **Then run:**
> ```bash
> python -m monitor.cli check
> pytest
> ```
> That's all four probes live, and the whole suite should be green. This is the full picture
> of the incident.

<details>
<summary><b>▶ What you should see</b> (open after your probe fires)</summary>

```
!! [Data Quality     ] label_space    4.6% of confirmed labels are outside the training label space (danablu, gammelost, herrgard, vasterbotten)
!! [Model Performance] accuracy (all traffic)        67.8% on confirmed labels vs 93.0% offline (25.2 pts below) — n=717
!! [Model Performance] image_source=partner_uploader accuracy 52.8% vs 67.8% overall (n=231)
!! [Model Performance] region=nordics                accuracy 35.5% vs 67.8% overall (n=62)
```

One near-duplicate finding per region (`region=eu`, `region=na`) means you compared each
slice against the offline number instead of against overall — see the tip below.

</details>

Now compare the aggregate with the slices:

```bash
python -m monitor.cli segments --by image_source
python -m monitor.cli segments --by region
```

```
accuracy by image_source — last 28d as of 2026-11-02
overall: 67.8%  (n=717)

  partner_uploader   n=  231   52.8%  █████████████████████
  app                n=  486   74.9%  █████████████████████████████
```

"Accuracy is 68%" is a fact about an average. It is true of no actual user. One traffic
source is at 53% and one region is at 36%, and the aggregate number is the mean of a healthy
majority and a broken minority — which is precisely why it looked like a slow, mysterious,
unattributable decline instead of three specific, fixable bugs. *Aggregates hide localized
failures.* You saw it with Brazil; this is the same slide with cheese.

> [!TIP]
> **Why compare the slice to overall and not to the offline number?** Try it the other way.
> Once global accuracy has fallen, *every* segment is below the offline number, so you get
> one near-duplicate finding per segment and learn nothing. Comparing to overall answers the
> question you actually have: **which slice is dragging the average down?**

> [!TIP]
> **The label-space finding is a different kind of failure.** Four confirmed labels are
> varieties the model was never trained to emit. It is not miscalibrated on those images —
> it *cannot* be right about them. There is no output neuron. Retraining on the existing
> classes, at any cost, on any hardware, would not move this number. The fix is a label
> space and a dataset, not a model. A pure accuracy dashboard would have shown you a
> degraded model and sent you straight to retrain — the most expensive possible non-fix.

> [!TIP]
> **Segment by the dimensions your system can actually change.** `image_source`, `region`,
> and `model_version` each correspond to something a human decided: a partner integration, a
> market launch, a deploy. Segmenting by 40 arbitrary columns generates suspects; segmenting
> by your own decisions generates causes.

## 7. The signal that needs no labels

Everything in Step 6 depended on confirmed labels — which cover 30% of traffic and arrive 11
days late. So the newest failure is, by construction, the one you can see least. What do you
do about the thing that broke yesterday?

```bash
python -m monitor.cli confidence --by region --days 42
```

```
(abbreviated — the real output gives every week for every region)

  region=eu
    2026-10-12  n= 370  mean=0.717  low=33.2%  ████████████████████████████
  region=na
    2026-10-12  n= 491  mean=0.732  low=28.1%  █████████████████████████████
  region=nordics
    2026-10-12  n= 106  mean=0.663  low=50.9%  ██████████████████████████
    2026-10-19  n= 164  mean=0.646  low=51.8%  █████████████████████████
```

One region's predictions are low-confidence **half the time**, against ~30% everywhere else.
No labels were involved in producing that table. It was visible within days of that region
launching — and Step 8 will show you that segmented accuracy could not confirm it for three
more weeks.

This is confidence-based performance estimation (NannyML's CBPE is the best-known tool for
it), stripped to about fifteen lines of pandas: use the model's own confidence distribution
as an early proxy for performance instead of waiting for ground truth.

> [!TIP]
> **A cheap secondary signal.** In the edge-camera incident, "page on confidence drift" was
> option C — and the lecture's verdict was that it *can be a cheap secondary signal but
> cannot prove anything on its own.* Same verdict here. Confidence is a proxy, and a badly
> calibrated model gives you a confident, wrong proxy. It is a trigger to investigate, never
> a number to report. The defensible use is exactly what you just did: compare segments
> against each other at the same moment, where miscalibration is at least a shared bias.

## 8. Replay: how early could you have known?

The payoff — and the lecture's VERIFY step turned around: not "did the fix work?" but "would
the monitor have worked?" Walk all 90 days through your probes and record when each would
first have alerted, requiring two consecutive days before it counts, so a single lucky
p-value doesn't page anyone.

```bash
python -m monitor.cli replay
```

> [!NOTE]
> This table is generated by **your four probes**, not by anything pre-baked. If it comes out
> empty, you have unimplemented probes — go back to Steps 3–6. If your dates differ from the
> ones below by a day or two, nothing is wrong: your thresholds are your own, and a tighter
> threshold buys you a later, more trustworthy alert. That trade *is* the lesson. Add
> `--solution` to see what the reference thresholds produce.

```
replay — 7d window, alert after 2 consecutive days

  first alert    days early  probe / subject
  -------------- ----------  --------------------------------------------
  2026-09-05          59d   feature_drift / image_source
  2026-09-07          57d   feature_drift / blur_score
  2026-09-08          56d   data_quality / brightness
  2026-09-08          56d   serving_parity / blur_score
  2026-09-19          45d   performance / image_source=partner_uploader
  2026-09-25          39d   feature_drift / confidence
  2026-09-25          39d   serving_parity / preprocessing.resize_px
  2026-09-26          38d   feature_drift / latency_ms
  2026-09-26          38d   performance / accuracy (all traffic)
  2026-10-06          28d   performance / model_version=v3.2
  2026-10-13          21d   feature_drift / region
  2026-10-21          13d   performance / label_space
  2026-10-30           4d   performance / region=nordics

  the humans noticed on 2026-11-03.
```

Read the timeline, not the list. It was never one bug. It was **three**, overlapping — each
first visible to a different probe, each **one day** after it began:

| Started | What began | First alert | Which probe caught it |
|---|---|---|---|
| 4 Sep | a partner's uploader started sending recompressed, EXIF-stripped images | 5 Sep | drift, then data quality and parity |
| 24 Sep | `v3.2` shipped serving at 224px against a model trained at 256px | 25 Sep | **parity** (and output drift, same day) |
| 8 Oct | a new market brought varieties outside the label space | 13 Oct | **categorical drift, then label space** |

Note `performance / model_version=v3.2` on 6 October: accuracy sliced by release, catching
the bad deploy on ground truth alone — 11 days after parity had already named the exact
cause, and 28 days before anyone looked.

And the lagging signal, for contrast: `performance / region=nordics` — plain segmented
accuracy on the newest failure — fired on **30 October, four days** before the humans
noticed anyway. Ground truth is the most trustworthy signal you have and the last one to
arrive. A monitor built only on accuracy is a monitor that is always three weeks late.

> [!TIP]
> **Three failures overlapped, and the aggregate hid all three.** That is the realistic
> part. The instinct to find *the* root cause is what kept this incident open for two
> months: the partner-uploader alert on 5 September was investigated, explained, filed as
> "known, low volume" — and it was still open when the deploy broke on 24 September and the
> market launched on 8 October, so every later signal looked like more of the same known
> issue. Alert fatigue is not caused by too many alerts. It is caused by alerts nobody
> closes.

**Now score your Step 0 guess.** Most people name exactly one of the three. Which one you
guessed says more about your priors than about the system.

Then read [`answers/make_production_log.py`](answers/make_production_log.py) — the generator
that planted all three, with dates and magnitudes in the source. It is the ground truth for
your write-up, and now is the right time to read it.

### Fill this in — from your own evidence

| Failure | Caught by | Blind to it | Days early |
|---|---|---|---|
| Partner uploader / degraded images | ? | ? | ? |
| Preprocessing skew in `v3.2` | ? | ? | ? |
| New market outside the label space | ? | ? | ? |

The middle column is the one worth arguing about. For each failure, name a probe that stayed
green — and say *why* it structurally could not have fired. "No probe substitutes for
another" is a sentence on a slide until you can point at the row where it happened.

## 9. Wire it into your own project

Milestone 3 asks for production monitoring. This is the shape of it.

**The four probes are not the deliverable — the loop is.** Every probe you wrote takes a
window of logged prediction events, compares it to a stored reference, and returns findings.
That works because the system logs the right things in the first place. Copy that part:

1. **Log one row per prediction**, not per request: inputs (or their summary statistics),
   the prediction, the confidence, the model version, the serving config, and an ID you can
   join a later label onto. *The join keys make evaluation possible.*
2. **Store a reference** the day you deploy. `reference/training_profile.json` is 20 lines
   of JSON and it is the only reason Step 5 was possible. A model shipped without a recorded
   training profile cannot be checked for skew, ever.
3. **Write labels back asynchronously**, with the timestamp at which they *arrived* — not
   just the timestamp of the prediction. This is the field most teams skip. Without it every
   backtest you run is cheating under the hood.
4. **Run the probes on a schedule** and require consecutive windows before alerting.
5. **Route alerts to a human with a runbook**, not to an automatic rollback. Pre-commit the
   threshold, the action, the owner and the scope — that is DECIDE.

The lecture drew two paths. This repo has both, in miniature:

| Path | In the lecture | In this repo | GCP | AWS |
|---|---|---|---|---|
| **Fast** — detect and contain | metrics → alert → rollback, in minutes | `data_quality`, `feature_drift`, `serving_parity`, `confidence` | Cloud Logging → BigQuery; Cloud Run Job + Cloud Scheduler; Cloud Monitoring alert policy | CloudWatch Logs → S3 + Athena; Lambda + EventBridge; CloudWatch Alarm + SNS |
| **Delayed** — establish truth | mature outcomes → warehouse join → release evidence, in days | `labels_known_by` + `check_performance` + `segments` | BigQuery scheduled query | Athena + Step Functions |
| **Managed** — all of the above | | | Vertex AI Model Monitoring | SageMaker Model Monitor |

> [!TIP]
> **You would not hand-roll this at work, and you should still have written it once.** In
> production, reach for Evidently AI, NannyML, TensorFlow Data Validation, whylogs, or your
> cloud's managed monitor. What none of them will do for you is decide which reference to
> compare against, which segments matter, what threshold is worth waking someone for, and
> which failure modes your system has that no distribution test can see. That judgment is
> the part of this tutorial that transfers — the KS call is one line.

### What this repo simplifies

Worth knowing, so you don't ship these assumptions:

- **Features are pre-computed in the log.** Real drift monitoring reads from a feature store
  or recomputes from raw inputs, and that recomputation is itself a common skew bug.
- **One model, one endpoint.** With shadow or canary traffic every probe needs to be
  segmented by variant, or you average the champion and the challenger together and learn
  nothing about either.
- **Labels are correct.** User-confirmed labels are noisy and self-selecting: users who
  bother to correct the model are not a random sample of users. This is the lecture's
  feedback path — *what becomes future truth?* — and it needs provenance.
- **Retraining is out of scope.** A retraining policy — a threshold sustained over
  consecutive windows, a champion/challenger validation before promotion — is the natural
  next step once your probes are trustworthy.

## Troubleshooting

| Symptom | Cause |
|---|---|
| `(all probes green)` after you wrote a probe | Your probe returns `[]` on every path, or you edited `answers/probes_solution.py` instead of `monitor/probes.py`. |
| Edits in VS Code don't show up inside the container | You started the container without the bind mount. Exit, and start it with `./docker-shell.sh` from the repo root. |
| `ModuleNotFoundError: monitor` | Run from `/app` (the repo root), as `python -m monitor.cli`, not `python monitor/cli.py`. |
| `python: command not found` or wrong Python inside the container | The venv isn't active. Run `source /home/app/.venv/bin/activate`. |
| Accuracy is `n/a` or n is tiny | You are inside the label delay. Widen `--perf-window`, or pick a later `--as-of`. |
| Every feature drifts on every day | You are comparing the window against itself, or against `win` instead of `golden`. |
| `replay` takes more than ~30s | You are recomputing something expensive inside a loop over features. |
| `docker: command not found` / Docker not running | Start Docker Desktop, or use the `uv` fallback in Step 1 and carry on. |
| `the input device is not a TTY` (Windows) | Run from Git Bash and prefix with `winpty`. |

**The Step 1 question:** the venv lives at `/home/app/.venv` because `/app` is covered by the
bind mount while the container runs. If the venv were at `/app/.venv`, it would be hidden
by whatever is in your repo folder on the host — an empty directory, or a macOS venv that
Linux cannot run. Putting it outside `/app` means you can mount your code over `/app` and
the installed packages are still there.

Stuck on a probe? `--solution` runs the reference implementation so you can compare outputs
without reading its source:

```bash
python -m monitor.cli check --solution
```
