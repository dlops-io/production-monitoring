# ⚠️ Spoilers

Two files in here will ruin the tutorial if you read them at the wrong time.

### `probes_solution.py`

A reference implementation of all four probes. There is no single correct set of thresholds —
this is one defensible set, with the reasoning in the comments. Compare it with yours after
you have something firing.

You can run against it without reading it, which is the better way to get unstuck:

```bash
python -m monitor.cli check --solution
python -m monitor.cli replay --solution
```

If your probe finds something this one misses, you have not made a mistake. Read the
`Finding` you produced and decide whether you would want to be woken up for it.

### `make_production_log.py`

The generator that produced `data/production_log.csv`. It names all three planted failures,
their start dates, and their exact magnitudes.

**Read this at Step 8, not before.** It is the answer key to the whole incident, and the
tutorial only works if you find the failures in the data first. Once you get there it is the
best thing in the repo — every effect you measured, written as the four or five lines of code
that caused it.

To regenerate the data (deterministic, `SEED = 215`, so you will get a byte-identical CSV):

```bash
python answers/make_production_log.py
```

`tests/test_generator.py` regenerates it into a temporary folder and checks that it matches
the committed files byte for byte, so nobody can drift the incident by accident.
