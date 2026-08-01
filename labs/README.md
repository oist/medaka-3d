# labs/ — experiment quarantine

Everything speculative lives here. Nothing here is production.

The retired repo did not fail because it ran too many experiments. It failed
because experiments **never ended**. 53 numbered scripts in one directory, three
parallel benchmark trees, no verdicts — so nothing could be deleted, because
nobody could tell what still mattered. The mess was undead experiments, not
experimentation.

These rules exist so that cannot happen again. They are enforced by
`tests/test_labs.py`, not by discipline.

## Rules

**1. One hypothesis per lab.** Directory name `YYYYMMDD_slug`.

**2. Every lab has a `README.md` with front matter:**

```markdown
---
hypothesis: SAM2 video propagation survives more encounters than NN linking.
stage: linker
expires: 2026-09-15
decision_rule: >
  Promote if encounter survival exceeds the production linker by >= 10 points
  on the frozen benchmark with no regression in abstention rate.
---
```

`decision_rule` is written **before** running anything. A rule written after
seeing results is not a rule, it is a rationalisation — and the retired repo's
one properly pre-registered audit is the only evaluation in it that still
stands.

**3. Every lab implements a stage Protocol** from `medaka3d.stages`, so the
shared benchmark harness runs it unmodified. No lab writes its own evaluation
code. Incomparable numbers are how the old repo settled method choice by
argument instead of measurement.

**4. `labs/` may import `medaka3d`. `medaka3d` may never import `labs/`.**
Enforced by `tests/test_labs.py::test_production_does_not_import_labs`. This is
the wall that stops experimental code leaking into the production line.

**5. Every lab expires.** Past `expires:` with no `RESULT.md`, the build fails.
You then either write the verdict or delete the directory. Both are fine;
leaving it is not.

**6. Every lab ends in a `RESULT.md`** with a verdict:

- `promoted` — implementation moves into `src/medaka3d/`, `RESULT.md` moves to
  `docs/decisions/`, **the lab directory is deleted**.
- `rejected` — `RESULT.md` records what was tried and why it failed, then the
  lab directory is deleted. The finding survives in `docs/decisions/`; the code
  does not.
- `parked` — blocked on something external. Must name the blocker and carry a
  new `expires:`. Parking twice without progress means rejected.

Deleting the code is the point. Git remembers it; the working tree should not.

## Getting started

```bash
cp -r labs/_template labs/$(date +%Y%m%d)_my_idea
```

Then edit the front matter, implement the Protocol, and register the component.
Run it against the benchmark with the shared harness — never with your own
scoring code.
