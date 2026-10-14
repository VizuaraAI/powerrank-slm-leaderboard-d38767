# CLAUDE.md

Instructions for the coding agent working on this repository with Sudharshan.

## What this project is

A statistically powered leaderboard for small language models. The thesis in
one line: **detecting a one-point accuracy difference requires 13,000 to 17,000
paired benchmark items, most published comparisons use 1,000 to 2,000, and for
SLMs the usual cost objection does not apply.**

The deliverables are a NeurIPS-workshop-length paper, a released item-level
response tensor, and a leaderboard that reports rank bands instead of integer
ranks. Timeline is eight weeks of self-paced work.

## The rule that matters most

**Never let an aggregate score be the stored artifact.** The whole project
exists because leaderboards discard which items each model got right, and that
is precisely what a paired test needs. If you are about to write code that
averages before saving, stop: save the per-item rows and average at read time.

Second rule: **the discordance rate is measured, not assumed.** `0.20` appears
in `configs/default.yaml` as a placeholder. Any analysis that silently uses the
placeholder when measured data exists is a bug. Every sample size in the paper
is proportional to it.

## Layout

```
src/powerrank/
  power.py        REQUIRED N, MDE, McNemar, paired bootstrap, Benjamini-Hochberg
  variance.py     variance components: item, prompt, seed, model
  itembank.py     pool + de-duplicate public benchmarks
  models.py       SLM zoo, prompt templates, log-likelihood and generative scoring
  evaluate.py     run the grid, write the response tensor as parquet shards
  leaderboard.py  pairwise tests, significant ranks, rank bands
  audit.py        retrospective power audit of published claims
  cli.py          argparse entry points, one subcommand per stage
configs/default.yaml
scripts/          00_power_check 01_smoke_run 02_full_eval 03_audit
experiments/      log_template.md, claims_template.csv
tests/test_power.py
```

Data flows strictly one way:

```
itembank -> evaluate -> (response tensor) -> variance -> power -> leaderboard
                                                              \-> audit
```

`power.py` imports nothing from the rest of the package. Keep it that way: it
is the part that must stay correct and independently testable.

## Conventions

- Python 3.10+, type hints on public functions, `from __future__ import annotations`.
- Numpy/pandas, not loops, for anything per-item.
- Deltas and accuracies are **proportions** (0.01 for one point) everywhere
  inside the library. Convert to percentages only for display. `power.py`
  raises if you pass something `>= 1.0`, which catches the usual mistake.
- Every function that takes a rate validates its range and raises `ValueError`
  with the offending value in the message.
- Heavy imports (`torch`, `transformers`, `datasets`, `statsmodels`) are
  **lazy**, imported inside the function that needs them, so the statistics
  path stays importable without a GPU stack.
- `logger = logging.getLogger(__name__)` per module. No bare `print` outside
  `cli.py`.
- Docstrings explain *why*, including the statistical reasoning. This repo is
  read by one person learning the method; a bare type signature does not help
  them.

## Testing

```bash
pytest -q          # 38 tests, ~2s, no GPU, no network
```

`tests/test_power.py` pins the headline numbers deliberately. If a change moves
`required_n_mcnemar(0.01, 0.20)` outside 13,000 to 17,000, that is not a failing
test to relax, it is the paper's central claim changing, and it needs a
conversation first.

When you fix a statistical bug, add a regression test naming the wrong output.
See `test_reported_interval_brackets_the_reported_accuracy`, which pins a real
bug where a leaderboard printed 62.6% next to a CI of [83.7, 85.9].

## Working on this

- Run `pytest -q` before and after every change. It is two seconds.
- Before adding a model to `MODEL_ZOO`, confirm it fits in 16GB at bfloat16 and
  that its Hub id is not gated, or note the gating in the README.
- Before adding a benchmark to `DEFAULT_SOURCES`, add a `_parse_row` branch and
  check the answer key convention; the per-source column names differ and
  getting the letter mapping wrong silently produces chance accuracy.
- Chance-level accuracy for a model that should be competent almost always
  means a prompt-format or answer-key bug, not a weak model. Check by printing
  three rendered prompts and three predictions.
- Long runs: always via `scripts/02_full_eval.sh`, always resumable. Never
  re-run a completed shard to "be safe"; it costs hours.

## Where the student is going

Eight weeks, ending in a workshop submission. The sequence:

1. Power calculator and the literature. Fills the paper's Related Work.
2. Item bank built and de-duplicated, MATH-500 and IFEval loaders added.
3. Smoke run end to end, then the first real multi-model run.
4. Measure the discordance rate and replace the placeholder. This is the hinge.
5. Variance decomposition and the prompt-count ablation.
6. Full sweep, rank bands, the main results table.
7. Retrospective audit with real transcribed claims.
8. Write up and submit.

When Sudharshan asks for "the next thing to do", prefer whatever most reduces
uncertainty in the discordance rate, because every other number depends on it.

## Things not to do

- Do not add a judge model or an LLM grader. The pool is deliberately
  restricted to deterministically scorable items so the tensor is reproducible.
- Do not commit parquet files, model weights or datasets. `.gitignore` covers
  these; check before committing.
- Do not "clean up" `power.py` into something clever. It is intentionally
  explicit, with the formulas written out, because its readability is the
  reason a reviewer will believe it.
- Do not report a single headline percentage for the audit. It depends on an
  assumed discordance rate, so it is reported as a sensitivity curve
  (`sensitivity_curve`). Resist the pull toward one dramatic number.
- Do not invent citations or claims for the audit. Every row needs a real
  source someone else can check.
