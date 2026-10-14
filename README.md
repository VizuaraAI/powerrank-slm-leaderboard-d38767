# powerrank-slm-leaderboard

**The first properly powered SLM leaderboard.**

Detecting a one-point accuracy difference between two language models requires
roughly 13,000 to 17,000 benchmark items. Most published comparisons use one to
two thousand. The field's standard defence is that running more is too
expensive, and for small language models that defence does not hold: the full
design in this repo is about 5.2M forward passes and fits on a single consumer
GPU.

This repository is the code for that argument. It builds a pooled item bank,
runs a grid of SLMs over it under several prompt templates, keeps the
**item-level** responses rather than a single score per model, and turns them
into a leaderboard that reports rank *bands* and a minimum detectable effect
instead of false precision.

Hi Sudharshan. This is your project scaffold. Everything here runs, but nothing
here is your result yet. Start with the one command below.

---

## Thirty seconds in

You do not need a GPU, a download, or even the model dependencies to see the
point of this project:

```bash
pip install -r requirements.txt     # or: pip install -e .
powerrank power --delta 1.0 --discordance 0.20
```

```
  paired (McNemar)          15,696 items
  unpaired                  37,675 items per model (75,350 total)

  at n =  1,000  MDE =  3.96 pt   CANNOT detect a 1.0 pt gap
  at n =  2,000  MDE =  2.80 pt   CANNOT detect a 1.0 pt gap
  at n = 17,248  MDE =  0.95 pt   can detect a 1.0 pt gap
```

That output is the paper. Everything else is the work of earning the right to
put real numbers in it.

---

## Setup

```bash
git clone <your-repo-url> powerrank-slm-leaderboard
cd powerrank-slm-leaderboard

python -m venv .venv && source .venv/bin/activate
pip install -e ".[eval,plots,dev]"      # or: pip install -r requirements.txt

pytest -q                               # 38 tests, about 2 seconds
```

Python 3.10 or newer. The `eval` extra pulls in torch, transformers and
datasets; skip it if you only want the statistics.

Datasets download from the Hugging Face Hub on first use and cache under
`HF_HOME`. No data files ship with this repo. Some models (Llama, Gemma) are
gated, so run `huggingface-cli login` before the full sweep.

---

## The first three things to run

**1. The power calculator** (no GPU, one second)

```bash
bash scripts/00_power_check.sh
```

Shows how the required item count moves with the effect size and the
discordance rate. Play with `--discordance`: notice how much the answer depends
on it, which is exactly why this project measures it rather than assuming it.

**2. The smoke run** (CPU, a few minutes)

```bash
bash scripts/01_smoke_run.sh
```

Two small models, 60 items, one prompt, end to end: bank, evaluate,
leaderboard. The result will be a single rank band with everything tied, and
that is the *correct* answer at n=60, where the MDE is around 15 points. If you
see a significant difference here, you have a bug, not a finding.

**3. The real evaluation** (GPU, plan for about a day)

```bash
bash scripts/02_full_eval.sh
```

Builds the full pool and sweeps the model zoo. Shards are written per
(model, prompt, seed) and skipped on resume, so it is safe to interrupt.

---

## Layout

```
src/powerrank/
  power.py        the statistics: required N, MDE, McNemar, bootstrap, BH
  variance.py     where the variance comes from: item, prompt, seed, model
  itembank.py     pool and de-duplicate public benchmarks into one item bank
  models.py       the SLM zoo, prompt templates, log-likelihood scoring
  evaluate.py     run the grid, write the item-level response tensor
  leaderboard.py  pairwise tests, significant ranks, rank bands
  audit.py        retrospective power audit of published claims
  cli.py          the `powerrank` command
configs/default.yaml    defaults, including the discordance rate to replace
scripts/                00 power, 01 smoke, 02 full eval, 03 audit
experiments/            log_template.md, claims_template.csv
tests/                  38 tests pinning the headline numbers
```

**Read `power.py` first.** It is the contribution; everything else feeds it.

---

## The one idea to hold on to

A leaderboard stores one number per model and throws away which items each
model got right. That discarded information is exactly what a paired test
needs. So the artifact this project produces is not a score table, it is the
response tensor:

| model | item_id | prompt_id | seed | correct |
| --- | --- | --- | --- | --- |
| Qwen2.5-3B | mmlu_pro-a1b2c3 | plain | 0 | 1 |

About 60MB as parquet for the full run. Once you have it, any pairwise
question can be answered offline in milliseconds, forever, without touching a
GPU again. That is also why releasing it is a contribution in its own right:
nobody else can recompute a paired test from a published score table.

---

## Known gaps, which are your first tasks

These are deliberate. The scaffold is honest about where it stops.

- **MATH-500 and IFEval loaders are not implemented.** The paper's pool lists
  eight sources; `itembank.py` implements six. Both missing ones need
  answer-equivalence checking rather than string equality. Writing that grader
  forces you to decide what "correct" means, which is a good week-2 problem.
- **The discordance rate in `configs/default.yaml` is assumed, not measured.**
  0.20 is a plausible placeholder. Replacing it with your measured value is the
  single highest-value thing you will do, because every sample size in the
  paper depends on it.
- **The variance decomposition is method-of-moments**, not a full crossed
  mixed-effects fit. Transparent and fast, but it will not give you credible
  intervals on the components themselves. `fit_glmm_statsmodels` is the first
  step of the upgrade path; `pymc` or `lme4` is the destination.
- **`experiments/claims_template.csv` contains no real data.** The two rows are
  invented examples so the audit runs. The audit means nothing until you
  transcribe real claims with real citations.
- **Use an odd number of prompt templates.** The paired test takes a majority
  vote across templates; an even number can split exactly and has no majority.
  The code warns you, the default config uses three.

---

## Citation and license

If this becomes a paper, cite the venue. Until then the artifact is the
response tensor and the power calculator. Released under the MIT license;
the pooled benchmarks retain their original licenses.
