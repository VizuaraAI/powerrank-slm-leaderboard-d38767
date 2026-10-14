#!/usr/bin/env bash
# The real run. Build the full pool, then sweep the model zoo.
#
# Budget before you start it: 10 models x 17k items x 3 prompts is about
# 520k scored items. On a single consumer GPU with log-likelihood scoring,
# plan for roughly a day. Shards are written per (model, prompt, seed) and
# skipped on resume, so it is safe to stop and restart.
set -euo pipefail

powerrank bank --out data/bank.parquet

powerrank evaluate \
  --bank data/bank.parquet \
  --out runs/full \
  --templates plain instructed qa_pair \
  --seeds 0

powerrank board --runs runs/full --out results/full
