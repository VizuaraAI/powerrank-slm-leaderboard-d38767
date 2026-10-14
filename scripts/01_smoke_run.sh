#!/usr/bin/env bash
# A complete end-to-end pass on a tiny slice: bank -> evaluate -> leaderboard.
# Two small models, 60 items, one prompt. Runs on CPU in a few minutes.
# The point is to prove the pipeline works, NOT to produce a real result:
# at n=60 the MDE is about 15 points, so expect every model to be tied.
set -euo pipefail

powerrank bank --out data/bank_smoke.parquet --max-per-source 10

powerrank evaluate \
  --bank data/bank_smoke.parquet \
  --out runs/smoke \
  --models SmolLM2-360M-Instruct Qwen2.5-0.5B-Instruct \
  --templates plain \
  --limit 60

powerrank board --runs runs/smoke --out results/smoke
