#!/usr/bin/env bash
# The retrospective power audit.
#
# Before this is meaningful you must fill experiments/claims_template.csv with
# real claims transcribed from technical reports and leaderboard snapshots.
# The shipped file has two worked examples and no real data.
set -euo pipefail

powerrank audit \
  --claims experiments/claims_template.csv \
  --out results/audit.csv
