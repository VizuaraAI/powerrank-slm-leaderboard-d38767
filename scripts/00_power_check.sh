#!/usr/bin/env bash
# Day one. No GPU, no downloads, about one second.
# Prints the number this whole project is built on.
set -euo pipefail

echo "=== how many items does a one-point claim need? ==="
powerrank power --delta 1.0 --discordance 0.20

echo "=== and a two-point claim? ==="
powerrank power --delta 2.0 --discordance 0.20

echo "=== sensitivity: the same question at a lower discordance rate ==="
powerrank power --delta 1.0 --discordance 0.15
