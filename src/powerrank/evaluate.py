"""Run the evaluation grid and write the item-level response tensor.

This is the expensive step and the one the whole paper depends on. The output
is NOT a table of scores. It is one row per (model, item, prompt, seed), which
is what makes paired tests, variance decomposition and rank bands possible
later. Aggregate scores are computed on demand from this table and are never
the stored artifact.

Size check for the full run: 10 models x 17,000 items x 5 prompts x 3 seeds is
about 2.6M rows, roughly 60MB as parquet. That is cheap to store and cheap to
re-analyse, which is the point: you pay the GPU cost once and can then answer
any statistical question offline.

Runs are resumable. Each (model, prompt, seed) shard is written separately and
skipped if it already exists, so a crash at hour nine costs you one shard.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd

from .models import MODEL_ZOO, ModelSpec, SLMRunner

logger = logging.getLogger(__name__)

__all__ = ["run_grid", "load_responses", "aggregate_scores", "response_matrix"]


def _shard_path(out_dir: Path, model: str, prompt: str, seed: int) -> Path:
    safe = model.replace("/", "_")
    return out_dir / f"{safe}__{prompt}__seed{seed}.parquet"


def run_grid(
    bank: pd.DataFrame,
    out_dir: str | Path,
    models: tuple[ModelSpec, ...] = MODEL_ZOO,
    templates: tuple[str, ...] = ("plain", "instructed", "qa_pair"),
    seeds: tuple[int, ...] = (0,),
    temperature: float = 0.0,
    resume: bool = True,
    limit: int | None = None,
) -> Path:
    """Evaluate every (model, prompt, seed) cell over the item bank.

    Returns the output directory. Call :func:`load_responses` on it afterwards.

    Note on ``seeds``: with ``temperature=0.0`` the multiple-choice path is
    deterministic, so multiple seeds only buy you something on the generative
    items or at a non-zero temperature. Keep ``seeds=(0,)`` until you are
    ready to pay for the seed axis.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    work = bank.head(limit) if limit else bank
    total_cells = len(models) * len(templates) * len(seeds)
    logger.info(
        "grid: %d models x %d items x %d prompts x %d seeds = %s scored items",
        len(models), len(work), len(templates), len(seeds),
        f"{len(models) * len(work) * len(templates) * len(seeds):,}",
    )

    cell = 0
    for spec in models:
        pending = [
            (t, s)
            for t in templates
            for s in seeds
            if not (resume and _shard_path(out_dir, spec.name, t, s).exists())
        ]
        if not pending:
            logger.info("%s: all shards present, skipping model load", spec.name)
            cell += len(templates) * len(seeds)
            continue

        with SLMRunner(spec) as runner:
            for template, seed in pending:
                cell += 1
                shard = _shard_path(out_dir, spec.name, template, seed)
                started = time.time()
                rows = []
                for row in work.itertuples():
                    scored = runner.score_item(row, template, seed, temperature)
                    rows.append(
                        {
                            "model": spec.name,
                            "params_b": spec.params_b,
                            "item_id": scored.item_id,
                            "source": row.source,
                            "skill": row.skill,
                            "prompt_id": template,
                            "seed": seed,
                            "predicted": scored.predicted,
                            "correct": scored.correct,
                        }
                    )
                df = pd.DataFrame(rows)
                df.to_parquet(shard, index=False)
                logger.info(
                    "[%d/%d] %s %s seed=%d  acc=%.3f  %.1fs",
                    cell, total_cells, spec.name, template, seed,
                    df["correct"].mean(), time.time() - started,
                )

    return out_dir


def load_responses(out_dir: str | Path) -> pd.DataFrame:
    """Concatenate all shards into the long-format response table."""
    out_dir = Path(out_dir)
    shards = sorted(out_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"no response shards in {out_dir}; run `powerrank evaluate` first"
        )
    df = pd.concat([pd.read_parquet(s) for s in shards], ignore_index=True)
    logger.info("loaded %s responses from %d shards", f"{len(df):,}", len(shards))
    return df


def aggregate_scores(responses: pd.DataFrame, by: tuple[str, ...] = ("model",)) -> pd.DataFrame:
    """Mean accuracy with item counts. Convenience only: the real analysis
    lives in ``leaderboard.py`` and works on the item level."""
    out = (
        responses.groupby(list(by), observed=True)["correct"]
        .agg(accuracy="mean", n="size")
        .reset_index()
        .sort_values("accuracy", ascending=False)
    )
    return out.reset_index(drop=True)


def response_matrix(responses: pd.DataFrame) -> pd.DataFrame:
    """Pivot to items x models, averaging over prompts and seeds.

    This is the matrix the paired tests consume. Averaging over the prompt
    axis before pairing is the "prompt averaging" design in the paper: it
    reduces the discordance rate and therefore the item count you need.
    Items not attempted by every model are dropped, because a paired test
    requires complete pairs.
    """
    wide = responses.pivot_table(
        index="item_id", columns="model", values="correct", aggfunc="mean", observed=True
    )
    before = len(wide)
    wide = wide.dropna(axis=0, how="any")
    if len(wide) < before:
        logger.warning(
            "dropped %d items not attempted by all models (pairing requires complete rows)",
            before - len(wide),
        )
    return wide
