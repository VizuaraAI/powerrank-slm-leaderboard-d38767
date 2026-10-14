"""Turn the response tensor into a leaderboard that admits what it cannot know.

A conventional leaderboard sorts by mean accuracy and assigns ranks 1..N. That
implies every adjacent pair is distinguishable, which is almost never true at
realistic benchmark sizes. Here we instead report:

  * accuracy with a Wilson interval
  * every pairwise McNemar test, Benjamini-Hochberg corrected across all
    N(N-1)/2 comparisons
  * a SIGNIFICANT RANK INTERVAL per model: the best and worst rank the model
    could hold given only the comparisons that reached significance
  * RANK BANDS: maximal groups of models that no test could separate

The significant-rank construction is the standard one: a model's best possible
rank is one plus the number of models that significantly beat it, and its
worst is N minus the number it significantly beats.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .power import (
    PairedCounts,
    benjamini_hochberg,
    mcnemar_test,
    mde_mcnemar,
    paired_bootstrap_ci,
)

logger = logging.getLogger(__name__)

__all__ = ["PairwiseResult", "pairwise_comparisons", "build_leaderboard", "format_leaderboard"]


@dataclass(frozen=True)
class PairwiseResult:
    model_a: str
    model_b: str
    acc_a: float
    acc_b: float
    delta: float
    discordance: float
    n: int
    p_value: float
    mde: float
    ci_low: float
    ci_high: float
    significant: bool = False


def _counts_from_means(a: np.ndarray, b: np.ndarray) -> PairedCounts:
    """Build a paired table from prompt-averaged scores in [0, 1].

    McNemar's test needs binary outcomes, so averaging over prompts is reduced
    to a majority vote: an item counts as correct if the model got it right
    under more than half of the templates. The continuous information is not
    wasted, it is what shrinks the discordance rate relative to a single-prompt
    run.

    Use an ODD number of prompt templates. With an even number, an exact 0.5
    split has no majority and the vote has to break the tie arbitrarily, which
    biases every downstream count. :func:`pairwise_comparisons` warns when it
    sees ties.
    """
    ab = a > 0.5
    bb = b > 0.5
    return PairedCounts(
        n11=int(np.sum(ab & bb)),
        n10=int(np.sum(ab & ~bb)),
        n01=int(np.sum(~ab & bb)),
        n00=int(np.sum(~ab & ~bb)),
    )


def _mean_ci(
    scores: np.ndarray, alpha: float = 0.05, n_boot: int = 2_000, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap interval for a model's mean score.

    Used instead of :func:`powerrank.power.wilson_ci` because the reported
    accuracy is a mean of prompt-averaged (continuous) per-item scores, and a
    binomial interval would not match it. Wilson remains the right choice for a
    genuinely binary single-prompt slice.
    """
    rng = np.random.default_rng(seed)
    n = scores.size
    draws = scores[rng.integers(0, n, size=(n_boot, n))].mean(axis=1)
    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def pairwise_comparisons(
    matrix: pd.DataFrame,
    alpha: float = 0.05,
    fdr: float = 0.05,
    n_boot: int = 2_000,
    power: float = 0.80,
) -> pd.DataFrame:
    """All pairwise paired comparisons from an items x models matrix.

    ``matrix`` comes from :func:`powerrank.evaluate.response_matrix`.
    """
    models = list(matrix.columns)
    if len(models) < 2:
        raise ValueError(f"need at least two models to compare, got {models}")

    n_ties = int(np.sum(matrix.to_numpy(dtype=float) == 0.5))
    if n_ties:
        logger.warning(
            "%d item/model cells sit exactly at 0.5 (an even prompt split with no "
            "majority); these count as incorrect. Use an odd number of prompt "
            "templates to avoid this.",
            n_ties,
        )

    results: list[PairwiseResult] = []
    for i, m_a in enumerate(models):
        for m_b in models[i + 1 :]:
            a = matrix[m_a].to_numpy(dtype=float)
            b = matrix[m_b].to_numpy(dtype=float)
            counts = _counts_from_means(a, b)
            _, p = mcnemar_test(counts)
            lo, hi = paired_bootstrap_ci(a, b, n_boot=n_boot, alpha=alpha)
            disc = max(counts.discordance, 1.0 / counts.n)  # guard an exact tie
            results.append(
                PairwiseResult(
                    model_a=m_a,
                    model_b=m_b,
                    acc_a=float(a.mean()),
                    acc_b=float(b.mean()),
                    delta=float(a.mean() - b.mean()),
                    discordance=disc,
                    n=counts.n,
                    p_value=p,
                    mde=mde_mcnemar(counts.n, disc, alpha=alpha, power=power),
                    ci_low=lo,
                    ci_high=hi,
                )
            )

    df = pd.DataFrame([vars(r) for r in results])
    df["significant"] = benjamini_hochberg(df["p_value"], fdr=fdr)
    logger.info(
        "%d pairwise comparisons, %d significant after BH at FDR=%.2f",
        len(df), int(df["significant"].sum()), fdr,
    )
    return df


def build_leaderboard(
    matrix: pd.DataFrame,
    pairwise: pd.DataFrame | None = None,
    alpha: float = 0.05,
    fdr: float = 0.05,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(leaderboard, pairwise)``.

    The leaderboard has one row per model with accuracy, a Wilson interval,
    the significant rank interval, and a band id. Models sharing a band id
    could not be separated by any test at this sample size.
    """
    if pairwise is None:
        pairwise = pairwise_comparisons(matrix, alpha=alpha, fdr=fdr)

    models = list(matrix.columns)
    n_models = len(models)
    n_items = len(matrix)

    beaten_by: dict[str, int] = {m: 0 for m in models}
    beats: dict[str, int] = {m: 0 for m in models}
    for row in pairwise.itertuples():
        if not row.significant:
            continue
        winner, loser = (row.model_a, row.model_b) if row.delta > 0 else (row.model_b, row.model_a)
        beats[winner] += 1
        beaten_by[loser] += 1

    rows = []
    for m in models:
        scores = matrix[m].to_numpy(dtype=float)
        # The interval must describe the same quantity as the reported
        # accuracy, so it is bootstrapped from the item means rather than from
        # a thresholded success count.
        lo, hi = _mean_ci(scores, alpha=alpha)
        rows.append(
            {
                "model": m,
                "accuracy": float(scores.mean()),
                "ci_low": lo,
                "ci_high": hi,
                "n_items": n_items,
                "rank_best": 1 + beaten_by[m],
                "rank_worst": n_models - beats[m],
                "sig_wins": beats[m],
                "sig_losses": beaten_by[m],
            }
        )

    lb = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)

    # Models with the same best-possible rank form one band.
    band_id = 0
    bands = []
    previous = None
    for best in lb["rank_best"]:
        if previous is not None and best != previous:
            band_id += 1
        bands.append(band_id + 1)
        previous = best
    lb["band"] = bands
    lb.insert(0, "displayed_rank", lb.index + 1)

    n_bands = lb["band"].nunique()
    logger.info(
        "%d models collapse into %d distinguishable bands at n=%d items",
        n_models, n_bands, n_items,
    )
    return lb, pairwise


def format_leaderboard(lb: pd.DataFrame) -> str:
    """Plain-text leaderboard, grouped by band. Paste straight into your log."""
    lines = [
        f"{'band':>4}  {'model':<26} {'accuracy':>9}  {'95% CI':>16}  {'rank':>9}",
        "-" * 74,
    ]
    for band, group in lb.groupby("band", sort=True):
        for i, row in enumerate(group.itertuples()):
            band_cell = f"{band}" if i == 0 else ""
            ci = f"[{row.ci_low * 100:.1f}, {row.ci_high * 100:.1f}]"
            rank = (
                f"{row.rank_best}"
                if row.rank_best == row.rank_worst
                else f"{row.rank_best}-{row.rank_worst}"
            )
            lines.append(
                f"{band_cell:>4}  {row.model:<26} {row.accuracy * 100:8.1f}%  {ci:>16}  {rank:>9}"
            )
        if len(group) > 1:
            lines.append(f"{'':>4}  ({len(group)} models statistically tied)")
    return "\n".join(lines)
