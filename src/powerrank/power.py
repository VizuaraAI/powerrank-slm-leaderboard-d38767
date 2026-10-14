"""Statistical power for benchmark comparisons.

This is the heart of the project. Everything else in the repo exists to feed
these functions real numbers.

The central question: if model A is truly one accuracy point better than model
B, how many benchmark items do we need before we can reliably detect it?

Three designs are implemented, in increasing efficiency:

1. ``required_n_unpaired``   : two independent samples of items (the design
   implicitly assumed when people quote a standard error of sqrt(p(1-p)/n)).
2. ``required_n_mcnemar``    : both models answer the SAME items, so we only
   pay for the items where they disagree. This is what a leaderboard actually
   does, and it is several times cheaper.
3. ``required_n_mcnemar`` with a reduced discordance, which is what prompt
   averaging buys you (see ``variance.py``).

Reference points worth remembering (alpha=0.05 two-sided, power=0.80):

    delta = 1.0 pt, discordance = 0.20  ->  about 15,700 paired items
    delta = 1.0 pt, unpaired, p = 0.60  ->  about 37,700 items PER MODEL
    delta = 2.0 pt, discordance = 0.20  ->  about 3,900 paired items

That first line is the claim the paper is built on, and it is why a 1,000 item
benchmark cannot settle a one-point difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
from scipy import stats

__all__ = [
    "PairedCounts",
    "paired_counts",
    "required_n_unpaired",
    "required_n_mcnemar",
    "mde_mcnemar",
    "power_mcnemar",
    "mcnemar_test",
    "paired_bootstrap_ci",
    "wilson_ci",
    "benjamini_hochberg",
]


# --------------------------------------------------------------------------
# paired contingency summary
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PairedCounts:
    """The 2x2 table for two models scored on the same items.

    ``n01`` is the number of items where A was wrong and B was right,
    ``n10`` where A was right and B was wrong. Those two cells carry all of
    the information about the difference; the agreeing cells do not.
    """

    n11: int  # both correct
    n10: int  # A correct, B wrong
    n01: int  # A wrong, B correct
    n00: int  # both wrong

    @property
    def n(self) -> int:
        return self.n11 + self.n10 + self.n01 + self.n00

    @property
    def discordance(self) -> float:
        """Fraction of items on which the two models disagree."""
        return (self.n10 + self.n01) / self.n if self.n else 0.0

    @property
    def delta(self) -> float:
        """Accuracy of A minus accuracy of B, as a proportion."""
        return (self.n10 - self.n01) / self.n if self.n else 0.0

    @property
    def acc_a(self) -> float:
        return (self.n11 + self.n10) / self.n if self.n else 0.0

    @property
    def acc_b(self) -> float:
        return (self.n11 + self.n01) / self.n if self.n else 0.0


def paired_counts(correct_a: Sequence[int], correct_b: Sequence[int]) -> PairedCounts:
    """Build the 2x2 table from two aligned 0/1 correctness vectors."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    if a.shape != b.shape:
        raise ValueError(
            f"correctness vectors must be aligned item-for-item, got {a.shape} and {b.shape}"
        )
    return PairedCounts(
        n11=int(np.sum(a & b)),
        n10=int(np.sum(a & ~b)),
        n01=int(np.sum(~a & b)),
        n00=int(np.sum(~a & ~b)),
    )


# --------------------------------------------------------------------------
# sample size / power / minimum detectable effect
# --------------------------------------------------------------------------


def _z(alpha: float, power: float) -> tuple[float, float]:
    z_alpha = stats.norm.ppf(1.0 - alpha / 2.0)
    z_beta = stats.norm.ppf(power)
    return float(z_alpha), float(z_beta)


def required_n_unpaired(
    delta: float,
    p_bar: float = 0.60,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Items required PER MODEL when the two models see different item samples.

    Standard two-proportion comparison. ``delta`` and ``p_bar`` are
    proportions, so a one point gap is ``delta=0.01``.
    """
    _validate_delta(delta)
    if not 0.0 < p_bar < 1.0:
        raise ValueError(f"p_bar must be in (0, 1), got {p_bar}")
    z_alpha, z_beta = _z(alpha, power)
    n = (z_alpha + z_beta) ** 2 * 2.0 * p_bar * (1.0 - p_bar) / delta**2
    return int(np.ceil(n))


def required_n_mcnemar(
    delta: float,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> int:
    """Paired items required to detect ``delta`` with McNemar's test.

    ``discordance`` is the fraction of items the two models answer
    differently. Measure it from your own runs with :func:`paired_counts`
    rather than guessing: it is the single most important input here, and for
    two closely matched SLMs it typically lands between 0.15 and 0.30.

    Uses the normal approximation to the conditional binomial:

        n = [z_{alpha/2} sqrt(pi_d) + z_beta sqrt(pi_d - delta^2)]^2 / delta^2
    """
    _validate_delta(delta)
    if not 0.0 < discordance <= 1.0:
        raise ValueError(f"discordance must be in (0, 1], got {discordance}")
    if abs(delta) > discordance:
        raise ValueError(
            f"|delta|={abs(delta)} cannot exceed the discordance rate {discordance}; "
            "the difference is carried entirely by discordant items"
        )
    z_alpha, z_beta = _z(alpha, power)
    num = z_alpha * np.sqrt(discordance) + z_beta * np.sqrt(discordance - delta**2)
    return int(np.ceil(num**2 / delta**2))


def mde_mcnemar(
    n: int,
    discordance: float,
    alpha: float = 0.05,
    power: float = 0.80,
) -> float:
    """Minimum detectable effect: the smallest gap ``n`` paired items can find.

    Returns a proportion. Any reported gap below this value is not evidence of
    a difference, and this number is what the paper asks leaderboards to print
    next to their scores.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < discordance <= 1.0:
        raise ValueError(f"discordance must be in (0, 1], got {discordance}")
    z_alpha, z_beta = _z(alpha, power)
    # Ignoring the delta^2 term inside the square root (it is negligible for
    # the small effects we care about) gives a closed form.
    return float((z_alpha + z_beta) * np.sqrt(discordance / n))


def power_mcnemar(
    n: int,
    delta: float,
    discordance: float,
    alpha: float = 0.05,
) -> float:
    """Achieved power of a paired comparison. Invert of :func:`required_n_mcnemar`."""
    _validate_delta(delta)
    if not 0.0 < discordance <= 1.0:
        raise ValueError(f"discordance must be in (0, 1], got {discordance}")
    z_alpha, _ = _z(alpha, 0.5)
    se = np.sqrt(discordance / n)
    return float(stats.norm.cdf(abs(delta) / se - z_alpha))


def _validate_delta(delta: float) -> None:
    if delta == 0.0:
        raise ValueError("delta must be non-zero; there is no sample size that detects no effect")
    if abs(delta) >= 1.0:
        raise ValueError(
            f"delta is a proportion, not a percentage: pass 0.01 for one point, got {delta}"
        )


# --------------------------------------------------------------------------
# tests and intervals
# --------------------------------------------------------------------------


def mcnemar_test(counts: PairedCounts, exact: bool | None = None) -> tuple[float, float]:
    """Paired test of equal accuracy. Returns ``(statistic, p_value)``.

    Uses the exact conditional binomial test when the discordant count is
    small (or when ``exact=True``), otherwise the continuity-corrected
    chi-square. With 17k items you will be in the chi-square regime, but the
    exact test matters when you slice down to a single skill.
    """
    b, c = counts.n10, counts.n01
    n_disc = b + c
    if n_disc == 0:
        return 0.0, 1.0
    if exact is None:
        exact = n_disc < 25
    if exact:
        result = stats.binomtest(b, n_disc, p=0.5, alternative="two-sided")
        return float(b), float(result.pvalue)
    stat = (abs(b - c) - 1.0) ** 2 / n_disc
    return float(stat), float(stats.chi2.sf(stat, df=1))


def paired_bootstrap_ci(
    correct_a: Sequence[int],
    correct_b: Sequence[int],
    n_boot: int = 10_000,
    alpha: float = 0.05,
    cluster: Sequence[int] | None = None,
    seed: int = 0,
) -> tuple[float, float]:
    """Bootstrap CI for the accuracy difference, resampling items (not runs).

    Pass ``cluster`` (for example a per-item source-benchmark id, or a passage
    id shared by several questions) to resample whole clusters. Ignoring
    clustering is a quiet way to make your intervals too narrow.
    """
    a = np.asarray(correct_a, dtype=float)
    b = np.asarray(correct_b, dtype=float)
    if a.shape != b.shape:
        raise ValueError("correctness vectors must be aligned item-for-item")
    diff = a - b
    rng = np.random.default_rng(seed)

    if cluster is None:
        n = diff.size
        idx = rng.integers(0, n, size=(n_boot, n))
        draws = diff[idx].mean(axis=1)
    else:
        clusters = np.asarray(cluster)
        groups = [np.flatnonzero(clusters == c) for c in np.unique(clusters)]
        sums = np.array([diff[g].sum() for g in groups])
        sizes = np.array([g.size for g in groups], dtype=float)
        k = len(groups)
        pick = rng.integers(0, k, size=(n_boot, k))
        draws = sums[pick].sum(axis=1) / sizes[pick].sum(axis=1)

    lo, hi = np.quantile(draws, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def wilson_ci(successes: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson interval for a single model's accuracy.

    Preferred over the Wald interval, which misbehaves badly for the small
    per-skill slices you will inevitably want to report.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    z = stats.norm.ppf(1.0 - alpha / 2.0)
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2.0 * n)) / denom
    half = z * np.sqrt(p * (1.0 - p) / n + z**2 / (4.0 * n**2)) / denom
    return float(centre - half), float(centre + half)


def benjamini_hochberg(p_values: Iterable[float], fdr: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up. Returns a boolean array of rejections.

    A 20 model leaderboard has 190 pairwise comparisons, so about 10 of them
    will look significant at p<0.05 under the null. Correct, or the "wins" you
    report are mostly noise.
    """
    p = np.asarray(list(p_values), dtype=float)
    if p.size == 0:
        return np.zeros(0, dtype=bool)
    order = np.argsort(p)
    ranked = p[order]
    thresholds = fdr * np.arange(1, p.size + 1) / p.size
    passing = np.flatnonzero(ranked <= thresholds)
    reject = np.zeros(p.size, dtype=bool)
    if passing.size:
        reject[order[: passing[-1] + 1]] = True
    return reject
