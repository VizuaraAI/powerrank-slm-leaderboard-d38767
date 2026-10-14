"""Tests for the power math.

The statistics are the contribution of this project, so they are the part that
has to be right. These tests pin the headline numbers: if a refactor moves
them, the paper's claims move too, and you want to find that out here rather
than in review.
"""

import numpy as np
import pandas as pd
import pytest

from powerrank.leaderboard import build_leaderboard, pairwise_comparisons
from powerrank.power import (
    benjamini_hochberg,
    mcnemar_test,
    mde_mcnemar,
    paired_bootstrap_ci,
    paired_counts,
    power_mcnemar,
    required_n_mcnemar,
    required_n_unpaired,
    wilson_ci,
)
from powerrank.variance import decompose, effective_discordance


# ---------------------------------------------------------------- headline
def test_one_point_gap_needs_13k_to_17k_items():
    """The claim the whole project rests on."""
    for disc in (0.17, 0.18, 0.20, 0.21):
        n = required_n_mcnemar(delta=0.01, discordance=disc)
        assert 13_000 <= n <= 17_000, f"discordance {disc} gave n={n}"


def test_paired_is_several_times_cheaper_than_unpaired():
    paired = required_n_mcnemar(0.01, 0.20)
    unpaired_total = 2 * required_n_unpaired(0.01, p_bar=0.60)
    assert unpaired_total > 4 * paired


def test_required_n_scales_as_inverse_square_of_delta():
    """Halving the effect size should roughly quadruple the item count."""
    n1 = required_n_mcnemar(0.02, 0.20)
    n2 = required_n_mcnemar(0.01, 0.20)
    assert 3.8 < n2 / n1 < 4.2


def test_mde_at_the_full_pool_is_about_one_point():
    mde = mde_mcnemar(n=17_248, discordance=0.20)
    assert 0.008 < mde < 0.011


def test_mde_and_required_n_are_inverses():
    n = required_n_mcnemar(0.01, 0.20)
    assert mde_mcnemar(n, 0.20) == pytest.approx(0.01, rel=0.02)


def test_power_rises_with_n():
    low = power_mcnemar(n=1_000, delta=0.01, discordance=0.20)
    high = power_mcnemar(n=17_248, delta=0.01, discordance=0.20)
    assert low < 0.2 < 0.75 < high


def test_typical_benchmark_is_underpowered_for_one_point():
    """A 1,500 item benchmark cannot see a one-point difference."""
    assert mde_mcnemar(1_500, 0.20) * 100 > 2.5
    assert power_mcnemar(1_500, 0.01, 0.20) < 0.25


# ---------------------------------------------------------------- guards
@pytest.mark.parametrize("bad", [0.0, 1.0, 1.5, -2.0])
def test_delta_must_be_a_nonzero_proportion(bad):
    with pytest.raises(ValueError):
        required_n_mcnemar(bad, 0.20)


def test_delta_cannot_exceed_discordance():
    with pytest.raises(ValueError):
        required_n_mcnemar(delta=0.25, discordance=0.20)


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_discordance_must_be_a_valid_rate(bad):
    with pytest.raises(ValueError):
        mde_mcnemar(1_000, bad)


def test_misaligned_vectors_are_rejected():
    with pytest.raises(ValueError):
        paired_counts([1, 0, 1], [1, 0])


# ---------------------------------------------------------------- paired table
def test_paired_counts_recovers_the_table():
    a = [1, 1, 1, 0, 0, 0]
    b = [1, 0, 0, 1, 0, 0]
    c = paired_counts(a, b)
    assert (c.n11, c.n10, c.n01, c.n00) == (1, 2, 1, 2)
    assert c.n == 6
    assert c.discordance == pytest.approx(3 / 6)
    assert c.delta == pytest.approx(1 / 6)
    assert c.acc_a == pytest.approx(0.5)
    assert c.acc_b == pytest.approx(2 / 6)


def test_mcnemar_finds_no_difference_when_there_is_none():
    rng = np.random.default_rng(0)
    a = rng.integers(0, 2, 4_000)
    _, p = mcnemar_test(paired_counts(a, a.copy()))
    assert p == 1.0


def test_mcnemar_finds_a_real_difference_at_adequate_n():
    rng = np.random.default_rng(1)
    n = 20_000
    a = rng.random(n) < 0.60
    # b is a copy with a 1.5 point handicap applied to a random subset
    b = a.copy()
    flip = rng.random(n) < 0.015
    b[flip & a] = False
    _, p = mcnemar_test(paired_counts(a.astype(int), b.astype(int)))
    assert p < 0.01


def test_exact_test_used_for_small_discordant_counts():
    counts = paired_counts([1, 1, 1, 0, 0], [1, 1, 0, 0, 0])
    stat, p = mcnemar_test(counts)
    assert 0.0 <= p <= 1.0
    assert stat == counts.n10


# ---------------------------------------------------------------- intervals
def test_bootstrap_ci_covers_zero_for_identical_models():
    rng = np.random.default_rng(2)
    a = rng.integers(0, 2, 1_000)
    lo, hi = paired_bootstrap_ci(a, a.copy(), n_boot=400, seed=3)
    assert lo <= 0.0 <= hi


def test_bootstrap_ci_excludes_zero_for_a_large_difference():
    a = np.ones(2_000, dtype=int)
    b = np.zeros(2_000, dtype=int)
    lo, hi = paired_bootstrap_ci(a, b, n_boot=400, seed=4)
    assert lo > 0.5


def test_clustered_bootstrap_is_wider_than_unclustered():
    """Correlated items inside clusters should widen the interval."""
    rng = np.random.default_rng(5)
    cluster = np.repeat(np.arange(50), 20)
    effect = rng.random(50) < 0.5
    a = np.repeat(effect, 20).astype(int)
    b = rng.integers(0, 2, 1_000)
    plain = paired_bootstrap_ci(a, b, n_boot=400, seed=6)
    clust = paired_bootstrap_ci(a, b, n_boot=400, cluster=cluster, seed=6)
    assert (clust[1] - clust[0]) > (plain[1] - plain[0])


def test_wilson_interval_is_inside_the_unit_range():
    lo, hi = wilson_ci(0, 30)
    assert lo >= 0.0 and hi > 0.0
    lo, hi = wilson_ci(30, 30)
    assert hi <= 1.0 and lo < 1.0


# ---------------------------------------------------------------- multiplicity
def test_bh_rejects_nothing_under_the_null():
    rng = np.random.default_rng(7)
    p = rng.random(190)
    assert benjamini_hochberg(p, fdr=0.05).sum() <= 3


def test_bh_rejects_clear_signal():
    p = np.array([1e-9] * 10 + list(np.linspace(0.2, 0.99, 180)))
    assert benjamini_hochberg(p, fdr=0.05)[:10].all()


def test_bh_is_less_permissive_than_uncorrected():
    rng = np.random.default_rng(8)
    p = np.concatenate([rng.random(185), [0.001, 0.004, 0.01, 0.02, 0.04]])
    assert benjamini_hochberg(p, fdr=0.05).sum() < (p < 0.05).sum()


def test_bh_handles_empty_input():
    assert benjamini_hochberg([]).size == 0


# ---------------------------------------------------------------- variance
def _toy_responses(seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    item_difficulty = rng.random(120)
    for m, skill in enumerate([0.0, 0.08]):
        for i, diff in enumerate(item_difficulty):
            for p_id in ("plain", "instructed"):
                bump = 0.03 if p_id == "instructed" else 0.0
                correct = int(rng.random() < min(0.95, diff + skill + bump))
                rows.append({"model": f"m{m}", "item_id": f"i{i}",
                             "prompt_id": p_id, "seed": 0, "correct": correct})
    return pd.DataFrame(rows)


def test_variance_decomposition_shares_sum_to_one():
    comps = decompose(_toy_responses())
    assert sum(comps.shares().values()) == pytest.approx(1.0)
    assert comps.total > 0


def test_item_variance_dominates_model_variance():
    """The central empirical claim of the paper, on toy data."""
    comps = decompose(_toy_responses())
    assert comps.item > comps.model


def test_decompose_rejects_a_malformed_table():
    with pytest.raises(ValueError):
        decompose(pd.DataFrame({"model": ["a"], "correct": [1]}))
    with pytest.raises(ValueError):
        decompose(pd.DataFrame(columns=["model", "item_id", "prompt_id", "seed", "correct"]))


def test_prompt_averaging_lowers_the_effective_discordance():
    comps = decompose(_toy_responses())
    one = effective_discordance(0.20, comps, n_prompts=1)
    five = effective_discordance(0.20, comps, n_prompts=5)
    assert one == pytest.approx(0.20)
    assert five < one


# ---------------------------------------------------------------- leaderboard
def test_identical_models_land_in_one_band():
    rng = np.random.default_rng(9)
    scores = rng.integers(0, 2, 600).astype(float)
    matrix = pd.DataFrame({"a": scores, "b": scores.copy(), "c": scores.copy()})
    lb, pairs = build_leaderboard(matrix)
    assert not pairs["significant"].any()
    assert lb["band"].nunique() == 1
    assert (lb["rank_best"] == 1).all()


def test_clearly_separated_models_get_distinct_bands():
    n = 3_000
    rng = np.random.default_rng(10)
    strong = (rng.random(n) < 0.80).astype(float)
    weak = (rng.random(n) < 0.30).astype(float)
    matrix = pd.DataFrame({"strong": strong, "weak": weak})
    lb, pairs = build_leaderboard(matrix)
    assert pairs["significant"].all()
    assert lb["band"].nunique() == 2
    assert lb.iloc[0]["model"] == "strong"


def test_pairwise_covers_every_pair():
    rng = np.random.default_rng(11)
    matrix = pd.DataFrame({c: rng.integers(0, 2, 300).astype(float)
                           for c in ("a", "b", "c", "d")})
    assert len(pairwise_comparisons(matrix, n_boot=100)) == 6


def test_leaderboard_needs_at_least_two_models():
    with pytest.raises(ValueError):
        pairwise_comparisons(pd.DataFrame({"only": [1.0, 0.0]}))


def test_reported_interval_brackets_the_reported_accuracy():
    """Regression: the CI must describe the same quantity as the accuracy.

    Prompt-averaged scores are continuous, so a binomial interval built from a
    thresholded success count does not match the mean that is printed beside
    it. This caught a leaderboard reporting 62.6% with a CI of [83.7, 85.9].
    """
    rng = np.random.default_rng(12)
    # scores in {0, 0.5, 1}, as produced by averaging over two prompts
    matrix = pd.DataFrame({
        "a": rng.choice([0.0, 0.5, 1.0], 800, p=[0.3, 0.3, 0.4]),
        "b": rng.choice([0.0, 0.5, 1.0], 800, p=[0.4, 0.3, 0.3]),
    })
    lb, _ = build_leaderboard(matrix)
    for row in lb.itertuples():
        assert row.ci_low <= row.accuracy <= row.ci_high, (
            f"{row.model}: {row.accuracy:.3f} outside [{row.ci_low:.3f}, {row.ci_high:.3f}]"
        )


def test_majority_vote_needs_a_strict_majority():
    """An exact prompt split is not a win."""
    matrix = pd.DataFrame({"a": [0.5] * 100, "b": [0.0] * 100})
    _, pairs = build_leaderboard(matrix)
    assert not pairs["significant"].any()
