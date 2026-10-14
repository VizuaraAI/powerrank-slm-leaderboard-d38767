"""Where does the variance in a benchmark score actually come from?

A leaderboard reports one number per model. That number moves around for at
least four reasons, and only the last one is the thing we want to measure:

  item      : some questions are harder than others (by far the largest share)
  prompt    : the same model scores differently under a reformatted prompt
  seed      : sampling / decoding noise across repeated runs
  model     : genuine capability difference, the signal

This module estimates those components by method of moments from the response
tensor produced by ``evaluate.py``. Method of moments is used rather than a
full hierarchical fit because it is transparent, has no optimiser to babysit,
and runs in milliseconds on a 5M cell tensor. ``fit_glmm_statsmodels`` is
provided as the upgrade path once you want proper credible intervals on the
components themselves.

The practical payoff: averaging a model's score over ``k`` prompt templates
shrinks the prompt component by a factor of ``k``, which lowers the
discordance rate between two models, which lowers the item count you need.
That is the "paired + prompt averaging" curve in Figure 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

__all__ = ["VarianceComponents", "decompose", "effective_discordance", "fit_glmm_statsmodels"]

REQUIRED_COLUMNS = ("model", "item_id", "prompt_id", "seed", "correct")


@dataclass(frozen=True)
class VarianceComponents:
    """Variance of the per-item correctness outcome, split by source."""

    item: float
    prompt: float
    seed: float
    model: float
    residual: float

    @property
    def total(self) -> float:
        return self.item + self.prompt + self.seed + self.model + self.residual

    def shares(self) -> dict[str, float]:
        """Each component as a fraction of the total. This is the stacked bar
        in Figure 1, and the model share is usually alarmingly thin."""
        total = self.total
        if total <= 0:
            return {k: 0.0 for k in asdict(self)}
        return {k: v / total for k, v in asdict(self).items()}

    def __str__(self) -> str:
        rows = [f"  {k:<9} {v:.5f}  ({s:5.1%})" for (k, v), s in zip(asdict(self).items(), self.shares().values())]
        return "variance components\n" + "\n".join(rows) + f"\n  {'total':<9} {self.total:.5f}"


def decompose(df: pd.DataFrame) -> VarianceComponents:
    """Estimate variance components from a long-format response table.

    Expects one row per (model, item, prompt, seed) with a 0/1 ``correct``
    column. Components are the variances of the corresponding marginal means,
    which is the method-of-moments estimator for a crossed random-effects
    design on a balanced grid.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"response table is missing columns: {missing}")
    if df.empty:
        raise ValueError("response table is empty; run the evaluation first")

    y = df["correct"].to_numpy(dtype=float)
    grand = float(y.mean())

    def marginal_var(key: str) -> float:
        means = df.groupby(key, observed=True)["correct"].mean().to_numpy(dtype=float)
        if means.size < 2:
            return 0.0
        return float(np.var(means, ddof=1))

    v_item = marginal_var("item_id")
    v_prompt = marginal_var("prompt_id")
    v_seed = marginal_var("seed")
    v_model = marginal_var("model")

    # Bernoulli total variance, minus what the named factors already explain.
    v_total = grand * (1.0 - grand)
    v_residual = max(0.0, v_total - (v_item + v_prompt + v_seed + v_model))

    return VarianceComponents(
        item=v_item, prompt=v_prompt, seed=v_seed, model=v_model, residual=v_residual
    )


def effective_discordance(
    observed_discordance: float,
    components: VarianceComponents,
    n_prompts: int,
) -> float:
    """Discordance after averaging each model over ``n_prompts`` templates.

    Prompt-induced disagreement averages out at rate 1/k; item-level
    disagreement does not, because it is a real property of the items. This
    returns the floor-respecting estimate to feed into
    :func:`powerrank.power.required_n_mcnemar`.
    """
    if n_prompts < 1:
        raise ValueError(f"n_prompts must be at least 1, got {n_prompts}")
    if not 0.0 < observed_discordance <= 1.0:
        raise ValueError(f"observed_discordance must be in (0, 1], got {observed_discordance}")
    shares = components.shares()
    prompt_share = shares["prompt"] + shares["seed"]
    reducible = observed_discordance * prompt_share
    irreducible = observed_discordance - reducible
    return float(irreducible + reducible / n_prompts)


def fit_glmm_statsmodels(df: pd.DataFrame, group: str = "item_id"):
    """Optional: a mixed-effects fit for one random factor at a time.

    ``statsmodels`` only supports a single grouping factor, so call this once
    per factor, or move to ``pymc`` / ``lme4`` for the fully crossed model
    described in the paper. Imported lazily so the rest of the package does
    not depend on it.
    """
    import statsmodels.formula.api as smf

    if group not in df.columns:
        raise ValueError(f"grouping column {group!r} not in response table")
    model = smf.mixedlm("correct ~ C(model)", data=df, groups=df[group])
    return model.fit()
