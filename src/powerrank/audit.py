"""Retrospective power audit of already-published comparisons.

The second contribution of the paper. For each claim of the form "model A
beats model B by delta points on benchmark X", we ask a single question: was
benchmark X large enough that a gap of that size could have been detected?

If the reported gap is smaller than the minimum detectable effect at that
benchmark's size, the claim is not evidence of a difference. It is not wrong,
it is unsupported, and that distinction is the paper's whole argument.

You supply the claims yourself in a CSV (see
``experiments/claims_template.csv``) by reading model cards, technical report
tables and leaderboard snapshots. Do not take claims from this file on faith:
transcribe them from the source and record the source in the ``citation``
column so the audit is reproducible and defensible under review.

The discordance rate is the one input a published table never reports, because
nobody publishes item-level outputs. Two options:

  1. measure it on your own runs of the same two models (preferred, and a nice
     argument for why item-level release should be standard practice)
  2. fall back to ``assumed_discordance``, and report the audit's sensitivity
     across a range of plausible values with :func:`sensitivity_curve`
"""

from __future__ import annotations

import logging

import pandas as pd

from .power import mde_mcnemar, required_n_mcnemar

logger = logging.getLogger(__name__)

__all__ = ["REQUIRED_CLAIM_COLUMNS", "audit_claims", "sensitivity_curve", "summarise_audit"]

REQUIRED_CLAIM_COLUMNS = (
    "claim_id",
    "source_name",  # where the claim appears, e.g. a technical report table
    "citation",
    "benchmark",
    "n_items",
    "model_a",
    "model_b",
    "reported_acc_a",  # percent
    "reported_acc_b",  # percent
)


def audit_claims(
    claims: pd.DataFrame,
    assumed_discordance: float = 0.20,
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Flag each claim as powered or underpowered.

    Adds columns: ``reported_delta`` (points), ``discordance_used``, ``mde``
    (points), ``required_n``, ``powered``, and ``shortfall`` (how many times
    larger the benchmark would need to be).

    A ``discordance`` column in the input overrides ``assumed_discordance``
    per row, which is how you mix measured and assumed values.
    """
    missing = [c for c in REQUIRED_CLAIM_COLUMNS if c not in claims.columns]
    if missing:
        raise ValueError(f"claims table is missing required columns: {missing}")
    if claims.empty:
        raise ValueError("claims table is empty; fill in experiments/claims_template.csv")

    df = claims.copy()
    df["reported_delta"] = (df["reported_acc_a"] - df["reported_acc_b"]).abs()

    if "discordance" in df.columns:
        df["discordance_used"] = df["discordance"].fillna(assumed_discordance)
    else:
        df["discordance_used"] = assumed_discordance

    audited = []
    for row in df.itertuples():
        delta = row.reported_delta / 100.0
        disc = float(row.discordance_used)
        mde = mde_mcnemar(int(row.n_items), disc, alpha=alpha, power=power)

        if delta <= 0.0:
            required = None  # a reported tie needs no power to confirm
        elif delta >= disc:
            # The gap exceeds the plausible disagreement rate, so the assumed
            # discordance is too low to be consistent with the claim.
            required = None
        else:
            required = required_n_mcnemar(delta, disc, alpha=alpha, power=power)

        audited.append(
            {
                "mde_points": mde * 100.0,
                "required_n": required,
                "powered": bool(delta >= mde),
                "shortfall": (required / row.n_items) if required else float("nan"),
            }
        )

    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(audited)], axis=1)
    n_under = int((~out["powered"]).sum())
    logger.info(
        "audited %d claims: %d underpowered (%.0f%%) at discordance=%.2f",
        len(out), n_under, 100.0 * n_under / len(out), assumed_discordance,
    )
    return out


def sensitivity_curve(
    claims: pd.DataFrame,
    discordances: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30),
    alpha: float = 0.05,
    power: float = 0.80,
) -> pd.DataFrame:
    """Underpowered fraction as a function of the assumed discordance.

    Report this instead of a single headline percentage. The conclusion that
    survives every plausible discordance is the one a reviewer will accept.
    """
    rows = []
    for disc in discordances:
        audited = audit_claims(claims, assumed_discordance=disc, alpha=alpha, power=power)
        rows.append(
            {
                "discordance": disc,
                "n_claims": len(audited),
                "n_underpowered": int((~audited["powered"]).sum()),
                "frac_underpowered": float((~audited["powered"]).mean()),
                "median_mde_points": float(audited["mde_points"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarise_audit(audited: pd.DataFrame) -> pd.DataFrame:
    """Per-benchmark breakdown, which is the audit table in the paper."""
    return (
        audited.groupby("benchmark", observed=True)
        .agg(
            n_claims=("claim_id", "size"),
            n_items=("n_items", "median"),
            median_reported_gap=("reported_delta", "median"),
            median_mde=("mde_points", "median"),
            frac_underpowered=("powered", lambda s: float(1.0 - s.mean())),
        )
        .reset_index()
        .sort_values("frac_underpowered", ascending=False)
        .reset_index(drop=True)
    )
