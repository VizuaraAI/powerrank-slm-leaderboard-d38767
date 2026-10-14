"""Assemble the pooled item bank.

The project needs on the order of 17,000 items answerable by a 1B parameter
model and scorable without a judge model. Rather than inventing a benchmark,
we pool existing public multiple-choice and short-answer sets, de-duplicate
them, and tag every item with its source, skill and an answer key.

Nothing here ships data. Every loader pulls from the Hugging Face Hub on
first use and caches under ``HF_HOME``.

Pooling has a subtlety worth stating in the paper: items from the same source
benchmark are not independent (shared passages, shared templates, correlated
difficulty). ``source`` is carried through as a cluster id so that
``paired_bootstrap_ci`` can resample whole clusters.
"""

from __future__ import annotations

import hashlib
import logging
import re
from dataclasses import dataclass, field
from typing import Iterator

import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["Item", "SourceSpec", "DEFAULT_SOURCES", "build_item_bank", "load_item_bank"]


@dataclass(frozen=True)
class Item:
    """One scorable benchmark item."""

    item_id: str
    source: str
    skill: str
    question: str
    choices: tuple[str, ...]  # empty for free-form answers
    answer: str  # the letter for multiple choice, the string otherwise
    answer_type: str  # "mcq" or "exact"


@dataclass(frozen=True)
class SourceSpec:
    """How to pull one benchmark off the Hub and normalise it."""

    name: str
    hf_path: str
    hf_config: str | None
    split: str
    skill: str
    answer_type: str
    # Approximate item count, used only for the budget report in the README.
    approx_n: int
    # Keep at most this many items from the source. Used to stop HellaSwag's
    # 10k validation split from dominating the skill mix.
    cap: int | None = None
    hf_kwargs: dict = field(default_factory=dict)


# The default pool, matching Table 1 of the paper draft. Counts are
# approximate and become exact once you build the bank; the total is
# deliberately just over the ~15.7k items that a one-point paired comparison
# needs at a 0.20 discordance rate.
#
# MATH-500 and IFEval are in the paper's target pool but are NOT implemented
# here: both need answer-equivalence checking rather than string equality.
# Adding those two loaders is your week 2 task, and it is a good one, because
# writing the grader forces you to decide what "correct" means.
DEFAULT_SOURCES: tuple[SourceSpec, ...] = (
    SourceSpec("mmlu_pro", "TIGER-Lab/MMLU-Pro", None, "test", "knowledge", "mcq", 12032),
    SourceSpec("arc_challenge", "allenai/ai2_arc", "ARC-Challenge", "test", "science", "mcq", 1172),
    SourceSpec("hellaswag", "Rowan/hellaswag", None, "validation", "commonsense", "mcq", 1000, cap=1000),
    SourceSpec("winogrande", "allenai/winogrande", "winogrande_xl", "validation", "coreference", "mcq", 1267),
    SourceSpec("gsm8k", "openai/gsm8k", "main", "test", "math", "exact", 1319),
    SourceSpec("truthfulqa", "truthfulqa/truthful_qa", "multiple_choice", "validation", "truthfulness", "mcq", 817),
)


def _stable_id(source: str, question: str) -> str:
    digest = hashlib.sha1(f"{source}::{question}".encode("utf-8")).hexdigest()
    return f"{source}-{digest[:12]}"


def _normalise_text(text: str) -> str:
    """Lowercase, strip punctuation and collapse whitespace, for dedup only."""
    text = re.sub(r"[^\w\s]", " ", text.lower())
    return re.sub(r"\s+", " ", text).strip()


def _iter_source(spec: SourceSpec) -> Iterator[Item]:
    """Load one source and yield normalised items."""
    from datasets import load_dataset

    logger.info("loading %s (%s)", spec.name, spec.hf_path)
    args = [spec.hf_path] + ([spec.hf_config] if spec.hf_config else [])
    ds = load_dataset(*args, split=spec.split, **spec.hf_kwargs)

    for row in ds:
        parsed = _parse_row(spec, row)
        if parsed is None:
            continue
        question, choices, answer = parsed
        if not question or not answer:
            continue
        yield Item(
            item_id=_stable_id(spec.name, question),
            source=spec.name,
            skill=spec.skill,
            question=question,
            choices=tuple(choices),
            answer=answer,
            answer_type=spec.answer_type,
        )


def _parse_row(spec: SourceSpec, row: dict):
    """Map one raw row onto (question, choices, answer).

    Each public benchmark uses its own column names, so this is the one place
    with per-source special casing. Add a branch when you add a source.
    """
    letters = "ABCDEFGHIJ"

    if spec.name == "mmlu_pro":
        opts = list(row["options"])
        return row["question"], opts, letters[int(row["answer_index"])]

    if spec.name == "arc_challenge":
        opts = list(row["choices"]["text"])
        labels = list(row["choices"]["label"])
        if row["answerKey"] not in labels:
            return None
        return row["question"], opts, letters[labels.index(row["answerKey"])]

    if spec.name == "hellaswag":
        opts = list(row["endings"])
        return f"{row['ctx']}", opts, letters[int(row["label"])]

    if spec.name == "winogrande":
        opts = [row["option1"], row["option2"]]
        return row["sentence"], opts, letters[int(row["answer"]) - 1]

    if spec.name == "gsm8k":
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        return row["question"], [], answer

    if spec.name == "truthfulqa":
        targets = row["mc1_targets"]
        opts = list(targets["choices"])
        labels = list(targets["labels"])
        if 1 not in labels:
            return None
        return row["question"], opts, letters[labels.index(1)]

    raise ValueError(f"no parser registered for source {spec.name!r}")


def build_item_bank(
    sources: tuple[SourceSpec, ...] = DEFAULT_SOURCES,
    max_per_source: int | None = None,
    dedup: bool = True,
) -> pd.DataFrame:
    """Build the pooled bank and return it as a dataframe.

    ``max_per_source`` caps each source, which is what the smoke config uses
    to get a 300 item bank you can iterate on in under a minute.
    """
    items: list[Item] = []
    for spec in sources:
        limits = [n for n in (spec.cap, max_per_source) if n is not None]
        limit = min(limits) if limits else None
        taken = 0
        for item in _iter_source(spec):
            items.append(item)
            taken += 1
            if limit is not None and taken >= limit:
                break
        logger.info("  %s: %d items", spec.name, taken)

    if not items:
        raise RuntimeError("no items loaded; check your network and HF cache")

    df = pd.DataFrame([vars(i) for i in items])

    if dedup:
        before = len(df)
        df["_norm"] = df["question"].map(_normalise_text)
        df = df.drop_duplicates(subset="_norm", keep="first").drop(columns="_norm")
        logger.info("de-duplicated %d -> %d items", before, len(df))

    df = df.reset_index(drop=True)
    logger.info("item bank: %d items across %d sources", len(df), df["source"].nunique())
    return df


def load_item_bank(path: str) -> pd.DataFrame:
    """Read a bank previously written by the ``build-bank`` command."""
    df = pd.read_parquet(path)
    # choices round-trip as numpy arrays through parquet; make them tuples again
    df["choices"] = df["choices"].map(tuple)
    return df
