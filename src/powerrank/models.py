"""The small language models under test, and how to score an item with one.

Two scoring modes, both standard in the evaluation literature:

  loglikelihood : for multiple choice, score each option by its length-
                  normalised log probability and pick the argmax. No parsing,
                  no refusals, works on base models. This is what
                  lm-evaluation-harness calls the ``acc_norm`` path.
  generate      : for exact-answer items (GSM8K), sample a continuation and
                  extract the final number.

Everything runs through Hugging Face ``transformers``. Swap in vLLM for the
full 5M forward pass run if you have it available; the ``ScoredItem``
interface is deliberately narrow so that backend is a drop-in replacement.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

import torch

logger = logging.getLogger(__name__)

__all__ = ["ModelSpec", "MODEL_ZOO", "PROMPT_TEMPLATES", "ScoredItem", "SLMRunner"]


@dataclass(frozen=True)
class ModelSpec:
    name: str  # short name used in the leaderboard
    hf_id: str
    params_b: float  # billions of parameters, for the cost-vs-power plot


# A 0.5B to 4B field. These are the models the leaderboard ranks; all are
# public on the Hub and all fit on a single 16GB GPU in bfloat16.
MODEL_ZOO: tuple[ModelSpec, ...] = (
    ModelSpec("Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B-Instruct", 0.5),
    ModelSpec("Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct", 1.5),
    ModelSpec("Qwen2.5-3B-Instruct", "Qwen/Qwen2.5-3B-Instruct", 3.1),
    ModelSpec("Llama-3.2-1B-Instruct", "meta-llama/Llama-3.2-1B-Instruct", 1.2),
    ModelSpec("Llama-3.2-3B-Instruct", "meta-llama/Llama-3.2-3B-Instruct", 3.2),
    ModelSpec("SmolLM2-1.7B-Instruct", "HuggingFaceTB/SmolLM2-1.7B-Instruct", 1.7),
    ModelSpec("SmolLM2-360M-Instruct", "HuggingFaceTB/SmolLM2-360M-Instruct", 0.36),
    ModelSpec("Phi-3-mini-4k-Instruct", "microsoft/Phi-3-mini-4k-instruct", 3.8),
    ModelSpec("gemma-2-2b-it", "google/gemma-2-2b-it", 2.6),
    ModelSpec("TinyLlama-1.1B-Chat", "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 1.1),
)

# The prompt axis of the response tensor. These are paraphrases that a
# reasonable person would consider equivalent, which is exactly the point:
# if the score moves, the score was never a property of the model alone.
PROMPT_TEMPLATES: dict[str, str] = {
    "plain": "{question}\n{options}\nAnswer:",
    "instructed": "Answer the following question. Reply with only the letter of the correct option.\n\n{question}\n{options}\nAnswer:",
    "qa_pair": "Question: {question}\nOptions:\n{options}\nThe correct answer is:",
    "expert": "You are an expert. Choose the best option.\n\n{question}\n{options}\nAnswer:",
    "terse": "{question}\n{options}\n->",
}

_LETTERS = "ABCDEFGHIJ"


@dataclass(frozen=True)
class ScoredItem:
    item_id: str
    predicted: str
    correct: int


def format_options(choices: tuple[str, ...]) -> str:
    return "\n".join(f"{_LETTERS[i]}. {c}" for i, c in enumerate(choices))


def render_prompt(template_name: str, question: str, choices: tuple[str, ...]) -> str:
    if template_name not in PROMPT_TEMPLATES:
        raise KeyError(
            f"unknown template {template_name!r}; available: {sorted(PROMPT_TEMPLATES)}"
        )
    return PROMPT_TEMPLATES[template_name].format(
        question=question, options=format_options(choices)
    )


def extract_number(text: str) -> str | None:
    """Pull the last number out of a generation, GSM8K style."""
    matches = re.findall(r"-?\d[\d,]*\.?\d*", text)
    if not matches:
        return None
    return matches[-1].replace(",", "").rstrip(".")


class SLMRunner:
    """Loads one model and scores items against it.

    Use as a context manager so the weights are freed between models; a 20
    model sweep will otherwise exhaust GPU memory.
    """

    def __init__(
        self,
        spec: ModelSpec,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_new_tokens: int = 256,
    ) -> None:
        self.spec = spec
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = getattr(torch, dtype)
        self.max_new_tokens = max_new_tokens
        self.model = None
        self.tokenizer = None

    def __enter__(self) -> "SLMRunner":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("loading %s on %s", self.spec.hf_id, self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.spec.hf_id, dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        return self

    def __exit__(self, *exc) -> None:
        self.model = None
        self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _require_loaded(self) -> None:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("use SLMRunner as a context manager: `with SLMRunner(spec) as r:`")

    @torch.no_grad()
    def score_mcq(self, prompt: str, choices: tuple[str, ...]) -> str:
        """Return the predicted letter by length-normalised log likelihood."""
        self._require_loaded()
        scores = []
        for i, choice in enumerate(choices):
            continuation = f" {_LETTERS[i]}. {choice}"
            full = self.tokenizer(prompt + continuation, return_tensors="pt").to(self.device)
            prompt_len = self.tokenizer(prompt, return_tensors="pt")["input_ids"].shape[1]

            logits = self.model(**full).logits
            labels = full["input_ids"]
            # shift so that logits[t] predicts labels[t+1]
            logprobs = torch.log_softmax(logits[:, :-1], dim=-1)
            target = labels[:, 1:]
            token_lp = logprobs.gather(2, target.unsqueeze(-1)).squeeze(-1)

            cont_lp = token_lp[:, prompt_len - 1 :]
            n_tokens = max(1, cont_lp.shape[1])
            scores.append(float(cont_lp.sum()) / n_tokens)

        return _LETTERS[int(max(range(len(scores)), key=scores.__getitem__))]

    @torch.no_grad()
    def generate_answer(self, prompt: str, seed: int = 0, temperature: float = 0.0) -> str:
        """Greedy by default; pass a temperature above zero to exercise the seed axis."""
        self._require_loaded()
        torch.manual_seed(seed)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=temperature > 0.0,
            temperature=temperature if temperature > 0.0 else None,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        return self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    def score_item(self, row, template: str, seed: int, temperature: float = 0.0) -> ScoredItem:
        """Score a single item bank row. Dispatches on ``answer_type``."""
        if row.answer_type == "mcq":
            prompt = render_prompt(template, row.question, tuple(row.choices))
            predicted = self.score_mcq(prompt, tuple(row.choices))
        elif row.answer_type == "exact":
            prompt = f"{row.question}\nThink step by step, then give the final number.\nAnswer:"
            raw = self.generate_answer(prompt, seed=seed, temperature=temperature)
            predicted = extract_number(raw) or ""
        else:
            raise ValueError(f"unknown answer_type {row.answer_type!r}")
        return ScoredItem(
            item_id=row.item_id,
            predicted=predicted,
            correct=int(predicted == row.answer),
        )
