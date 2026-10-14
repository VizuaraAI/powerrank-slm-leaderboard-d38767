"""POWERRANK: a statistically powered leaderboard for small language models.

The one-sentence version of this project: a leaderboard that reports what its
sample size can actually resolve, instead of sorting by a mean and printing
integer ranks.

Typical session:

    from powerrank import itembank, evaluate, power, leaderboard

    bank = itembank.build_item_bank(max_per_source=50)       # small, for now
    evaluate.run_grid(bank, "runs/smoke")
    responses = evaluate.load_responses("runs/smoke")
    matrix = evaluate.response_matrix(responses)
    lb, pairs = leaderboard.build_leaderboard(matrix)
    print(leaderboard.format_leaderboard(lb))

Start with ``power.py``. You can answer the paper's headline question in one
line before you have run a single model:

    power.required_n_mcnemar(delta=0.01, discordance=0.20)   # -> ~15,700
"""

from . import audit, evaluate, itembank, leaderboard, models, power, variance

__version__ = "0.1.0"

__all__ = ["audit", "evaluate", "itembank", "leaderboard", "models", "power", "variance"]
