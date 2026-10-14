"""Command line entry points.

    powerrank power    --delta 1.0 --discordance 0.20
    powerrank bank     --out data/bank.parquet --max-per-source 50
    powerrank evaluate --bank data/bank.parquet --out runs/smoke --limit 100
    powerrank board    --runs runs/smoke
    powerrank audit    --claims experiments/claims_template.csv

Every command reads ``configs/default.yaml`` for defaults, which any flag
overrides. Run ``powerrank power`` first: it needs no GPU and no data, and it
is the fastest way to see what the project is about.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger("powerrank")


def _load_config(path: str | None) -> dict:
    if path is None:
        default = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"
        path = str(default) if default.exists() else None
    if path is None:
        return {}
    import yaml

    with open(path) as fh:
        return yaml.safe_load(fh) or {}


def cmd_power(args) -> int:
    from .power import mde_mcnemar, required_n_mcnemar, required_n_unpaired

    delta = args.delta / 100.0
    n_paired = required_n_mcnemar(delta, args.discordance, args.alpha, args.power)
    n_unpaired = required_n_unpaired(delta, args.p_bar, args.alpha, args.power)

    print(f"\n  target difference      {args.delta:.2f} percentage points")
    print(f"  discordance rate       {args.discordance:.3f}")
    print(f"  alpha / power          {args.alpha} / {args.power}\n")
    print(f"  paired (McNemar)       {n_paired:>9,} items")
    print(f"  unpaired               {n_unpaired:>9,} items per model "
          f"({2 * n_unpaired:,} total)")
    print(f"  paired is              {n_unpaired * 2 / n_paired:>9.1f}x cheaper\n")

    for n in (1_000, 2_000, 5_000, 17_248):
        mde = mde_mcnemar(n, args.discordance, args.alpha, args.power) * 100.0
        verdict = "can detect" if mde <= args.delta else "CANNOT detect"
        print(f"  at n = {n:>6,}  MDE = {mde:5.2f} pt   {verdict} a {args.delta:.1f} pt gap")
    print()
    return 0


def cmd_bank(args) -> int:
    from .itembank import build_item_bank

    bank = build_item_bank(max_per_source=args.max_per_source)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    bank.to_parquet(out, index=False)
    print(f"\n  wrote {len(bank):,} items to {out}")
    print(bank.groupby("source").size().to_string(), "\n")
    return 0


def cmd_evaluate(args) -> int:
    from .evaluate import run_grid
    from .itembank import load_item_bank
    from .models import MODEL_ZOO

    bank = load_item_bank(args.bank)
    chosen = MODEL_ZOO
    if args.models:
        wanted = set(args.models)
        chosen = tuple(m for m in MODEL_ZOO if m.name in wanted)
        missing = wanted - {m.name for m in chosen}
        if missing:
            raise SystemExit(f"unknown model(s): {sorted(missing)}")

    run_grid(
        bank,
        args.out,
        models=chosen,
        templates=tuple(args.templates),
        seeds=tuple(args.seeds),
        limit=args.limit,
    )
    print(f"\n  responses written to {args.out}\n")
    return 0


def cmd_board(args) -> int:
    from .evaluate import load_responses, response_matrix
    from .leaderboard import build_leaderboard, format_leaderboard
    from .power import paired_counts
    from .variance import decompose

    responses = load_responses(args.runs)
    print("\n" + str(decompose(responses)) + "\n")

    matrix = response_matrix(responses)
    lb, pairs = build_leaderboard(matrix, alpha=args.alpha, fdr=args.fdr)
    print(format_leaderboard(lb))

    print(f"\n  {int(pairs['significant'].sum())} of {len(pairs)} pairs separated "
          f"at FDR={args.fdr}")
    print(f"  median discordance {pairs['discordance'].median():.3f}, "
          f"median MDE {pairs['mde'].median() * 100:.2f} pt\n")

    if args.out:
        Path(args.out).mkdir(parents=True, exist_ok=True)
        lb.to_csv(Path(args.out) / "leaderboard.csv", index=False)
        pairs.to_csv(Path(args.out) / "pairwise.csv", index=False)
        print(f"  wrote tables to {args.out}\n")
    return 0


def cmd_audit(args) -> int:
    import pandas as pd

    from .audit import audit_claims, sensitivity_curve, summarise_audit

    claims = pd.read_csv(args.claims, comment="#")
    audited = audit_claims(claims, assumed_discordance=args.discordance)
    print("\n" + summarise_audit(audited).to_string(index=False))
    print("\nsensitivity to the assumed discordance rate:")
    print(sensitivity_curve(claims).to_string(index=False), "\n")
    if args.out:
        audited.to_csv(args.out, index=False)
        print(f"  wrote {args.out}\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    cfg = _load_config(None)
    pw = cfg.get("power", {})
    ev = cfg.get("evaluate", {})

    ap = argparse.ArgumentParser(prog="powerrank", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("power", help="sample size and MDE, no data needed")
    p.add_argument("--delta", type=float, default=1.0, help="gap in percentage points")
    p.add_argument("--discordance", type=float, default=pw.get("discordance", 0.20))
    p.add_argument("--p-bar", type=float, default=pw.get("p_bar", 0.60))
    p.add_argument("--alpha", type=float, default=pw.get("alpha", 0.05))
    p.add_argument("--power", type=float, default=pw.get("power", 0.80))
    p.set_defaults(func=cmd_power)

    b = sub.add_parser("bank", help="build the pooled item bank")
    b.add_argument("--out", default="data/bank.parquet")
    b.add_argument("--max-per-source", type=int, default=None)
    b.set_defaults(func=cmd_bank)

    e = sub.add_parser("evaluate", help="run the model x item x prompt x seed grid")
    e.add_argument("--bank", default="data/bank.parquet")
    e.add_argument("--out", default="runs/default")
    e.add_argument("--models", nargs="*", default=None)
    e.add_argument("--templates", nargs="+", default=ev.get("templates", ["plain", "instructed", "qa_pair"]))
    e.add_argument("--seeds", nargs="+", type=int, default=ev.get("seeds", [0]))
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_evaluate)

    d = sub.add_parser("board", help="variance components, pairwise tests, rank bands")
    d.add_argument("--runs", default="runs/default")
    d.add_argument("--alpha", type=float, default=pw.get("alpha", 0.05))
    d.add_argument("--fdr", type=float, default=pw.get("fdr", 0.05))
    d.add_argument("--out", default=None)
    d.set_defaults(func=cmd_board)

    a = sub.add_parser("audit", help="retrospective power audit of published claims")
    a.add_argument("--claims", default="experiments/claims_template.csv")
    a.add_argument("--discordance", type=float, default=pw.get("discordance", 0.20))
    a.add_argument("--out", default=None)
    a.set_defaults(func=cmd_audit)

    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
