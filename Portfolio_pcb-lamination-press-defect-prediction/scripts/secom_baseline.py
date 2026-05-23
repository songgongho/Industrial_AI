"""SECOM baseline evaluation CLI.

Example:
  python scripts/secom_baseline.py --data-dir data/raw/secom --target-length 128
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.baselines.secom import (  # noqa: E402
    evaluate_secom_baseline,
    format_secom_baseline_report,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="secom_baseline.py")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "raw" / "secom")
    parser.add_argument("--target-length", type=int, default=128)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cost-fn", type=float, default=100.0)
    parser.add_argument("--cost-fp", type=float, default=5.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--output", type=Path, help="Optional markdown report path")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = evaluate_secom_baseline(
            args.data_dir,
            target_length=args.target_length,
            test_size=args.test_size,
            seed=args.seed,
            cost_fn=args.cost_fn,
            cost_fp=args.cost_fp,
            max_iter=args.max_iter,
        )
    except Exception as exc:
        print(f"SECOM baseline failed: {exc}")
        return 1

    text = format_secom_baseline_report(
        result,
        data_dir=args.data_dir,
        target_length=args.target_length,
    )
    print(text)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

