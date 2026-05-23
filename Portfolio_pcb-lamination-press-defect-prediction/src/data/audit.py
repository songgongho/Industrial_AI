"""Basic dataset audit helpers for local development."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd


def _load_tabular(path: Path) -> pd.DataFrame:
    if path.is_dir():
        files = sorted(
            [*path.rglob("*.csv"), *path.rglob("*.parquet"), *path.rglob("*.pq")]
        )
        if not files:
            raise FileNotFoundError(f"No csv/parquet files found under {path}")
        frames = [_load_tabular(file_path) for file_path in files]
        return pd.concat(frames, ignore_index=True, sort=False)

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _label_distribution(df: pd.DataFrame) -> dict[str, int] | None:
    for candidate in ("label", "y", "target", "defect"):
        if candidate in df.columns:
            counts = df[candidate].value_counts(dropna=False).to_dict()
            return {str(key): int(value) for key, value in counts.items()}
    return None


def audit_dataset(input_path: str | Path, output_path: str | Path | None = None) -> dict[str, Any]:
    """Audit a tabular dataset and optionally write a markdown report."""

    path = Path(input_path)
    df = _load_tabular(path)

    missing_rate = (
        df.isna().mean().sort_values(ascending=False).rename("missing_rate").reset_index()
    )
    missing_rate.columns = ["column", "missing_rate"]

    report: dict[str, Any] = {
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "missing_rate": missing_rate.to_dict(orient="records"),
        "label_distribution": _label_distribution(df),
    }

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            f"# Dataset Audit: {path.name}",
            "",
            f"- Rows: {report['rows']}",
            f"- Columns: {report['columns']}",
            "",
            "## Missing rate",
            "",
            "| column | missing_rate |",
            "| --- | ---: |",
        ]
        for item in report["missing_rate"]:
            lines.append(f"| {item['column']} | {item['missing_rate']:.4f} |")
        if report["label_distribution"] is not None:
            lines.extend(["", "## Label distribution", ""])
            for key, value in report["label_distribution"].items():
                lines.append(f"- {key}: {value}")
        out.write_text("\n".join(lines), encoding="utf-8")

    return report


def main(argv: Iterable[str] | None = None) -> int:
    """CLI entry point for dataset auditing."""

    import argparse

    parser = argparse.ArgumentParser(description="Audit a tabular dataset")
    parser.add_argument("--input", required=True, help="CSV/parquet file or directory")
    parser.add_argument("--output", help="Optional markdown report path")
    args = parser.parse_args(list(argv) if argv is not None else None)

    audit_dataset(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

