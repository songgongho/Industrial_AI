"""Generate a preprocessing template YAML from a dataset analysis.

Usage:
  python scripts/generate_preprocess_template.py --source data/raw/secom
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    import sys
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.data.dataset_inspector import analyze_dataset_source


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="generate_preprocess_template.py")
    p.add_argument("--source", required=True, help="파일 또는 폴더 경로")
    p.add_argument("--output-dir", default="configs/preprocess_templates", help="템플릿 저장 디렉토리")
    return p


def _render_yaml(template: dict[str, Any]) -> str:
    # Simple YAML emitter without external dependency
    lines: list[str] = []
    def write(k: str, v: Any, indent: int = 0) -> None:
        prefix = "  " * indent
        if isinstance(v, dict):
            lines.append(f"{prefix}{k}:")
            for kk, vv in v.items():
                write(kk, vv, indent + 1)
        elif isinstance(v, list):
            lines.append(f"{prefix}{k}:")
            for item in v:
                if isinstance(item, dict):
                    lines.append(f"{prefix}-")
                    for kk, vv in item.items():
                        write(kk, vv, indent + 2)
                else:
                    lines.append(f"{prefix}- {item}")
        else:
            lines.append(f"{prefix}{k}: {v}")

    for key, val in template.items():
        write(key, val, 0)
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    source = Path(args.source)
    analysis = analyze_dataset_source(source)

    # Build a minimal template
    template: dict[str, Any] = {
        "source": str(analysis.source),
        "type": analysis.source_type,
        "columns": [],
        "rules": {
            "drop_duplicates": True,
            "numeric_impute": "median",
            "categorical_impute": "mode_or_missing",
        },
    }

    if analysis.tabular_shape and analysis.dtype_frame is not None:
        for col in analysis.column_names:
            dtype = str(analysis.dtype_frame.loc[col, "dtype"]) if (col in analysis.dtype_frame.index) else "unknown"
            template["columns"].append({"name": col, "dtype": dtype})

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = (analysis.source.stem or "preprocess_template").replace(".", "_")
    out_path = out_dir / f"{stem}.yaml"
    out_path.write_text(_render_yaml(template), encoding="utf-8")
    print(f"Wrote preprocess template: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

