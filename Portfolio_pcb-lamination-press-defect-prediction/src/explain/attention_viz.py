"""Attention visualization helpers skeleton."""

from __future__ import annotations

from pathlib import Path


def save_attention_placeholder(output_path: str | Path) -> Path:
    """Create a placeholder file for future attention heatmaps."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("Attention visualization will be generated here.\n", encoding="utf-8")
    return path

