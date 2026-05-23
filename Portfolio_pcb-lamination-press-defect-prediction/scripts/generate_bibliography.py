#!/usr/bin/env python
"""CLI tool to generate bibliography and paper notes.

Usage (PowerShell):
  python scripts/generate_bibliography.py
  python scripts/generate_bibliography.py --overwrite
  python scripts/generate_bibliography.py --bibtex-only
  python scripts/generate_bibliography.py --notes-only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.research.bibliography import (
    BibliographyManager,
    PaperNoteGenerator,
    generate_all_papers_assets,
)


def main(argv: list[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="generate_bibliography.py",
        description="Generate thesis bibliography and paper notes from guide.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files (default: skip)",
    )
    parser.add_argument(
        "--bibtex-only",
        action="store_true",
        help="Generate only BibTeX file",
    )
    parser.add_argument(
        "--notes-only",
        action="store_true",
        help="Generate only markdown paper notes",
    )
    parser.add_argument(
        "--bibtex-path",
        type=str,
        default="paper/references.bib",
        help="Output BibTeX file path (default: paper/references.bib)",
    )
    parser.add_argument(
        "--notes-dir",
        type=str,
        default="paper/notes",
        help="Output notes directory (default: paper/notes)",
    )

    args = parser.parse_args(argv)

    # Validate paths
    bibtex_path = Path(args.bibtex_path)
    notes_dir = Path(args.notes_dir)

    print("[Bibliography] Starting asset generation...")

    # Generate based on flags
    if args.bibtex_only:
        print(f"[Bibliography] Generating BibTeX → {bibtex_path}")
        manager = BibliographyManager()
        manager.write_bibtex_file(bibtex_path, overwrite=args.overwrite)
        print(f"[Bibliography] BibTeX written ({len(manager.insights)} entries)")

    elif args.notes_only:
        print(f"[Bibliography] Generating paper notes → {notes_dir}")
        generator = PaperNoteGenerator(notes_dir)
        results = generator.generate_all_notes(overwrite=args.overwrite)
        print(f"[Bibliography] {len(results)} markdown notes generated")

    else:
        # Generate all (default)
        result = generate_all_papers_assets(
            bibtex_path=bibtex_path,
            notes_dir=notes_dir,
            overwrite=args.overwrite,
        )
        print(f"[Bibliography] BibTeX written → {result['bibtex_path']}")
        print(f"[Bibliography] {result['count_papers']} papers with notes")
        print(f"[Bibliography] Notes in → {notes_dir}")

    print("[Bibliography] Done!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


