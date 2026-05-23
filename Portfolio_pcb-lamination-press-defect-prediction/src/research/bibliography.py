"""Automated bibliography and paper notes management for the thesis project.

Manages:
  1. BibTeX reference generation from guide.py paper_insights()
  2. Individual markdown notes (paper/notes/{author_year}.md)
  3. Cross-linking between citations and notes
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from src.research.guide import PaperInsight, paper_insights
from datetime import datetime
from shutil import copyfile


def slugify_author_year(authors: str, year: int) -> str:
    """Convert "Author1 and Author2" + year → "author1_year.md" format.

    Parameters
    ----------
    authors : str
        Author names, typically "First et al." or "A & B" format.
    year : int
        Publication year.

    Returns
    -------
    str
        Slug in format: lastname_year
    """
    # Extract first author's last name
    first_author = authors.split(" ")[0].lower()
    first_author = re.sub(r"[^a-z0-9]", "", first_author)
    return f"{first_author}_{year}"


class BibliographyManager:
    """Manage BibTeX bibliography generation from paper insights.

    Attributes
    ----------
    insights : list[PaperInsight]
        Loaded paper insights.
    """

    def __init__(self, insights: list[PaperInsight] | None = None) -> None:
        """Initialize with paper insights (defaults to guide data).

        Parameters
        ----------
        insights : list[PaperInsight] | None
            Paper insights. If None, loads from guide.paper_insights().
        """
        self.insights = insights or paper_insights()

    def generate_bibtex_entry(self, paper: PaperInsight) -> str:
        """Generate a single BibTeX entry for a paper.

        Parameters
        ----------
        paper : PaperInsight
            Paper metadata.

        Returns
        -------
        str
            BibTeX formatted entry (no newlines at end).
        """
        slug = slugify_author_year(paper.authors, paper.year)
        # Create a minimal but valid BibTeX entry
        entry = (
            f"@article{{{slug},\n"
            f"  title={{{paper.title}}},\n"
            f"  author={{{paper.authors}}},\n"
            f"  year={{{paper.year}}},\n"
            f"  journal={{{paper.venue}}}\n"
            "}}"
        )
        return entry

    def generate_full_bibtex(self, header: str = "% BibTeX references for the semiconductor PCB lamination thesis\n\n") -> str:
        """Generate complete BibTeX file content.

        Parameters
        ----------
        header : str
            Header comment to include.

        Returns
        -------
        str
            Full BibTeX content (with newlines and header).
        """
        entries = [self.generate_bibtex_entry(p) for p in self.insights]
        return header + "\n".join(entries) + "\n"

    def write_bibtex_file(self, output_path: str | Path, overwrite: bool = False) -> None:  # pragma: no cover
        """Write BibTeX file to disk.

        Parameters
        ----------
        output_path : str | Path
            Output file path.
        overwrite : bool
            If False, skip if file exists.
        """
        output_path = Path(output_path)
        if output_path.exists() and not overwrite:
            return
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.generate_full_bibtex())


class PaperNoteGenerator:
    """Generate individual markdown notes for each paper.

    One file per paper: paper/notes/{author_year}.md
    """

    def __init__(self, notes_dir: str | Path = "paper/notes") -> None:
        """Initialize note generator.

        Parameters
        ----------
        notes_dir : str | Path
            Directory to store paper notes.
        """
        self.notes_dir = Path(notes_dir)

    @staticmethod
    def generate_note_content(paper: PaperInsight) -> str:
        """Generate markdown content for one paper.

        Parameters
        ----------
        paper : PaperInsight
            Paper metadata and analysis.

        Returns
        -------
        str
            Markdown formatted note.
        """
        slug = slugify_author_year(paper.authors, paper.year)
        md = f"""# {paper.title}

**Authors:** {paper.authors}  
**Year:** {paper.year}  
**Venue:** {paper.venue}

## What it does

{paper.what_it_does}

## Technique

{paper.technique}

## Strength

{paper.strength}

## Weakness

{paper.weakness}

## Our Response

{paper.our_response}

---

*Auto-generated from `src/research/guide.py`. Edit with care.*
"""
        return md

    def generate_all_notes(self, overwrite: bool = False) -> dict[str, Path]:  # pragma: no cover
        """Generate all paper notes.

        Parameters
        ----------
        overwrite : bool
            If False, skip existing files.

        Returns
        -------
        dict[str, Path]
            Mapping of paper slug to output path.
        """
        self.notes_dir.mkdir(parents=True, exist_ok=True)
        insights = paper_insights()
        result: dict[str, Path] = {}

        for paper in insights:
            slug = slugify_author_year(paper.authors, paper.year)
            output_path = self.notes_dir / f"{slug}.md"

            if output_path.exists() and not overwrite:
                result[slug] = output_path
                continue

            content = self.generate_note_content(paper)
            output_path.write_text(content)
            result[slug] = output_path

        return result

    def get_note_path(self, paper: PaperInsight) -> Path:
        """Get expected path for a paper note.

        Parameters
        ----------
        paper : PaperInsight
            Paper metadata.

        Returns
        -------
        Path
            Expected markdown file path.
        """
        slug = slugify_author_year(paper.authors, paper.year)
        return self.notes_dir / f"{slug}.md"


def generate_all_papers_assets(
    bibtex_path: str | Path = "paper/references.bib",
    notes_dir: str | Path = "paper/notes",
    overwrite: bool = False,
) -> dict[str, Any]:  # pragma: no cover
    """One-shot generation of all bibliography assets.

    Parameters
    ----------
    bibtex_path : str | Path
        Output BibTeX file path.
    notes_dir : str | Path
        Output notes directory.
    overwrite : bool
        If False, skip existing files.

    Returns
    -------
    dict[str, Any]
        Summary of generated files:
        - "bibtex_path": Path to .bib file
        - "notes_generated": dict of slug → Path
        - "count_papers": Total paper count
    """
    # Generate BibTeX
    bib_mgr = BibliographyManager()
    bib_mgr.write_bibtex_file(bibtex_path, overwrite=overwrite)

    # Generate paper notes
    note_gen = PaperNoteGenerator(notes_dir)
    notes_dict = note_gen.generate_all_notes(overwrite=overwrite)

    insights = paper_insights()
    return {
        "bibtex_path": Path(bibtex_path),
        "notes_generated": notes_dict,
        "count_papers": len(insights),
    }


# ---------------------------------------------------------------------------
# Additional helper APIs for note management (read / write / list)
def list_notes(notes_dir: str | Path = "paper/notes") -> list[str]:
    p = Path(notes_dir)
    if not p.exists():
        return []
    return sorted([item.name for item in p.glob("*.md") if item.is_file()])


def read_note(name: str, notes_dir: str | Path = "paper/notes") -> str:
    p = Path(notes_dir) / name
    if not p.exists():
        raise FileNotFoundError(f"Note not found: {p}")
    return p.read_text(encoding="utf-8")


def save_note(name: str, content: str, notes_dir: str | Path = "paper/notes") -> Path:
    """Save a note and create a timestamped backup of the previous version.

    Behavior:
    - If the target note already exists, copy it to `.history/{name}.{timestamp}.bak`
      before overwriting. The history directory is created automatically.
    - Then write the new content to the note file and return its path.
    """
    p = Path(notes_dir)
    p.mkdir(parents=True, exist_ok=True)
    out = p / name

    # Create history backup if file exists
    if out.exists():
        history_dir = p / ".history"
        history_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        backup_name = f"{name}.{ts}.bak"
        backup_path = history_dir / backup_name
        try:
            copyfile(out, backup_path)
        except Exception:
            # If backup fails, continue to allow saving — do not block the user
            pass

    out.write_text(content, encoding="utf-8")
    return out


def list_note_backups(name: str, notes_dir: str | Path = "paper/notes") -> list[str]:
    """List available backup filenames for a given note (sorted newest first)."""
    p = Path(notes_dir) / ".history"
    if not p.exists():
        return []
    pattern = f"{name}.*.bak"
    files = sorted([f.name for f in p.glob(pattern) if f.is_file()], reverse=True)
    return files


def restore_note_backup(name: str, backup_filename: str, notes_dir: str | Path = "paper/notes") -> Path:
    """Restore a backup (from .history) to the live note file.

    Returns the path to the restored note.
    """
    p = Path(notes_dir)
    history_dir = p / ".history"
    src = history_dir / backup_filename
    if not src.exists():
        raise FileNotFoundError(f"Backup not found: {src}")
    dst = p / name
    copyfile(src, dst)
    return dst



