"""Tests for bibliography and paper notes management."""

from pathlib import Path

import pytest

from src.research.bibliography import (
    BibliographyManager,
    PaperNoteGenerator,
    slugify_author_year,
)
from src.research.guide import PaperInsight, paper_insights


class TestSlugifyAuthorYear:
    """Test author_year slug generation."""

    def test_single_author(self) -> None:
        """Test single author name."""
        assert slugify_author_year("Smith", 2020) == "smith_2020"

    def test_multiple_authors_et_al(self) -> None:
        """Test 'et al.' format."""
        assert slugify_author_year("Lim et al.", 2021) == "lim_2021"

    def test_multiple_authors_and(self) -> None:
        """Test 'Author1 & Author2' format."""
        assert slugify_author_year("Lundberg & Lee", 2017) == "lundberg_2017"

    def test_removes_special_chars(self) -> None:
        """Test that special characters are removed."""
        assert slugify_author_year("O'Brien-Smith", 2019) == "obriensmith_2019"

    def test_lowercase_normalization(self) -> None:
        """Test that output is lowercase."""
        assert slugify_author_year("UPPERCASE", 2020) == "uppercase_2020"


class TestBibliographyManager:
    """Test BibTeX generation."""

    def test_generate_bibtex_entry_valid(self) -> None:
        """Test single BibTeX entry generation."""
        paper = PaperInsight(
            title="Test Paper",
            authors="TestAuthor",
            year=2020,
            venue="Test Journal",
            what_it_does="Does something",
            technique="Some technique",
            strength="Very strong",
            weakness="Not very weak",
            our_response="We respond well",
        )
        mgr = BibliographyManager([paper])
        entry = mgr.generate_bibtex_entry(paper)

        assert "@article{testauthor_2020," in entry
        assert "title={Test Paper}" in entry
        assert "author={TestAuthor}" in entry
        assert "year={2020}" in entry
        assert "journal={Test Journal}" in entry

    def test_generate_full_bibtex_multiple(self) -> None:
        """Test generation of multiple entries."""
        papers = [
            PaperInsight(
                title="Paper 1", authors="A", year=2020, venue="V1",
                what_it_does="x", technique="y", strength="z", weakness="w", our_response="r"
            ),
            PaperInsight(
                title="Paper 2", authors="B", year=2021, venue="V2",
                what_it_does="x", technique="y", strength="z", weakness="w", our_response="r"
            ),
        ]
        mgr = BibliographyManager(papers)
        final = mgr.generate_full_bibtex()

        assert "@article{a_2020," in final
        assert "@article{b_2021," in final
        assert final.startswith("% BibTeX")

    def test_full_bibtex_has_header(self) -> None:
        """Test that header is included."""
        mgr = BibliographyManager([])
        final = mgr.generate_full_bibtex()
        assert final.startswith("% BibTeX")

    def test_manager_loads_default_insights(self) -> None:
        """Test that default loads guide insights."""
        mgr = BibliographyManager()
        assert len(mgr.insights) > 0
        # Should have at least Lim et al.
        assert any("Lim" in p.authors for p in mgr.insights)


class TestPaperNoteGenerator:
    """Test markdown paper notes generation."""

    def test_generate_note_content_structure(self) -> None:
        """Test markdown note structure."""
        paper = PaperInsight(
            title="Example Paper",
            authors="Example Author",
            year=2022,
            venue="Example Venue",
            what_it_does="Does X",
            technique="Tech Y",
            strength="Strong Z",
            weakness="Weak W",
            our_response="Response R",
        )
        gen = PaperNoteGenerator()
        content = gen.generate_note_content(paper)

        assert "# Example Paper" in content
        assert "**Authors:** Example Author" in content
        assert "**Year:** 2022" in content
        assert "**Venue:** Example Venue" in content
        assert "## What it does" in content
        assert "## Technique" in content
        assert "## Strength" in content
        assert "## Weakness" in content
        assert "## Our Response" in content
        assert "Does X" in content
        assert "Tech Y" in content

    def test_get_note_path(self) -> None:
        """Test correct path generation for notes."""
        paper = PaperInsight(
            title="T", authors="Lim et al.", year=2021, venue="V",
            what_it_does="x", technique="y", strength="z", weakness="w", our_response="r"
        )
        gen = PaperNoteGenerator("custom/notes")
        path = gen.get_note_path(paper)

        assert path == Path("custom/notes") / "lim_2021.md"

    def test_default_notes_directory(self) -> None:
        """Test default notes directory."""
        gen = PaperNoteGenerator()
        assert gen.notes_dir == Path("paper/notes")

    def test_note_content_is_valid_markdown(self) -> None:
        """Test that generated content follows markdown conventions."""
        paper = PaperInsight(
            title="T", authors="A", year=2020, venue="V",
            what_it_does="x", technique="y", strength="z", weakness="w", our_response="r"
        )
        gen = PaperNoteGenerator()
        content = gen.generate_note_content(paper)

        # Check markdown structure
        lines = content.split("\n")
        assert lines[0].startswith("# ")  # Title is h1
        assert any(line.startswith("**") for line in lines)  # Has bold
        assert any(line.startswith("## ") for line in lines)  # Has h2 sections


class TestIntegration:
    """Integration tests for bibliography system."""

    def test_manager_and_generator_consistency(self) -> None:
        """Test that manager and generator use consistent slugs."""
        paper = PaperInsight(
            title="T", authors="TestAuth", year=2020, venue="V",
            what_it_does="x", technique="y", strength="z", weakness="w", our_response="r"
        )
        mgr = BibliographyManager([paper])
        gen = PaperNoteGenerator()

        entry = mgr.generate_bibtex_entry(paper)
        path = gen.get_note_path(paper)

        # Both should use same slug
        assert "testauth_2020" in entry
        assert "testauth_2020.md" in str(path)

    def test_all_guide_papers_have_valid_entries(self) -> None:
        """Test that all papers from guide can be converted."""
        mgr = BibliographyManager()
        gen = PaperNoteGenerator()

        for paper in mgr.insights:
            # Should not raise
            entry = mgr.generate_bibtex_entry(paper)
            content = gen.generate_note_content(paper)
            path = gen.get_note_path(paper)

            assert len(entry) > 0
            assert len(content) > 100
            assert ".md" in str(path)

