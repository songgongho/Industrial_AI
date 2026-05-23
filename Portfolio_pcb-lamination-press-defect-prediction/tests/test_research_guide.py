from src.research.guide import build_up_items, glossary_terms, paper_insights, project_direction, security_principles


def test_project_direction_contains_planner_friendly_summary() -> None:
    direction = project_direction()
    assert "멀티모달" in direction.one_liner
    assert "불량" in direction.problem
    assert "보안" in " ".join(direction.security_notes)
    assert len(direction.next_ideas) >= 3


def test_paper_insights_cover_core_methods() -> None:
    papers = paper_insights()
    titles = {paper.title for paper in papers}
    assert any("Temporal Fusion" in title for title in titles)
    assert any("SHAP" in title for title in titles)
    assert any("Grad-CAM" in title for title in titles)
    assert len(papers) >= 6


def test_glossary_and_security_are_populated() -> None:
    glossary = glossary_terms()
    security = security_principles()
    build_up = build_up_items()

    assert any(item.term == "DVC" for item in glossary)
    assert any(item.term == "MLflow" for item in glossary)
    assert any(rule.title == "비밀 정보는 코드에 쓰지 않기" for rule in security)
    assert any(item.title == "참고문헌 관리" for item in build_up)

