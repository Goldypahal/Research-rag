import pytest
from unittest.mock import MagicMock, patch
from app.services.paper_comparator import PaperComparator
from app.services.literature_reviewer import LiteratureReviewer
from app.services.contradiction_detector import ContradictionDetector
from app.services.gap_finder import ResearchGapFinder
from app.services.timeline_generator import ResearchTimelineGenerator

@pytest.fixture
def mock_bm25():
    mock = MagicMock()
    # Mock some chunks
    chunk_mock = MagicMock()
    chunk_mock.paper_id = "paper1"
    chunk_mock.text = "This is chunk text of paper 1"
    mock.chunks = [chunk_mock]
    return mock

@patch("app.services.paper_comparator.PaperComparator._call_llm")
@patch("app.services.paper_comparator.sqlite3.connect")
def test_paper_comparator(mock_connect, mock_call, mock_bm25):
    # Mock SQLite rows return values
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor
    
    mock_cursor.fetchall.side_effect = [
        # Entities for paper1
        [{"name": "Method A", "type": "Method"}, {"name": "Dataset B", "type": "Dataset"}],
        # Relationships for paper1
        [{"src": "Method A", "src_type": "Method", "target": "Dataset B", "target_type": "Dataset", "relation_type": "evaluated_on", "description": "evaluated on dataset B"}]
    ]

    mock_call.return_value = "| Paper | Method/Model | Dataset |\n|---|---|---|\n| paper1 | Method A | Dataset B |"

    comparator = PaperComparator(bm25=mock_bm25)
    result = comparator.compare_papers(["paper1"])
    
    assert "Method A" in result
    assert "Dataset B" in result
    mock_call.assert_called_once()

@patch("app.services.literature_reviewer.LiteratureReviewer._call_llm")
@patch("app.services.paper_comparator.PaperComparator._get_paper_facts_from_db")
def test_literature_reviewer(mock_get_facts, mock_call, mock_bm25):
    mock_get_facts.return_value = "Facts of paper1"
    mock_call.return_value = "# Literature Review\n\n1. Introduction..."

    reviewer = LiteratureReviewer(bm25=mock_bm25)
    result = reviewer.generate_review(["paper1"])
    
    assert "Introduction" in result
    mock_call.assert_called_once()

@patch("app.services.contradiction_detector.ContradictionDetector._call_llm")
@patch("app.services.paper_comparator.PaperComparator._get_paper_facts_from_db")
def test_contradiction_detector(mock_get_facts, mock_call, mock_bm25):
    mock_get_facts.return_value = "Facts of paper1"
    mock_call.return_value = "# Contradictions Found\n\nNo discrepancies found."

    detector = ContradictionDetector(bm25=mock_bm25)
    result = detector.detect_contradictions(["paper1"])
    
    assert "Contradictions" in result
    mock_call.assert_called_once()

@patch("app.services.gap_finder.ResearchGapFinder._call_llm")
@patch("app.services.gap_finder.sqlite3.connect")
def test_gap_finder(mock_connect, mock_call):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.side_effect = [
        # Methods
        [{"name": "Method A", "paper_id": "paper1"}],
        # Datasets
        [{"name": "Dataset B", "paper_id": "paper2"}],
        # Evals
        [{"method": "Method A", "dataset": "Dataset B"}]
    ]
    mock_call.return_value = "Gap: Method A not yet tested on Dataset C."

    finder = ResearchGapFinder()
    result = finder.find_gaps()

    assert "Gap" in result
    mock_call.assert_called_once()

@patch("app.services.timeline_generator.ResearchTimelineGenerator._call_llm")
@patch("app.services.timeline_generator.sqlite3.connect")
def test_timeline_generator(mock_connect, mock_call):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    # Return paper titles and years, then methods
    mock_cursor.fetchall.side_effect = [
        # Papers
        [{"paper_id": "paper1", "title": "Paper One", "pub_year": 2021},
         {"paper_id": "paper2", "title": "Paper Two", "pub_year": 2023}],
        # Methods for paper1
        [{"name": "Method A"}],
        # Methods for paper2
        [{"name": "Method B"}]
    ]
    mock_call.return_value = "**2021**\n- Paper One: Method A\n**2023**\n- Paper Two: Method B"

    generator = ResearchTimelineGenerator()
    result = generator.generate_timeline()

    assert "2021" in result
    assert "2023" in result
    mock_call.assert_called_once()

@patch("sqlite3.connect")
def test_get_graph_route(mock_connect):
    mock_conn = MagicMock()
    mock_connect.return_value.__enter__.return_value = mock_conn
    mock_cursor = MagicMock()
    mock_conn.cursor.return_value = mock_cursor

    mock_cursor.fetchall.side_effect = [
        # Entities
        [{"entity_id": "e1", "name": "Paper 1", "type": "Paper"},
         {"entity_id": "e2", "name": "Method A", "type": "Method"}],
        # Relations
        [{"source_id": "e1", "target_id": "e2", "relation_type": "contains", "description": ""}]
    ]

    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)
    
    response = client.post("/api/v1/intel/graph", json={})
    assert response.status_code == 200
    data = response.json()
    assert "nodes" in data
    assert "edges" in data
    assert len(data["nodes"]) == 2
    assert len(data["edges"]) == 1
    assert data["nodes"][0]["data"]["id"] == "e1"
