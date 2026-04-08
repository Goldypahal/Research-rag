import pytest
from unittest.mock import patch, MagicMock
from app.services.query_service import QueryService
from app.retrieval.hybrid_retriever import HybridRetriever
from app.models.chunk import Chunk

@pytest.fixture
def mock_retriever():
    retriever = MagicMock(spec=HybridRetriever)
    retriever.retrieve.return_value = {
        "top_chunks": [
            Chunk(
                chunk_id="1", 
                paper_id="test_paper", 
                text="The learning rate is 0.001",
                section="Methods"
            )
        ],
        "merged_candidates": []
    }
    return retriever

@pytest.fixture
def mock_answer_chain():
    chain = MagicMock()
    chain.generate.return_value = "The paper specifies a learning rate of 0.001."
    return chain

@pytest.fixture
def mock_citation_enforcer():
    enforcer = MagicMock()
    enforcer.enforce.return_value = {
        "answer": "The paper specifies a learning rate of 0.001. [1]",
        "citations": [{"label": "Paper A", "quote": "learning rate is 0.001"}]
    }
    return enforcer

def test_query_service_integration(mock_retriever, mock_answer_chain, mock_citation_enforcer):
    """Test the orchestration of the query service."""
    service = QueryService(
        retriever=mock_retriever,
        answer_chain=mock_answer_chain,
        citation_enforcer=mock_citation_enforcer
    )
    
    result = service.ask("What is the learning rate?")
    
    assert "0.001" in result["answer"]
    assert len(result["chunks"]) == 1
    assert len(result["citations"]) == 1
    mock_retriever.retrieve.assert_called_once()
    mock_answer_chain.generate.assert_called_once()

def test_multimodal_classification():
    """Verify that figure questions trigger the vision model path."""
    with patch("app.services.query_service.is_figure_question", return_value=True):
        with patch("app.services.query_service.FigureAnalyzer") as mock_analyzer_cls:
            mock_analyzer = mock_analyzer_cls.return_value
            mock_analyzer.analyze_figure.return_value = "Vision analysis result"
            
            # Setup retriever to return a chunk with an image_path
            retriever = MagicMock()
            fig_chunk = Chunk(
                chunk_id="fig1", 
                paper_id="p1", 
                text="Fig 1", 
                chunk_type="figure", 
                image_path="fake/path.png"
            )
            retriever.retrieve.return_value = {"top_chunks": [fig_chunk]}
            
            # Mock os.path.exists to true for the image
            with patch("os.path.exists", return_value=True), \
                 patch("app.services.query_service.settings") as mock_settings:
                
                mock_settings.OPENAI_API_KEY = "fake-key"
                # Patch _log_trace to avoid JSON serialization of MagicMocks
                with patch.object(QueryService, "_log_trace"):
                    service = QueryService(retriever, MagicMock(), MagicMock())
                    # Manually ensure figure_analyzer is set if the mock_settings didn't propagate through __init__
                    service.figure_analyzer = mock_analyzer
                    
                    result = service.ask("Describe Figure 1")
                    
                    assert result.get("is_multimodal") is True
                    assert result["answer"] == "Vision analysis result"
