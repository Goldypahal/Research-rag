import pytest
from unittest.mock import MagicMock, patch
from app.retrieval.temporal_parser import extract_temporal_filter
from app.retrieval.adaptive_retriever import AdaptiveRetrieverConfig, AdaptiveRetriever
from app.retrieval.query_decomposer import QueryDecomposer

def test_extract_temporal_filter():
    # Test 'after YYYY'
    f = extract_temporal_filter("latest transformer methods after 2023")
    assert f == {"year": {"$gt": 2023}}
    
    # Test 'since YYYY'
    f = extract_temporal_filter("deep learning papers since 2020")
    assert f == {"year": {"$gte": 2020}}
    
    # Test 'between YYYY and YYYY'
    f = extract_temporal_filter("papers between 2018 and 2022")
    assert f == {"year": {"$gte": 2018, "$lte": 2022}}
    
    # Test 'in YYYY'
    f = extract_temporal_filter("BERT publication in 2019")
    assert f == {"year": 2019}
    
    # Test 'before YYYY'
    f = extract_temporal_filter("machine learning models before 2015")
    assert f == {"year": {"$lt": 2015}}
    
    # Test query without any year
    f = extract_temporal_filter("how to train ResNet-50")
    assert f is None

def test_adaptive_retriever_config():
    assert AdaptiveRetrieverConfig.get_limits("definition") == (3, 10, 10, 8)
    assert AdaptiveRetrieverConfig.get_limits("equation") == (8, 15, 15, 12)
    assert AdaptiveRetrieverConfig.get_limits("methodology") == (8, 15, 15, 12)
    assert AdaptiveRetrieverConfig.get_limits("comparison") == (15, 30, 30, 20)
    assert AdaptiveRetrieverConfig.get_limits("literature review") == (25, 40, 40, 30)
    # Default fallback
    assert AdaptiveRetrieverConfig.get_limits("invalid_intent") == (8, 20, 20, 15)

@patch("app.retrieval.query_decomposer.QueryDecomposer._call_llm")
def test_query_decomposer_success(mock_call):
    # Mock LLM returning valid JSON array
    mock_call.return_value = '```json\n["Query A", "Query B"]\n```'
    
    decomposer = QueryDecomposer()
    queries = decomposer.decompose("Some multi-part query")
    
    assert queries == ["Query A", "Query B"]
    mock_call.assert_called_once()

@patch("app.retrieval.query_decomposer.QueryDecomposer._call_llm")
def test_query_decomposer_fallback(mock_call):
    # Mock LLM returning malformed JSON or error
    mock_call.return_value = 'not a json'
    
    decomposer = QueryDecomposer()
    queries = decomposer.decompose("Original Query")
    
    # Should fallback to original query
    assert queries == ["Original Query"]

@patch("app.retrieval.adaptive_retriever.AdaptiveRetriever._call_llm")
def test_adaptive_retriever_success(mock_call):
    # Mock LLM returning a valid intent string
    mock_call.return_value = "intent is COMPARISON"
    
    retriever = AdaptiveRetriever()
    limits = retriever.get_retrieval_limits("Compare paper X and Y")
    
    assert limits["intent"] == "comparison"
    assert limits["final_top_k"] == 15

@patch("app.retrieval.adaptive_retriever.AdaptiveRetriever._call_llm")
def test_adaptive_retriever_fallback(mock_call):
    # Mock LLM returning unrecognized response
    mock_call.return_value = "unknown garbage"
    
    retriever = AdaptiveRetriever()
    limits = retriever.get_retrieval_limits("What is SGD?")
    
    # Should fallback to 'methodology' intent config
    assert limits["intent"] == "methodology"
    assert limits["final_top_k"] == 8
