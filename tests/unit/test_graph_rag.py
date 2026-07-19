import os
import pytest
from unittest.mock import MagicMock, patch
from app.indexing.graph_index import SQLiteGraphIndex
from app.indexing.entity_extractor import EntityExtractor
from app.retrieval.multihop_retriever import MultiHopRetriever
from app.services.agent_planner import AgentPlanner
from app.models.paper import Paper, ParsedElement
from app.models.chunk import Chunk

@pytest.fixture
def temp_db_path(tmp_path):
    return str(tmp_path / "test_graph.db")

def test_sqlite_graph_index_crud(temp_db_path):
    index = SQLiteGraphIndex(db_path=temp_db_path)
    
    # 1. Add entities
    src_id = index.add_entity("AlphaFold", "Method", "paper1")
    target_id = index.add_entity("UniProt", "Dataset", "paper1")
    
    assert src_id == "alphafold::method::paper1"
    assert target_id == "uniprot::dataset::paper1"
    
    # 2. Add relation
    index.add_relation(src_id, target_id, "evaluated_on", "AlphaFold is evaluated on UniProt data")
    
    # 3. Retrieve neighbors
    neighbors = index.get_entity_neighbors(src_id)
    assert len(neighbors) == 1
    assert neighbors[0]["name"] == "UniProt"
    assert neighbors[0]["relation_type"] == "evaluated_on"
    assert neighbors[0]["direction"] == "outgoing"
    
    # 4. Search entity
    res = index.search_entities("Alpha")
    assert len(res) == 1
    assert res[0]["name"] == "AlphaFold"

def test_sqlite_graph_context(temp_db_path):
    index = SQLiteGraphIndex(db_path=temp_db_path)
    src_id = index.add_entity("AdamW", "Algorithm", "p1")
    target_id = index.add_entity("Transformer", "Method", "p1")
    index.add_relation(src_id, target_id, "uses", "Transformer uses AdamW optimizer")
    
    context = index.get_graph_context_for_query("What optimization is used in Transformers?")
    assert "AdamW" in context
    assert "Transformer" in context
    assert "uses" in context

@patch("app.indexing.entity_extractor.EntityExtractor._call_llm")
def test_entity_extractor_indexing(mock_call, temp_db_path):
    # Mock LLM response extracting entities/relations
    mock_call.return_value = """
    {
      "entities": [
        {"name": "ResNet-50", "type": "Method"},
        {"name": "ImageNet", "type": "Dataset"}
      ],
      "relationships": [
        {
          "source_name": "ResNet-50",
          "source_type": "Method",
          "relation_type": "evaluated_on",
          "target_name": "ImageNet",
          "target_type": "Dataset",
          "description": "ResNet-50 is trained and evaluated on ImageNet."
        }
      ]
    }
    """
    
    paper = Paper(
        paper_id="resnet_p",
        source_path="resnet.pdf",
        title="Deep Residual Learning",
        authors=["Kaiming He"],
        year=2016,
        elements=[
            ParsedElement(element_type="Text", text="We present residual learning framework. ResNet-50 is evaluated on ImageNet dataset.")
        ]
    )
    
    extractor = EntityExtractor(db_path=temp_db_path)
    results = extractor.extract_and_index_paper(paper)
    
    assert results["entities"] > 0
    assert results["relations"] > 0
    
    # Verify entity is stored in the database
    entities = extractor.db.search_entities("ResNet-50")
    assert len(entities) == 1
    assert entities[0]["type"] == "Method"

@patch("app.retrieval.multihop_retriever.MultiHopRetriever._call_llm")
def test_multihop_retriever_routing(mock_call, temp_db_path):
    # Mock planner generating follow-up query
    mock_call.return_value = "UniProt dataset details"
    
    # Mock hybrid retriever
    mock_hybrid = MagicMock()
    mock_hybrid.final_top_k = 2
    mock_hybrid.retrieve.side_effect = [
        # First hop response
        {
            "top_chunks": [
                Chunk(chunk_id="c1", paper_id="p1", text="AlphaFold uses UniProt references.", chunk_level="paragraph")
            ],
            "merged_candidates": []
        },
        # Second hop response
        {
            "top_chunks": [
                Chunk(chunk_id="c2", paper_id="p2", text="UniProt contains protein sequences.", chunk_level="paragraph")
            ],
            "merged_candidates": []
        }
    ]
    
    multihop = MultiHopRetriever(hybrid_retriever=mock_hybrid, db_path=temp_db_path)
    
    # Set up some facts in the Graph DB to pull
    multihop.graph_db.add_entity("AlphaFold", "Method", "p1")
    
    res = multihop.retrieve_multi_hop("What database did AlphaFold train on?", max_hops=2)
    
    assert len(res["top_chunks"]) >= 2
    assert "c1" in [c.chunk_id for c in res["top_chunks"]]
    assert "c2" in [c.chunk_id for c in res["top_chunks"]]
    assert len(res["follow_up_queries"]) == 1
    assert res["follow_up_queries"][0] == "UniProt dataset details"

@patch("app.services.agent_planner.AgentPlanner._call_llm")
def test_agent_planner_routing(mock_call):
    mock_call.return_value = "Routed to compare path"
    
    planner = AgentPlanner()
    route = planner.plan_route("Compare ResNet and Transformer performance")
    
    assert route == "compare"
