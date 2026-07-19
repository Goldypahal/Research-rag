import pytest
from app.indexing.chunker import HeadingTracker, SectionAwareChunker
from app.models.paper import Paper, ParsedElement

def test_heading_tracker_logic():
    tracker = HeadingTracker()
    
    # Level 1
    meta = tracker.update_if_heading("1 Introduction")
    assert meta["section"] == "1 Introduction"
    assert meta["heading_level"] == 1
    
    # Level 2
    meta = tracker.update_if_heading("1.1 Background")
    assert meta["section"] == "1 Introduction"
    assert meta["subsection"] == "1.1 Background"
    assert meta["heading_level"] == 2
    
    # Back to Level 1
    meta = tracker.update_if_heading("2 Methods")
    assert meta["section"] == "2 Methods"
    assert meta["subsection"] is None
    assert meta["heading_level"] == 1

def test_section_aware_chunker_splitting():
    # Test token overlap chunking (use_semantic_splits=False)
    chunker = SectionAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10, use_semantic_splits=False)
    
    paper = Paper(
        paper_id="test_p",
        source_path="fake.pdf",
        title="Test Paper",
        elements=[
            ParsedElement(element_type="Title", text="1 Methods", metadata={"chunk_type": "heading"}),
            ParsedElement(element_type="Text", text="This is some narrative text for testing."),
            ParsedElement(element_type="Image", text="Figure 1: Accuracy Plot", metadata={"chunk_type": "figure"}),
            ParsedElement(element_type="Title", text="2 Results", metadata={"chunk_type": "heading"}),
            ParsedElement(element_type="Text", text="This is the result section.")
        ]
    )
    
    chunks = chunker.split(paper)
    
    assert len(chunks) >= 3
    
    chunk_types = [c.chunk_type for c in chunks]
    assert "figure" in chunk_types
    assert "text" in chunk_types

def test_semantic_chunker_splitting():
    # Test semantic splitting
    chunker = SectionAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10, use_semantic_splits=True)
    
    paper = Paper(
        paper_id="test_semantic",
        source_path="fake.pdf",
        title="Test Semantic Paper",
        elements=[
            ParsedElement(element_type="Title", text="1 Introduction", metadata={"chunk_type": "heading"}),
            ParsedElement(element_type="Text", text="Deep learning models are expanding. They achieve state-of-the-art results on multiple datasets. However, they require massive computation.")
        ]
    )
    
    chunks = chunker.split(paper)
    
    # Check that we have multiple chunk levels
    levels = {c.chunk_level for c in chunks}
    assert "paragraph" in levels
    assert "sentence" in levels
    assert "section" in levels

def test_hierarchical_chunk_linking():
    chunker = SectionAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10, use_semantic_splits=False)
    
    paper = Paper(
        paper_id="test_hierarchical",
        source_path="fake.pdf",
        title="Test Hierarchical Paper",
        elements=[
            ParsedElement(element_type="Title", text="1 Methods", metadata={"chunk_type": "heading"}),
            ParsedElement(element_type="Title", text="1.1 Neural Network", metadata={"chunk_type": "heading"}),
            ParsedElement(element_type="Text", text="We design a neural network with 10 layers. The network is trained with AdamW. It converges quickly.")
        ]
    )
    
    chunks = chunker.split(paper)
    
    # Find different levels of chunks
    sentence_chunks = [c for c in chunks if c.chunk_level == "sentence"]
    paragraph_chunks = [c for c in chunks if c.chunk_level == "paragraph"]
    subsection_chunks = [c for c in chunks if c.chunk_level == "subsection"]
    section_chunks = [c for c in chunks if c.chunk_level == "section"]
    
    assert len(sentence_chunks) > 0
    assert len(paragraph_chunks) > 0
    assert len(subsection_chunks) > 0
    assert len(section_chunks) > 0
    
    # Check parent/child linkings
    p_chunk = paragraph_chunks[0]
    assert len(p_chunk.children_ids) > 0
    
    # A sentence chunk should point to the paragraph chunk as parent
    for s_id in p_chunk.children_ids:
        s_chunk = next((c for c in sentence_chunks if c.chunk_id == s_id), None)
        assert s_chunk is not None
        assert s_chunk.parent_id == p_chunk.chunk_id
        
    # Subsection chunk should point to the paragraph chunk as child
    sub_chunk = subsection_chunks[0]
    assert p_chunk.chunk_id in sub_chunk.children_ids
    assert p_chunk.parent_id == sub_chunk.chunk_id
    
    # Section chunk should point to the subsection chunk as child
    sec_chunk = section_chunks[0]
    assert sub_chunk.chunk_id in sec_chunk.children_ids
    assert sub_chunk.parent_id == sec_chunk.chunk_id
