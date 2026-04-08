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
    chunker = SectionAwareChunker(chunk_size_tokens=100, chunk_overlap_tokens=10)
    
    # Mock paper with sections and text
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
    
    # Should have separated figure and text, and respected sections
    assert len(chunks) >= 3
    
    chunk_types = [c.chunk_type for c in chunks]
    assert "figure" in chunk_types
    assert "text" in chunk_types
    
    # Check section assignment
    method_chunks = [c for c in chunks if c.section == "1 Methods"]
    assert len(method_chunks) > 0
    assert any("testing" in c.text for c in method_chunks)
