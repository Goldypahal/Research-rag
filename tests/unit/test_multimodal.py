from unittest.mock import MagicMock, patch, mock_open
from app.ingestion.pymupdf_parser import PyMuPDFPaperParser
from app.multimodal.figure_analyzer import FigureAnalyzer
from app.models.paper import Paper

@patch("app.ingestion.pymupdf_parser.fitz.open")
@patch("builtins.open", new_callable=mock_open, read_data=b"pdf_mock_data")
def test_pymupdf_table_and_equation_parsing(mock_file, mock_open):
    # Setup mock document
    mock_doc = MagicMock()
    mock_open.return_value = mock_doc
    mock_doc.metadata = {
        "title": "Multimodal Test",
        "author": "Author A",
        "creationDate": "D:20260719120000"
    }
    
    mock_page = MagicMock()
    mock_doc.__iter__.return_value = [mock_page]
    
    # Mock text with standard text and equation
    mock_page.get_text.return_value = "Standard line of research.\nE = mc^2 (1)\n"
    
    # Mock table
    mock_table = MagicMock()
    mock_table.extract.return_value = [
        ["ColA", "ColB"],
        ["Val1", "Val2"]
    ]
    mock_page.find_tables.return_value = [mock_table]
    
    parser = PyMuPDFPaperParser()
    paper = parser.parse("mock_paper.pdf", "mock_id")
    
    assert isinstance(paper, Paper)
    assert paper.title == "Multimodal Test"
    
    # Check elements: should contain 1 Table, 1 standard text, 1 Equation
    elements = paper.elements
    
    tables = [e for e in elements if e.element_type == "Table"]
    assert len(tables) == 1
    assert "ColA" in tables[0].text
    
    equations = [e for e in elements if e.element_type == "Equation"]
    assert len(equations) == 1
    assert "E = mc^2" in equations[0].text
    
def test_figure_analyzer_prompt():
    analyzer = FigureAnalyzer(api_key="mock_key")
    with patch.object(analyzer, "_call_vision_model") as mock_call:
        mock_call.return_value = "Mocked explanation"
        
        # Mock prepare image
        with patch.object(analyzer, "_prepare_image") as mock_prep:
            mock_prep.return_value = {"mime_type": "image/png", "data": b""}
            
            result = analyzer.analyze_figure("fig.png", "Show values", "A plot of accuracy")
            
            # Check if call_vision_model was called with the digitize instructions
            args, kwargs = mock_call.call_args
            prompt = args[0]
            assert "Chart Digitization" in prompt
            assert "Table Reasoning" in prompt
            assert "Equation Extraction" in prompt
