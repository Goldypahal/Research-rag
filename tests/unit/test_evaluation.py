import pytest
import json
from unittest.mock import MagicMock, patch
from app.evaluation.synthetic_dataset_generator import SyntheticDatasetGenerator

def test_synthetic_qa_generation():
    """Verify that the generator parses LLM JSON output correctly."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json.dumps([
        {"question": "What is X?", "answer": "X is Y"}
    ])
    
    with patch("openai.resources.chat.completions.Completions.create", return_value=mock_response):
        generator = SyntheticDatasetGenerator(api_key="fake")
        qa_pairs = generator.generate_qa_pairs("This is paper text about X.")
        
        assert len(qa_pairs) == 1
        assert qa_pairs[0]["question"] == "What is X?"

def test_save_dataset_jsonl(tmp_path):
    """Verify that QA pairs are saved correctly in JSONL format."""
    dummy_qa = [{"question": "Q1", "answer": "A1"}, {"question": "Q2", "answer": "A2"}]
    dataset_path = tmp_path / "test_eval.jsonl"
    
    generator = SyntheticDatasetGenerator(api_key="fake")
    generator.save_dataset(dummy_qa, output_path=str(dataset_path))
    
    assert dataset_path.exists()
    lines = dataset_path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["question"] == "Q1"
