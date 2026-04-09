import google.generativeai as genai
import json
import logging
from pathlib import Path
from typing import List, Dict, Any
from ..core.retry_utils import retry_api_call, APITimeoutError, APIRateLimitError, APIServerError

logger = logging.getLogger(__name__)

class SyntheticDatasetGenerator:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.api_key = api_key

    def _call_llm(self, prompt: str) -> str:
        """Isolated LLM call for easier mocking in tests."""
        # Gemini 1.5 Flash supports system instructions and simplified JSON response
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            generation_config={"response_mime_type": "application/json"}
        )
        response = model.generate_content(prompt)
        return response.text

    @retry_api_call(max_attempts=3, min_wait=2, max_wait=15)
    def generate_qa_pairs(self, paper_text: str, num_questions: int = 5) -> List[Dict[str, str]]:
        """
        Generate synthetic question-answer pairs based on the text of a research paper.
        """
        prompt = f"""
You are creating evaluation questions for a research paper RAG system.
Generate {num_questions} question-answer pairs based on the text below.

Rules:
- Questions should test factual understanding (methodology, results, metrics).
- Answers must be grounded strictly in the provided text.
- Output ONLY a valid JSON list of objects with "question" and "answer" keys.

Text:
{paper_text[:8000]} # Truncate to avoid context window issues
"""
        try:
            content = self._call_llm(prompt)
            data = json.loads(content)
            # Handle variations in output format
            if isinstance(data, dict):
                for v in data.values():
                    if isinstance(v, list): return v
            return data if isinstance(data, list) else []
            
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            logger.error(f"Failed to generate synthetic QA pairs: {exc}")
            return []

    def save_dataset(self, qa_pairs: List[Dict[str, str]], output_path: str = "data/golden_eval/synthetic_qa.jsonl"):
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "a", encoding="utf-8") as f:
            for qa in qa_pairs:
                f.write(json.dumps(qa) + "\n")
        logger.info(f"Saved {len(qa_pairs)} synthetic QA pairs to {output_path}")
