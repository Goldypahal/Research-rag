import base64
import logging
import google.generativeai as genai
from ..core.retry_utils import retry_api_call, APITimeoutError, APIRateLimitError, APIServerError

logger = logging.getLogger(__name__)

class FigureAnalyzer:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")

    def _prepare_image(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            return {
                "mime_type": "image/png", # Assume png or handle dynamically
                "data": f.read()
            }

    def _call_vision_model(self, prompt: str, image_data: dict) -> str:
        """Isolated Gemini vision call for easier mocking."""
        response = self.model.generate_content([prompt, image_data])
        return response.text

    @retry_api_call(max_attempts=3, min_wait=1, max_wait=10)
    def analyze_figure(self, image_path: str, question: str, caption: str = None) -> str:
        try:
            image_data = self._prepare_image(image_path)

            prompt = f"""
You are a scientific research assistant.
A user asked a question about a figure in a research paper.

Figure caption:
{caption or "No caption provided."}

User question:
{question}

Explain what the chart or table shows.
Focus on trends, patterns, comparisons, and conclusions.
"""

            return self._call_vision_model(prompt, image_data)
        except Exception as exc:
            msg = str(exc).lower()
            if "timeout" in msg:
                raise APITimeoutError(str(exc)) from exc
            if "429" in msg or "rate" in msg:
                raise APIRateLimitError(str(exc)) from exc
            if any(x in msg for x in ["500", "502", "503", "504", "server"]):
                raise APIServerError(str(exc)) from exc
            raise
