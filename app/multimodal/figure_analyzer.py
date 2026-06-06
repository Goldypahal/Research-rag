import logging
from ..core.retry_utils import retry_api_call, APITimeoutError, APIRateLimitError, APIServerError

logger = logging.getLogger(__name__)

class FigureAnalyzer:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._client = None

    def _get_client(self):
        """Lazy-load the google-genai client (new SDK, replaces google.generativeai)."""
        if self._client is None:
            import google.genai as genai
            self._client = genai.Client(api_key=self.api_key)
        return self._client

    def _prepare_image(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            return {
                "mime_type": "image/png",
                "data": f.read()
            }

    def _call_vision_model(self, prompt: str, image_data: dict) -> str:
        """Isolated Gemini vision call for easier mocking."""
        import google.genai as genai
        from google.genai import types
        client = self._get_client()
        image_part = types.Part.from_bytes(
            data=image_data["data"],
            mime_type=image_data["mime_type"],
        )
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[prompt, image_part],
        )
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
