import logging
from ..core.retry_utils import retry_api_call, APITimeoutError, APIRateLimitError, APIServerError
from ..core.settings import settings

logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, api_key: str = None):
        self._api_key = api_key or settings.GOOGLE_API_KEY
        self._client = None

    def _get_client(self):
        """Lazy-load google-genai client (new SDK)."""
        if not self._api_key:
            return None
        if self._client is None:
            import google.genai as genai
            self._client = genai.Client(api_key=self._api_key)
        return self._client

    def _prepare_image(self, image_path: str) -> dict:
        with open(image_path, "rb") as f:
            return {
                "mime_type": "image/png",
                "data": f.read()
            }

    @retry_api_call(max_attempts=3, min_wait=2, max_wait=15)
    def extract_tabular_data(self, image_path: str) -> str:
        """
        Extract raw data from a table or chart image.
        Returns a markdown table or CSV representation.
        """
        client = self._get_client()
        if not client:
            return "Data Extraction unavailable: Missing API Key."

        try:
            from google.genai import types
            image_data = self._prepare_image(image_path)
            image_part = types.Part.from_bytes(
                data=image_data["data"],
                mime_type=image_data["mime_type"],
            )
            prompt = """
You are a data extraction specialist. 
Analyze the provided image of a chart or table from a research paper.
Extract all numerical data and present it in a clean Markdown table format.
If it is a chart, estimate the values as accurately as possible.
Include units and labels for all axes/columns.
"""
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=[prompt, image_part],
            )
            return response.text
        except Exception as exc:
            logger.error(f"Data extraction failed: {exc}")
            return f"Error extracting data: {exc}"
