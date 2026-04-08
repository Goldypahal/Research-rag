import google.generativeai as genai
import logging
from ..core.retry_utils import retry_api_call, APITimeoutError, APIRateLimitError, APIServerError
from ..core.settings import settings

logger = logging.getLogger(__name__)

class DataExtractor:
    def __init__(self, api_key: str = None):
        key = api_key or settings.GOOGLE_API_KEY
        if key:
            genai.configure(api_key=key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            self.model = None

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
        if not self.model:
            return "Data Extraction unavailable: Missing API Key."

        try:
            image_data = self._prepare_image(image_path)
            prompt = """
You are a data extraction specialist. 
Analyze the provided image of a chart or table from a research paper.
Extract all numerical data and present it in a clean Markdown table format.
If it is a chart, estimate the values as accurately as possible.
Include units and labels for all axes/columns.
"""
            response = self.model.generate_content([prompt, image_data])
            return response.text
        except Exception as exc:
            logger.error(f"Data extraction failed: {exc}")
            return f"Error extracting data: {exc}"
