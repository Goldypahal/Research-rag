import json
from pathlib import Path
from typing import Dict, Any

class FeedbackStore:
    def __init__(self, path="data/feedback/feedback.jsonl"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, feedback_dict: Dict[str, Any]):
        # Convert datetime to string since JSON doesn't support it directly
        if "timestamp" in feedback_dict and hasattr(feedback_dict["timestamp"], "isoformat"):
            feedback_dict["timestamp"] = feedback_dict["timestamp"].isoformat()
            
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_dict) + "\n")
