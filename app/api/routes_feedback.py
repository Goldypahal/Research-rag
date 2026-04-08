from fastapi import APIRouter
from ..models.feedback import Feedback
from ..storage.feedback_store import FeedbackStore

router = APIRouter()
store = FeedbackStore()

@router.post("/feedback")
async def submit_feedback(feedback: Feedback):
    """
    User submits 👍 / 👎 feedback on answer.
    """
    store.save(feedback.model_dump())
    return {"status": "feedback stored", "rating": feedback.rating}
