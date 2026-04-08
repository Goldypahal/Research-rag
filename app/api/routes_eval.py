from fastapi import APIRouter
from ..evaluation.offline_eval import run_evaluation_task
import threading

router = APIRouter()

@router.post("/evaluate")
async def trigger_evaluation():
    # Run evaluation in background
    thread = threading.Thread(target=run_evaluation_task)
    thread.start()
    return {"status": "Evaluation started in background"}
