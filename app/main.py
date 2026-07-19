import os
import gc

# Restrict multithreading to minimize memory footprint on resource-constrained containers (like Render)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api import routes_ingest, routes_query, routes_eval, routes_feedback, routes_intel
import uvicorn

from .core.init_system import ensure_directories
from .core.connectivity import mode_manager

app = FastAPI(title="Research RAG Pipeline")

@app.on_event("startup")
async def startup_event():
    ensure_directories()
    # Detect internet and set initial mode (local if offline, user-pref if online)
    mode_manager.detect_and_set()
    
    # Run auto-ingest in a background thread to prevent startup block
    import threading
    from .api.routes_query import chroma, bm25
    from .core.auto_ingest import run_auto_ingest_background
    
    threading.Thread(
        target=run_auto_ingest_background,
        args=(chroma, bm25),
        daemon=True
    ).start()
    
    # Run garbage collection to free import-time memory
    gc.collect()

app.include_router(routes_ingest.router, prefix="/api/v1")
app.include_router(routes_query.router, prefix="/api/v1")
app.include_router(routes_eval.router, prefix="/api/v1")
app.include_router(routes_feedback.router, prefix="/api/v1")
app.include_router(routes_intel.router, prefix="/api/v1/intel")

# Serve UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
