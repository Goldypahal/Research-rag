from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from .api import routes_ingest, routes_query, routes_eval, routes_feedback
import uvicorn

from .core.init_system import ensure_directories

app = FastAPI(title="Research RAG Pipeline")

@app.on_event("startup")
async def startup_event():
    ensure_directories()

app.include_router(routes_ingest.router, prefix="/api/v1")
app.include_router(routes_query.router, prefix="/api/v1")
app.include_router(routes_eval.router, prefix="/api/v1")
app.include_router(routes_feedback.router, prefix="/api/v1")

# Serve UI
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
async def root():
    return FileResponse("app/static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
