from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional
from ..services.paper_comparator import PaperComparator
from ..services.literature_reviewer import LiteratureReviewer
from ..services.contradiction_detector import ContradictionDetector
from ..services.gap_finder import ResearchGapFinder
from ..services.timeline_generator import ResearchTimelineGenerator
from ..indexing.bm25_index import BM25Index

router = APIRouter()
bm25 = BM25Index()

# Instantiate services
comparator = PaperComparator(bm25=bm25)
reviewer = LiteratureReviewer(bm25=bm25)
detector = ContradictionDetector(bm25=bm25)
gap_finder = ResearchGapFinder()
timeline_gen = ResearchTimelineGenerator()

class IntelRequest(BaseModel):
    paper_ids: Optional[List[str]] = None
    prompt_version: str = "v1"

@router.post("/compare")
async def compare_papers(request: IntelRequest):
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one paper_id for comparison.")
    result = comparator.compare_papers(request.paper_ids, prompt_version=request.prompt_version)
    return {"comparison": result}

@router.post("/review")
async def generate_literature_review(request: IntelRequest):
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one paper_id for review.")
    result = reviewer.generate_review(request.paper_ids, prompt_version=request.prompt_version)
    return {"review": result}

@router.post("/contradictions")
async def detect_contradictions(request: IntelRequest):
    if not request.paper_ids:
        raise HTTPException(status_code=400, detail="Must provide at least one paper_id for contradiction detection.")
    result = detector.detect_contradictions(request.paper_ids, prompt_version=request.prompt_version)
    return {"contradictions": result}

@router.post("/gaps")
async def find_research_gaps(request: IntelRequest):
    result = gap_finder.find_gaps(request.paper_ids, prompt_version=request.prompt_version)
    return {"gaps": report_clean_response(result)}

@router.post("/timeline")
async def generate_timeline(request: IntelRequest):
    result = timeline_gen.generate_timeline(request.paper_ids, prompt_version=request.prompt_version)
    return {"timeline": result}

@router.post("/graph")
async def get_graph(request: Optional[IntelRequest] = None):
    import sqlite3
    from ..indexing.graph_index import SQLiteGraphIndex
    db = SQLiteGraphIndex()
    
    nodes = []
    edges = []
    
    with sqlite3.connect(db.db_path) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # If paper_ids filter is provided, select subset, else select all
        if request and request.paper_ids:
            placeholders = ",".join("?" for _ in request.paper_ids)
            # Fetch entities belonging to these papers
            cursor.execute(f"SELECT entity_id, name, type FROM entities WHERE paper_id IN ({placeholders})", request.paper_ids)
            rows = cursor.fetchall()
            for r in rows:
                nodes.append({
                    "data": {
                        "id": r["entity_id"],
                        "label": r["name"],
                        "type": r["type"]
                    }
                })
            
            # Fetch relations where source or target belongs to these papers
            cursor.execute(f"""
                SELECT DISTINCT r.source_id, r.target_id, r.relation_type, r.description
                FROM relations r
                JOIN entities e1 ON r.source_id = e1.entity_id
                JOIN entities e2 ON r.target_id = e2.entity_id
                WHERE e1.paper_id IN ({placeholders}) OR e2.paper_id IN ({placeholders})
            """, request.paper_ids * 2)
            rows = cursor.fetchall()
            for r in rows:
                edges.append({
                    "data": {
                        "source": r["source_id"],
                        "target": r["target_id"],
                        "label": r["relation_type"],
                        "description": r["description"] or ""
                    }
                })
        else:
            # Query all entities
            cursor.execute("SELECT entity_id, name, type FROM entities")
            rows = cursor.fetchall()
            for r in rows:
                nodes.append({
                    "data": {
                        "id": r["entity_id"],
                        "label": r["name"],
                        "type": r["type"]
                    }
                })
                
            # Query all relations
            cursor.execute("SELECT source_id, target_id, relation_type, description FROM relations")
            rows = cursor.fetchall()
            for r in rows:
                edges.append({
                    "data": {
                        "source": r["source_id"],
                        "target": r["target_id"],
                        "label": r["relation_type"],
                        "description": r["description"] or ""
                    }
                })
                
    return {"nodes": nodes, "edges": edges}

def report_clean_response(text: str) -> str:
    """Helper to clean response wrapping."""
    return text.strip()
