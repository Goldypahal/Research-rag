import re
from typing import Dict, Any, Optional

def extract_temporal_filter(query: str) -> Optional[Dict[str, Any]]:
    """
    Extracts publication year filters from query strings.
    Supported patterns:
      - 'between YYYY and YYYY'
      - 'after YYYY', 'post-YYYY', 'post YYYY', 'later than YYYY'
      - 'before YYYY', 'prior to YYYY', 'earlier than YYYY'
      - 'since YYYY'
      - 'in YYYY'
    
    Returns a dictionary suitable for metadata filtering, e.g.,
      {"year": {"$gte": 2023}}
    """
    q = query.lower()
    
    # 1. between YYYY and YYYY
    m = re.search(r"\bbetween\s+(\d{4})\s+and\s+(\d{4})\b", q)
    if m:
        y1, y2 = int(m.group(1)), int(m.group(2))
        return {"year": {"$gte": min(y1, y2), "$lte": max(y1, y2)}}
        
    # 2. after YYYY, post-YYYY, post YYYY, later than YYYY
    m = re.search(r"\b(?:after|post-?|later\s+than)\s+(\d{4})\b", q)
    if m:
        return {"year": {"$gt": int(m.group(1))}}
        
    # 3. since YYYY
    m = re.search(r"\bsince\s+(\d{4})\b", q)
    if m:
        return {"year": {"$gte": int(m.group(1))}}
        
    # 4. before YYYY, prior to YYYY, earlier than YYYY
    m = re.search(r"\b(?:before|prior\s+to|earlier\s+than)\s+(\d{4})\b", q)
    if m:
        return {"year": {"$lt": int(m.group(1))}}
        
    # 5. in YYYY
    m = re.search(r"\bin\s+(\d{4})\b", q)
    if m:
        return {"year": int(m.group(1))}
        
    return None
