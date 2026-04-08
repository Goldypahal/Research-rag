import json
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
import os
import sys
import time

# Add parent dir to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.retrieval.hybrid_retriever import HybridRetriever
from app.indexing.chroma_index import ChromaIndex
from app.indexing.bm25_index import BM25Index
from app.retrieval.rerank_cohere import CohereReranker
from app.generation.answer_chain import AnswerChain
from app.retrieval.citation_enforcer import CitationEnforcer

def run_eval(golden_file="data/golden_eval/qa_pairs.jsonl"):
    chroma = ChromaIndex()
    bm25 = BM25Index()
    retriever = HybridRetriever(chroma, bm25)
    reranker = CohereReranker()
    generator = AnswerChain()
    enforcer = CitationEnforcer()
    
    data = []
    with open(golden_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    results = []
    for item in data:
        start_time = time.time()
        query = item['question']
        
        # 1. Retrieve
        retrieval_res = retriever.retrieve(query)
        raw_chunks = retrieval_res["top_chunks"]
        
        # 2. Rerank
        reranked_chunks = reranker.rerank(query, raw_chunks)
        
        # 3. Generate
        answer = generator.generate(query, reranked_chunks)
        
        # 4. Enforce Citations
        enforced = enforcer.enforce(answer, reranked_chunks)
        
        latency = time.time() - start_time
        
        results.append({
            "question": query,
            "answer": enforced['enforced_answer'],
            "contexts": [c.text for c in reranked_chunks],
            "ground_truth": item['gold_answer'],
            "latency": latency
        })
        
    # Convert to HuggingFace dataset for Ragas
    ds = Dataset.from_list(results)
    
    # Evaluate with Ragas
    score = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    print(score)
    
    # Save to CSV
    df = score.to_pandas()
    df.to_csv("data/evaluation_report.csv", index=False)

if __name__ == "__main__":
    # Create dummy golden set if not exists
    os.makedirs("data/golden_eval", exist_ok=True)
    golden_path = "data/golden_eval/qa_pairs.jsonl"
    if not os.path.exists(golden_path):
        with open(golden_path, 'w') as f:
            f.write(json.dumps({
                "question": "What is the main topic?",
                "gold_answer": "AI RAG systems",
                "paper_id": "p001",
                "gold_chunk_ids": ["c1"],
                "answerable": True
            }) + "\n")
            
    run_eval(golden_path)
