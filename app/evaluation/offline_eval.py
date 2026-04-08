import json
import os
import pandas as pd
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from datasets import Dataset
from ..core.logging import logger
from ..indexing.chroma_index import ChromaIndex
from ..indexing.bm25_index import BM25Index
from ..retrieval.hybrid_retriever import HybridRetriever
from ..retrieval.rerank_cohere import CohereReranker
from ..generation.answer_chain import AnswerChain
from ..retrieval.citation_enforcer import CitationEnforcer
from ..services.query_service import QueryService

def run_evaluation_task(golden_file="data/golden_eval/qa_pairs.jsonl"):
    logger.info("Starting background evaluation task...")
    
    # Initialize components
    chroma = ChromaIndex()
    bm25 = BM25Index()
    reranker = CohereReranker()
    retriever = HybridRetriever(chroma, bm25, reranker=reranker)
    generator = AnswerChain()
    enforcer = CitationEnforcer()
    query_service = QueryService(retriever, generator, enforcer)
    
    # Load golden set
    if not os.path.exists(golden_file):
        logger.error(f"Golden file {golden_file} not found.")
        return
        
    data = []
    with open(golden_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            
    results = []
    for item in data:
        query = item['question']
        res = query_service.ask(query)
        
        results.append({
            "question": query,
            "answer": res['answer'],
            "contexts": [c['text'] for c in res['chunks']],
            "ground_truth": item['gold_answer']
        })
        
    ds = Dataset.from_list(results)
    score = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
    )
    
    output_path = "data/evaluation_report_latest.csv"
    score.to_pandas().to_csv(output_path, index=False)
    logger.info(f"Evaluation complete. Report saved to {output_path}")
