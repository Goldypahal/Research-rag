# Remaining Tasks

## High Priority
- [x] **LLM-based Citation Check**: Move beyond simple keyword matching to full LLM verification of claims. (Done via `CitationEnforcer`)
- [x] **Metadata Extraction**: Enhance parsers to extract DOI, Year, and Authors reliably. (Done in `PyMuPDFPaperParser` and `UnstructuredPaperParser`)
- [x] **Dynamic Prompting**: Implement a better prompt management system (v1/v2/v3). (Done via `PromptLoader`)
- [x] **Trace Persistence**: Save retrieval and generation traces to local Storage (JSONL). (Done in `QueryService`)
- [x] **Parallel Evaluation**: Use Threading or Async for faster Ragas processing. (Done via background task)
- [x] **Checksum Deduplication**: Avoid re-indexing the same file multiple times. (Done in parsers)
- [x] **Comparison Logic**: Build the `compare_papers` endpoint with cross-paper retrieval. (Done in `routes_query.py`)
- [x] **Resiliency Layer**: Add retries with exponential backoff for all API calls (Cohere, OpenAI). (Done via `retry_utils.py`)
- [x] **Advanced Ingestion**: Integrate specialized table/figure parsing using Unstructured. (Done)
- [x] **Literature Note Exports**: Support Markdown and DOCX exports for RAG sessions. (Done)
- [x] **Multi-modal Support**: Use vision models (GPT-4o) to process figure images directly. (Done)
- [x] **Feedback Loop**: Allow users to 'star' good answers to build a local dataset. (Done)
- [x] **Active Learning Evaluator**: Generate synthetic QA pairs from papers automatically. (Done)
- [x] **S3/Cloud Storage Ingestion**: Batch process PDFs from a cloud bucket. (Done)

## Medium Priority
- [x] **Sub-section Aware Retrieval**: Boost chunks if the query mentions a specific section like 'Methodology'. (Done via `metadata_bonus`)
- [x] **Polishing & QoL**: Implement prompt switching and unified query routing.
- [x] **Testing Suite**: Unit, integration, and environment tests. (Done)

## Low Priority
- [ ] **UI/UX Refinement**: (Visuals handled by the system).
- [ ] **Chart Data Extraction**: Extract raw numbers from tables and charts for deeper analysis.
