# VERITI RAG: LangGraph + OpenRouter + DuckDB (simplified)

One API key (OPENROUTER_API_KEY) powers both chat and embeddings.

## Pipelines (both are LangGraph graphs)

INGEST   load -> split -> index
  load:   md -> by heading | yml -> per rule / per table | csv -> per row
  split:  RecursiveCharacterTextSplitter (size 800, overlap 100)
  index:  embeds only NEW chunks, deletes STALE ones (chunk id = hash of text)

QUERY    analyze -> retrieve -> [plan_sql <-> execute_sql] -> answer -> verify
  retrieve: dense search in Chroma, filtered by client_id
  verify:   every number in the answer must appear in the DuckDB result,
            otherwise the answer is regenerated once, then flagged

## Run
    pip install -r requirements.txt
    export OPENROUTER_API_KEY=sk-or-...
    python -m veriti_rag.cli ingest
    python -m veriti_rag.cli ask "How many won deals have no close value?" --show-context
    python -m veriti_rag.cli ask "Which product has the highest price?" --report question_report.pdf
    python -m veriti_rag.cli report --output dataset_insights.pdf
    python -m veriti_rag.cli report --clean yes --output business_insights.pdf
    python -m veriti_rag.cli sales-report --output sales_report.pdf
    python -m veriti_rag.cli sales-report --prompt "Emphasize product conversion"
    python -m veriti_rag.cli eval evaluation/retrieval_eval.jsonl
    pytest            # offline: fake LLM + hash embeddings, no key needed

  The `ask --report` option writes a question-specific PDF with the answer, SQL, returned rows, citations, and a chart when the result has a label column and a numeric value column. The `report` command creates a full-dataset PDF with table inventory, missing-value findings, available sales and product insights, recommendations, charts, and SQL evidence. Full-dataset reporting uses DuckDB directly and does not require the vector index or an LLM API call.

  Use `report --clean` (or `report --clean yes`) for a concise PDF containing only business insights, recommendations, and charts. Pass `--clean no` to explicitly request the full report.

  `sales-report` creates a business-focused PDF through a bounded prompt-enhancement, planning, coverage-review, and narrative workflow. Python/DuckDB calculates the report metrics; this command does not use Chroma or retrieve knowledge-base content. It requires `OPENROUTER_API_KEY` for the planning and narrative steps. The report includes won revenue, average won deal value, sales-cycle weeks, team and agent performance, quarter-over-quarter results, product win rates, sector sales, top products, and the top five accounts. Open-pipeline potential is estimated from matching product list prices, is not booked revenue or a probability-weighted forecast, and excludes opportunities without a matching catalog price.

## Models (env vars)
    VERITI_MODEL=anthropic/claude-opus-5.5          # any OpenRouter chat slug
    VERITI_EMBED_MODEL=openai/text-embedding-3-small  # any OpenRouter embedding slug
Check slugs at https://openrouter.ai/models (filter output: embeddings).
Changing the embedding model creates a new Chroma collection; re-run `ingest`.
Never mix embeddings from different models in one index.

## Your data
- CSVs in data/raw/ (table name = file name).
- Reference files in knowledge_base/<client_id>/ ; set VERITI_CLIENT_ID.
  File name decides doc_type (schema*, *dictionary*, *mapping*, drc*, *rule*,
  bsd*, remediation*, technical*, handover*, business_profile*).
  Bump `version` in _meta.yml to force a re-embed of that client's chunks.

## Removed vs the previous version (to simplify)
Custom chunker, BM25 hybrid retrieval (+RRF), LLM chunk grading, query rewrite loop,
sentence-transformers. Add hybrid back if exact-ID lookups (rule ids, column names) miss.

## Known limits
- Live OpenRouter calls are untested in the build environment.
- Offline hash embeddings are word-overlap only (tests/dev).
- verify does not normalise "$1.2M"-style figures.
