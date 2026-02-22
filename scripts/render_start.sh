#!/bin/bash
set -e
echo "=== Indexing documents ==="
python scripts/index_documents.py --input data/processed/jira_issues.jsonl
echo "=== Starting server ==="
uvicorn backend.main:app --host 0.0.0.0 --port $PORT