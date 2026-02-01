"""
Data loading and processing package for Jira Ticket RAG.

This package provides:
  - Document loaders for JSONL and CSV files
  - Text splitters for chunking documents
  - Utilities for generating chunk IDs

LangChain Integration:
  - Loaders produce LangChain Document objects
  - Splitters use LangChain's RecursiveCharacterTextSplitter
  - All output is compatible with LangChain vector stores

Exports:
    JiraIssueLoader: Base loader interface
    JiraJsonlLoader: Load from JSONL files
    JiraCsvLoader: Load from CSV files
    create_loader: Factory function for loaders
    create_jira_text_splitter: Create configured text splitter
    split_documents_with_ids: Split docs and generate IDs
    generate_chunk_id: Generate stable chunk IDs
"""

from src.data.loader import (
    JiraCsvLoader,
    JiraIssueLoader,
    JiraJsonlLoader,
    create_loader,
)
from src.data.splitter import (
    create_jira_text_splitter,
    estimate_chunks,
    generate_chunk_id,
    split_document,
    split_documents_with_ids,
)

__all__ = [
    # Loaders
    "JiraIssueLoader",
    "JiraJsonlLoader",
    "JiraCsvLoader",
    "create_loader",
    # Splitters
    "create_jira_text_splitter",
    "split_documents_with_ids",
    "split_document",
    "generate_chunk_id",
    "estimate_chunks",
]
