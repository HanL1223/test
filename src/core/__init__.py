"""
Core domain models and utilities for the Jira Ticket RAG system.

Exports:
    JiraIssue: Domain model for Jira issues
    TicketGenerationResult: Result of ticket generation
    RetrievedChunk: A retrieved document chunk
    build_issue_text: Build structured text from issue fields
    normalize_whitespace: Normalize text whitespace
    format_retrieved_context: Format chunks for LLM context
"""

from src.core.models import (
    JiraIssue,
    RetrievedChunk,
    TicketGenerationResult,
)
from src.core.text_utils import (
    build_issue_text,
    extract_jira_links,
    format_context_with_links,
    format_retrieved_context,
    normalize_whitespace,
    redact_jira_tokens,
)

__all__ = [
    # Models
    "JiraIssue",
    "TicketGenerationResult",
    "RetrievedChunk",
    # Text utilities
    "build_issue_text",
    "normalize_whitespace",
    "redact_jira_tokens",
    "format_retrieved_context",
    "format_context_with_links",
    "extract_jira_links",
]
