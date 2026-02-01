"""
Retrieval package for Jira Ticket RAG.

Provides retriever for finding relevant Jira documents.

Exports:
    JiraIssueRetriever: Main retriever class
    RetrievalConfig: Configuration dataclass
    create_retriever: Factory function
    get_langchain_retriever: Get LangChain-compatible retriever
"""

from src.retrieval.retriever import (
    JiraIssueRetriever,
    RetrievalConfig,
    create_retriever,
    get_langchain_retriever,
)

__all__ = [
    "JiraIssueRetriever",
    "RetrievalConfig",
    "create_retriever",
    "get_langchain_retriever",
]
