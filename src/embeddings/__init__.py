"""
Embeddings package for Jira Ticket RAG.

Provides Google Gemini embeddings via LangChain integration.

Exports:
    create_embeddings: Create a new embeddings instance
    get_embeddings: Get cached singleton embeddings
    embed_documents: Embed multiple documents
    embed_query: Embed a single query
    aembed_documents: Async document embedding
    aembed_query: Async query embedding
"""

from src.embeddings.gemini import (
    aembed_documents,
    aembed_query,
    clear_cache,
    create_embeddings,
    embed_documents,
    embed_query,
    get_embedding_dimension,
    get_embeddings,
)

__all__ = [
    "create_embeddings",
    "get_embeddings",
    "embed_documents",
    "embed_query",
    "aembed_documents",
    "aembed_query",
    "get_embedding_dimension",
    "clear_cache",
]
