"""
Vector store package for Jira Ticket RAG.

Provides ChromaDB vector store via LangChain integration.

Exports:
    create_vector_store: Create a new vector store instance
    get_vector_store: Get cached singleton vector store
    add_documents: Add documents to the store
    create_from_documents: Create store from documents
    similarity_search: Search for similar documents
    similarity_search_with_score: Search with similarity scores
    get_collection_count: Get document count
    collection_exists: Check if collection has data
"""

from src.vectorstore.chroma import (
    add_documents,
    clear_cache,
    collection_exists,
    create_from_documents,
    create_vector_store,
    delete_collection,
    get_collection_count,
    get_vector_store,
    similarity_search,
    similarity_search_with_score,
)

__all__ = [
    "create_vector_store",
    "get_vector_store",
    "add_documents",
    "create_from_documents",
    "similarity_search",
    "similarity_search_with_score",
    "get_collection_count",
    "collection_exists",
    "delete_collection",
    "clear_cache",
]
