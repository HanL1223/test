"""
================================================================================
TUTORIAL: LangChain ChromaDB Vector Store
================================================================================

WHAT IS A VECTOR STORE?
-----------------------
A vector store is a database optimized for:
  1. Storing high-dimensional vectors (embeddings)
  2. Fast similarity search (find similar vectors)
  3. Metadata filtering (filter by issue_type, status, etc.)

CHROMADB:
---------
ChromaDB is an open-source, embedded vector database:
  - No server required (runs in-process)
  - Supports persistence (save to disk)
  - Fast approximate nearest neighbor search
  - Rich metadata filtering

LANGCHAIN INTEGRATION:
----------------------
LangChain's Chroma wrapper provides:
  - Unified interface (same API as other vector stores)
  - Automatic embedding on add
  - Built-in similarity search methods
  - Async support

WHY LANGCHAIN'S WRAPPER?
------------------------
Rather than using ChromaDB directly, we use LangChain's wrapper because:
  1. Consistent API (can swap to Pinecone, Weaviate, etc.)
  2. Integrated with LangChain retrievers and chains
  3. Automatic embedding handling
  4. Better error handling

================================================================================
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings
from src.embeddings import create_embeddings

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: VECTOR STORE FACTORY
# =============================================================================


def create_vector_store(
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path | str] = None,
    embeddings: Optional[GoogleGenerativeAIEmbeddings] = None,
) -> Chroma:
    """
    Create a ChromaDB vector store with LangChain wrapper.
    
    LANGCHAIN'S CHROMA CLASS:
    -------------------------
    The Chroma class provides:
      - from_documents(): Create and populate from documents
      - add_documents(): Add documents to existing store
      - similarity_search(): Find similar documents
      - similarity_search_with_score(): Include similarity scores
      - as_retriever(): Convert to LangChain retriever
    
    PERSISTENCE:
    When persist_directory is provided:
      - Data is saved to disk
      - Survives application restarts
      - Can be shared across processes
    
    Args:
        collection_name: Name of the collection (default from settings)
        persist_directory: Directory for persistence (default from settings)
        embeddings: Embedding model (default from settings)
        
    Returns:
        Configured Chroma vector store
        
    Example:
        >>> store = create_vector_store()
        >>> results = store.similarity_search("data modeling")
    """
    # Use settings defaults if not provided
    collection_name = collection_name or settings.chroma.collection
    persist_directory = persist_directory or settings.chroma.persist_dir
    
    # Ensure persist directory is a string (ChromaDB requirement)
    persist_directory = str(Path(persist_directory).resolve())
    
    # Create embeddings if not provided
    if embeddings is None:
        embeddings = create_embeddings()
    
    logger.debug(
        f"Creating Chroma vector store: "
        f"collection={collection_name}, "
        f"persist_dir={persist_directory}"
    )
    
    return Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )


# =============================================================================
# SECTION 2: CACHED SINGLETON
# =============================================================================


@lru_cache(maxsize=1)
def get_vector_store() -> Chroma:
    """
    Get a cached singleton vector store instance.
    
    WHY CACHE?
    Vector stores maintain connections and state:
      - Database connections
      - In-memory indexes
      - Embedding function reference
    
    Caching avoids reconnecting on each request.
    
    Returns:
        Cached Chroma vector store instance
    """
    return create_vector_store()


def clear_cache():
    """Clear the cached vector store instance."""
    get_vector_store.cache_clear()


# =============================================================================
# SECTION 3: DOCUMENT OPERATIONS
# =============================================================================


def add_documents(
    documents: list[Document],
    ids: Optional[list[str]] = None,
    collection_name: Optional[str] = None,
) -> list[str]:
    """
    Add documents to the vector store.
    
    LANGCHAIN'S ADD_DOCUMENTS:
    - Automatically embeds document content
    - Stores metadata alongside embeddings
    - Returns IDs of added documents
    
    IDEMPOTENCY:
    If documents with the same IDs already exist:
      - ChromaDB will update them (upsert behavior)
      - This enables safe re-indexing
    
    Args:
        documents: List of LangChain Documents to add
        ids: Optional list of IDs (if not provided, auto-generated)
        collection_name: Optional collection name (uses default if not provided)
        
    Returns:
        List of document IDs that were added
        
    Example:
        >>> docs = [Document(page_content="Hello", metadata={"key": "1"})]
        >>> ids = add_documents(docs, ids=["doc-1"])
    """
    if not documents:
        logger.warning("No documents to add")
        return []
    
    # Get or create vector store
    if collection_name:
        store = create_vector_store(collection_name=collection_name)
    else:
        store = get_vector_store()
    
    # Add documents (LangChain handles embedding)
    result_ids = store.add_documents(documents, ids=ids)
    
    logger.info(f"Added {len(documents)} documents to vector store")
    
    return result_ids


def create_from_documents(
    documents: list[Document],
    ids: Optional[list[str]] = None,
    collection_name: Optional[str] = None,
    persist_directory: Optional[Path | str] = None,
) -> Chroma:
    """
    Create a new vector store populated with documents.
    
    Use this when starting fresh (not adding to existing store).
    The from_documents class method is optimized for bulk loading.
    
    Args:
        documents: Documents to index
        ids: Optional IDs for documents
        collection_name: Collection name
        persist_directory: Persistence directory
        
    Returns:
        New Chroma vector store with documents
    """
    collection_name = collection_name or settings.chroma.collection
    persist_directory = str(persist_directory or settings.chroma.persist_dir)
    embeddings = create_embeddings()
    
    logger.info(
        f"Creating new vector store with {len(documents)} documents "
        f"in collection '{collection_name}'"
    )
    
    return Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        ids=ids,
        collection_name=collection_name,
        persist_directory=persist_directory,
    )


# =============================================================================
# SECTION 4: SEARCH OPERATIONS
# =============================================================================


def similarity_search(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[dict[str, Any]] = None,
) -> list[Document]:
    """
    Search for similar documents.
    
    SIMILARITY SEARCH:
    1. Query text is embedded
    2. Embedding is compared to stored vectors
    3. Top-k most similar documents returned
    
    FILTERING:
    ChromaDB supports metadata filtering:
      - {"status": "Done"} - Exact match
      - {"priority": {"$in": ["High", "Critical"]}} - In list
      - {"created": {"$gte": "2024-01-01"}} - Comparison
    
    Args:
        query: Search query text
        k: Number of results (default from settings)
        filter_dict: Optional metadata filter
        
    Returns:
        List of similar documents (most similar first)
        
    Example:
        >>> docs = similarity_search("data modeling", k=5)
        >>> docs[0].page_content  # Most similar
    """
    k = k or settings.rag.top_k
    store = get_vector_store()
    
    if filter_dict:
        return store.similarity_search(query, k=k, filter=filter_dict)
    return store.similarity_search(query, k=k)


def similarity_search_with_score(
    query: str,
    k: Optional[int] = None,
    filter_dict: Optional[dict[str, Any]] = None,
) -> list[tuple[Document, float]]:
    """
    Search with similarity scores.
    
    SCORES:
    The score is the distance metric (lower = more similar for L2).
    For cosine distance: score 0 = identical, score 2 = opposite.
    
    Args:
        query: Search query text
        k: Number of results
        filter_dict: Optional metadata filter
        
    Returns:
        List of (document, score) tuples
        
    Example:
        >>> results = similarity_search_with_score("data modeling")
        >>> for doc, score in results:
        ...     print(f"{doc.metadata['issue_key']}: {score:.4f}")
    """
    k = k or settings.rag.top_k
    store = get_vector_store()
    
    if filter_dict:
        return store.similarity_search_with_score(query, k=k, filter=filter_dict)
    return store.similarity_search_with_score(query, k=k)


# =============================================================================
# SECTION 5: UTILITY FUNCTIONS
# =============================================================================


def get_collection_count() -> int:
    """
    Get the number of documents in the collection.
    
    Returns:
        Number of documents (0 if collection doesn't exist)
    """
    try:
        store = get_vector_store()
        # Access the underlying ChromaDB collection
        return store._collection.count()
    except Exception as e:
        logger.warning(f"Failed to get collection count: {e}")
        return 0


def collection_exists() -> bool:
    """
    Check if the collection exists and has documents.
    
    Returns:
        True if collection exists with at least one document
    """
    return get_collection_count() > 0


def delete_collection(collection_name: Optional[str] = None) -> bool:
    """
    Delete a collection (for testing/cleanup).
    
    WARNING: This permanently deletes all data in the collection!
    
    Args:
        collection_name: Collection to delete (default from settings)
        
    Returns:
        True if deletion succeeded
    """
    collection_name = collection_name or settings.chroma.collection
    
    try:
        import chromadb
        
        persist_dir = str(settings.chroma.persist_dir)
        client = chromadb.PersistentClient(path=persist_dir)
        client.delete_collection(collection_name)
        
        # Clear cache since collection is gone
        clear_cache()
        
        logger.info(f"Deleted collection: {collection_name}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to delete collection: {e}")
        return False


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. LangChain's Chroma wrapper for vector operations
# 2. Document adding with automatic embedding
# 3. Similarity search with optional filtering
# 4. Singleton pattern for vector store
#
# LANGCHAIN INTEGRATION:
# - Uses official langchain-chroma package
# - Compatible with LangChain retrievers
# - Automatic embedding on add
#
# CHROMADB FEATURES USED:
# - Persistent storage
# - Similarity search
# - Metadata filtering
# - Collection management
#
# INTERVIEW TALKING POINTS:
# - "LangChain's Chroma wrapper provides a consistent API across vector stores"
# - "Metadata filtering enables advanced retrieval (by status, priority, etc.)"
# - "Persistent storage means no re-indexing on restart"
#
# =============================================================================
