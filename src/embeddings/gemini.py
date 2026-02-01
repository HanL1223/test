"""
================================================================================
TUTORIAL: LangChain Gemini Embeddings
================================================================================

WHAT ARE EMBEDDINGS?
--------------------
Embeddings convert text into dense vectors (lists of numbers) that capture
semantic meaning. Similar texts have similar vectors.

Example:
  "create a database table" → [0.2, -0.5, 0.8, ...]
  "build a new SQL table"   → [0.21, -0.48, 0.79, ...]  # Similar!
  "go get some coffee"      → [-0.3, 0.7, -0.1, ...]    # Different!

LANGCHAIN EMBEDDINGS:
---------------------
LangChain provides a unified interface for embedding models:
  - GoogleGenerativeAIEmbeddings (Gemini)
  - OpenAIEmbeddings
  - HuggingFaceEmbeddings
  - etc.

All implement the same interface:
  - embed_documents(texts): Embed multiple texts
  - embed_query(text): Embed a single query

WHY WRAP?
---------
We create a thin wrapper around LangChain's embeddings for:
  1. Centralized configuration from settings
  2. Consistent logging
  3. Error handling
  4. Easy testing/mocking

================================================================================
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: EMBEDDINGS FACTORY
# =============================================================================


def create_embeddings(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
) -> GoogleGenerativeAIEmbeddings:
    """
    Create a configured Gemini embeddings client.
    
    LANGCHAIN'S GoogleGenerativeAIEmbeddings:
    -----------------------------------------
    This is the official LangChain integration for Google's embedding models.
    
    Features:
    - Automatic batching for large document sets
    - Built-in retry logic
    - Async support (aembed_documents, aembed_query)
    - Task type specification (retrieval_document, retrieval_query, etc.)
    
    TASK TYPES:
    Google's embedding model supports task-specific embeddings:
    - RETRIEVAL_DOCUMENT: For documents being indexed
    - RETRIEVAL_QUERY: For search queries (optimized for matching documents)
    - SEMANTIC_SIMILARITY: For comparing similarity of texts
    - CLASSIFICATION: For text classification tasks
    
    We use RETRIEVAL_DOCUMENT when indexing and RETRIEVAL_QUERY when searching.
    This asymmetric approach improves retrieval quality.
    
    Args:
        api_key: Google API key (optional, uses settings if not provided)
        model: Embedding model name (optional, uses settings if not provided)
        
    Returns:
        Configured GoogleGenerativeAIEmbeddings instance
        
    Example:
        >>> embeddings = create_embeddings()
        >>> vector = embeddings.embed_query("How do I create a Jira ticket?")
        >>> len(vector)  # e.g., 768
    """
    # Use provided values or fall back to settings
    api_key = api_key or settings.google_api_key
    model = model or settings.embedding.model
    
    if not api_key:
        raise ValueError(
            "Google API key is required. "
            "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
        )
    
    logger.debug(f"Creating Gemini embeddings with model: {model}")
    
    return GoogleGenerativeAIEmbeddings(
        model=model,
        google_api_key=api_key,
        # Task type is set dynamically per operation in LangChain
        # Default is RETRIEVAL_DOCUMENT for embed_documents
        # and RETRIEVAL_QUERY for embed_query
    )


# =============================================================================
# SECTION 2: CACHED SINGLETON
# =============================================================================


@lru_cache(maxsize=1)
def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    """
    Get a cached singleton embeddings instance.
    
    WHY CACHE?
    Creating embeddings clients has overhead:
      - API authentication
      - Connection setup
      - Possible model loading
    
    By caching, we reuse the same client across the application.
    This is especially useful for the API server handling many requests.
    
    THREAD SAFETY:
    lru_cache is thread-safe for getting/setting cache entries.
    The GoogleGenerativeAIEmbeddings client is also thread-safe.
    
    Returns:
        Cached GoogleGenerativeAIEmbeddings instance
    
    Example:
        >>> emb1 = get_embeddings()
        >>> emb2 = get_embeddings()
        >>> emb1 is emb2  # True - same instance
    """
    return create_embeddings()


# =============================================================================
# SECTION 3: CONVENIENCE FUNCTIONS
# =============================================================================


def embed_documents(texts: list[str]) -> list[list[float]]:
    """
    Embed multiple texts (for document indexing).
    
    Uses RETRIEVAL_DOCUMENT task type which optimizes embeddings
    for being found in similarity search.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    
    Example:
        >>> texts = ["Document 1 content", "Document 2 content"]
        >>> vectors = embed_documents(texts)
        >>> len(vectors) == 2  # True
    """
    if not texts:
        return []
    
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)


def embed_query(text: str) -> list[float]:
    """
    Embed a single query (for search).
    
    Uses RETRIEVAL_QUERY task type which optimizes the embedding
    for finding similar documents.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    
    Example:
        >>> query_vec = embed_query("How do I reset my password?")
        >>> len(query_vec)  # e.g., 768
    """
    embeddings = get_embeddings()
    return embeddings.embed_query(text)


async def aembed_documents(texts: list[str]) -> list[list[float]]:
    """
    Async version of embed_documents.
    
    Useful for async web frameworks (like FastAPI) to avoid blocking.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors
    """
    if not texts:
        return []
    
    embeddings = get_embeddings()
    return await embeddings.aembed_documents(texts)


async def aembed_query(text: str) -> list[float]:
    """
    Async version of embed_query.
    
    Args:
        text: Query text to embed
        
    Returns:
        Embedding vector
    """
    embeddings = get_embeddings()
    return await embeddings.aembed_query(text)


# =============================================================================
# SECTION 4: UTILITY FUNCTIONS
# =============================================================================


def get_embedding_dimension() -> int:
    """
    Get the dimension of embeddings produced by the current model.
    
    EMBEDDING DIMENSIONS:
    - text-embedding-004: 768 dimensions
    - text-embedding-005 (if released): TBD
    
    Returns:
        Number of dimensions in embedding vectors
    
    Example:
        >>> dim = get_embedding_dimension()
        >>> dim  # 768
    """
    # The dimension depends on the model
    # text-embedding-004 produces 768-dimensional vectors
    model = settings.embedding.model
    
    # Known dimensions for Google models
    dimension_map = {
        "models/text-embedding-004": 768,
        "text-embedding-004": 768,
    }
    
    return dimension_map.get(model, 768)  # Default to 768


def clear_cache():
    """
    Clear the cached embeddings instance.
    
    Useful for testing or when you need to reconfigure.
    """
    get_embeddings.cache_clear()


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. LangChain's GoogleGenerativeAIEmbeddings for Gemini
# 2. Task types for document vs query embeddings
# 3. Singleton pattern with lru_cache
# 4. Sync and async embedding methods
#
# LANGCHAIN INTEGRATION:
# - Uses official LangChain Google GenAI integration
# - Compatible with all LangChain vector stores
# - Supports both sync and async operations
#
# INTERVIEW TALKING POINTS:
# - "We use task-specific embeddings (document vs query) for better retrieval"
# - "The cached singleton avoids connection overhead on each request"
# - "LangChain's abstraction lets us swap embedding providers easily"
#
# =============================================================================
