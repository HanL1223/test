"""
================================================================================
TUTORIAL: LangChain Retriever for Jira Documents
================================================================================

WHAT IS A RETRIEVER?
--------------------
A retriever is a component that:
  1. Takes a natural language query
  2. Returns relevant documents from a data source

LANGCHAIN RETRIEVERS:
---------------------
LangChain provides a Retriever abstraction that:
  - Has a simple interface: get_relevant_documents(query)
  - Supports async: aget_relevant_documents(query)
  - Integrates with chains via the | pipe operator

RETRIEVER VS VECTOR STORE:
--------------------------
Vector stores are lower-level (add, query operations).
Retrievers are higher-level (query → documents).

A vector store can be converted to a retriever:
    retriever = vector_store.as_retriever()

WHY CUSTOM RETRIEVER?
---------------------
We add features beyond basic vector search:
  - Context formatting for LLM consumption
  - Metadata extraction for display
  - Score-based filtering
  - Multi-retriever fusion (future enhancement)

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from src.config import settings
from src.core.models import RetrievedChunk
from src.core.text_utils import format_retrieved_context
from src.vectorstore import get_vector_store, similarity_search_with_score

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: RETRIEVAL CONFIGURATION
# =============================================================================


@dataclass
class RetrievalConfig:
    """
    Configuration for retrieval operations.
    
    TUNING GUIDANCE:
    - top_k: More results = richer context, higher latency/cost
    - score_threshold: Filter out low-quality matches
    - filter_metadata: Narrow search to specific issue types, etc.
    """
    # Number of documents to retrieve
    top_k: int = field(default_factory=lambda: settings.rag.top_k)
    
    # Minimum similarity score (optional, None = no filtering)
    # For cosine distance: 0.0 = identical, 2.0 = opposite
    score_threshold: Optional[float] = None
    
    # Metadata filter (e.g., {"status": "Done"})
    filter_metadata: Optional[dict[str, Any]] = None
    
    # Include metadata in formatted context
    include_metadata: bool = True


# =============================================================================
# SECTION 2: JIRA ISSUE RETRIEVER
# =============================================================================


class JiraIssueRetriever:
    """
    Retriever for Jira issue documents.
    
    RESPONSIBILITIES:
    1. Convert query to embedding (via vector store)
    2. Find similar documents
    3. Format results for LLM consumption
    4. Convert to RetrievedChunk objects
    
    LANGCHAIN INTEGRATION:
    We don't inherit from BaseRetriever directly because:
      - We want simpler configuration
      - We add domain-specific methods (format_context)
      - We return our RetrievedChunk type alongside Documents
    
    We can still get a LangChain retriever via as_langchain_retriever().
    """
    
    def __init__(self, config: Optional[RetrievalConfig] = None):
        """
        Initialize the retriever.
        
        Args:
            config: Retrieval configuration (uses defaults if not provided)
        """
        self.config = config or RetrievalConfig()
        self._vector_store = get_vector_store()
    
    def retrieve(self, query: str) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.
        
        PROCESS:
        1. Query is embedded by the vector store
        2. Similar documents are found
        3. Results are filtered by score (if threshold set)
        4. Documents are converted to RetrievedChunk
        
        Args:
            query: Natural language search query
            
        Returns:
            List of RetrievedChunk objects (most similar first)
            
        Example:
            >>> retriever = JiraIssueRetriever()
            >>> chunks = retriever.retrieve("data modeling for customer")
            >>> chunks[0].issue_key  # "CSCI-123"
        """
        query = (query or "").strip()
        if not query:
            return []
        
        # Search with scores
        results = similarity_search_with_score(
            query=query,
            k=self.config.top_k,
            filter_dict=self.config.filter_metadata,
        )
        
        # Convert to RetrievedChunk and optionally filter by score
        chunks: list[RetrievedChunk] = []
        for doc, score in results:
            # Filter by score threshold
            if self.config.score_threshold is not None:
                if score > self.config.score_threshold:
                    logger.debug(
                        f"Filtering out {doc.metadata.get('issue_key')} "
                        f"with score {score:.4f}"
                    )
                    continue
            
            chunks.append(
                RetrievedChunk.from_langchain_document(
                    doc=doc,
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    distance=score,
                )
            )
        
        logger.info(f"Retrieved {len(chunks)} chunks for query: {query[:50]}...")
        return chunks
    
    def retrieve_documents(self, query: str) -> list[Document]:
        """
        Retrieve as LangChain Documents (for chain compatibility).
        
        This returns the raw LangChain Documents without our
        RetrievedChunk wrapper.
        
        Args:
            query: Search query
            
        Returns:
            List of LangChain Documents
        """
        chunks = self.retrieve(query)
        return [
            Document(
                page_content=chunk.text,
                metadata={**chunk.metadata, "distance": chunk.distance},
            )
            for chunk in chunks
        ]
    
    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into context for the LLM.
        
        This creates a structured text that:
          - Clearly separates each chunk
          - Shows relevant metadata (issue_key, status)
          - Indicates similarity ranking
        
        Args:
            chunks: Retrieved chunks to format
            
        Returns:
            Formatted context string
            
        Example:
            >>> chunks = retriever.retrieve("query")
            >>> context = retriever.format_context(chunks)
            >>> print(context)
            ## Retrieved Historical Tickets
            ### Match 1: CSCI-123 (status=Done, priority=High)
            ...
        """
        return format_retrieved_context(
            chunks=chunks,
            include_metadata=self.config.include_metadata,
        )
    
    def retrieve_and_format(self, query: str) -> tuple[list[RetrievedChunk], str]:
        """
        Convenience method: retrieve and format in one call.
        
        Args:
            query: Search query
            
        Returns:
            Tuple of (chunks, formatted_context)
        """
        chunks = self.retrieve(query)
        context = self.format_context(chunks)
        return chunks, context
    
    def as_langchain_retriever(self) -> BaseRetriever:
        """
        Convert to a LangChain-compatible retriever.
        
        This returns the vector store's as_retriever() method,
        configured with our settings.
        
        Useful when you need to use this retriever in
        LangChain chains or agents.
        
        Returns:
            LangChain BaseRetriever
        """
        return self._vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": self.config.top_k,
                **({"filter": self.config.filter_metadata} 
                   if self.config.filter_metadata else {}),
            },
        )


# =============================================================================
# SECTION 3: FACTORY FUNCTIONS
# =============================================================================


def create_retriever(
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
    filter_metadata: Optional[dict[str, Any]] = None,
) -> JiraIssueRetriever:
    """
    Factory function to create a configured retriever.
    
    Args:
        top_k: Number of documents to retrieve
        score_threshold: Minimum similarity score
        filter_metadata: Metadata filter
        
    Returns:
        Configured JiraIssueRetriever
        
    Example:
        >>> # Get retriever with defaults
        >>> retriever = create_retriever()
        >>>
        >>> # Get retriever for high-priority items only
        >>> retriever = create_retriever(
        ...     top_k=10,
        ...     filter_metadata={"priority": "High"}
        ... )
    """
    config = RetrievalConfig(
        top_k=top_k or settings.rag.top_k,
        score_threshold=score_threshold,
        filter_metadata=filter_metadata,
    )
    return JiraIssueRetriever(config=config)


def get_langchain_retriever(
    top_k: Optional[int] = None,
    filter_metadata: Optional[dict[str, Any]] = None,
) -> BaseRetriever:
    """
    Get a LangChain-compatible retriever directly.
    
    Use this when you need a retriever for LangChain chains
    and don't need our additional methods.
    
    Args:
        top_k: Number of documents to retrieve
        filter_metadata: Metadata filter
        
    Returns:
        LangChain BaseRetriever
    """
    store = get_vector_store()
    search_kwargs: dict[str, Any] = {"k": top_k or settings.rag.top_k}
    
    if filter_metadata:
        search_kwargs["filter"] = filter_metadata
    
    return store.as_retriever(
        search_type="similarity",
        search_kwargs=search_kwargs,
    )


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Retriever pattern for document search
# 2. Score-based filtering
# 3. Context formatting for LLM consumption
# 4. LangChain retriever integration
#
# LANGCHAIN INTEGRATION:
# - Uses vector store's similarity search
# - Can convert to LangChain retriever
# - Compatible with LangChain chains
#
# RETRIEVAL STRATEGIES (for future enhancement):
# - Hybrid search (combine vector + keyword)
# - Multi-query retriever (generate multiple queries)
# - Contextual compression (filter retrieved content)
# - Re-ranking (reorder results with a second model)
#
# INTERVIEW TALKING POINTS:
# - "We wrap vector search with domain-specific formatting"
# - "Score thresholds prevent low-quality results from polluting context"
# - "The retriever is composable with LangChain chains"
#
# =============================================================================
