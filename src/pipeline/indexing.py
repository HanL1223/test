"""
================================================================================
TUTORIAL: Indexing Pipeline
================================================================================

PIPELINE A: DOCUMENT INDEXING
-----------------------------
The indexing pipeline prepares documents for RAG retrieval:

    Raw Data (CSV/JSONL)
         ↓
    Load (LangChain Loaders)
         ↓
    Clean (Text Preprocessing)
         ↓
    Chunk (Semantic Splitting)
         ↓
    Embed (Gemini Embeddings)
         ↓
    Store (ChromaDB)

IDEMPOTENT INDEXING:
--------------------
The pipeline supports re-indexing without duplicates:
1. Each chunk gets a stable ID (based on content hash)
2. Existing chunks with same ID are skipped
3. New chunks are added

This allows incremental updates as new tickets are added.

CONFIGURATION:
--------------
IndexingConfig controls:
- data_path: Input file path
- collection_name: ChromaDB collection
- chunk_size: Max characters per chunk
- chunk_overlap: Overlap between chunks
- batch_size: Embedding batch size

================================================================================
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import time

from langchain_core.documents import Document

from src.config import settings
from src.data import create_loader, split_documents_with_ids
from src.embeddings import get_embeddings
from src.vectorstore import (
    create_vector_store,
    create_from_documents,
    get_collection_count,
    delete_collection,
    get_vector_store,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================


@dataclass
class IndexingConfig:
    """
    Configuration for the indexing pipeline.
    
    Attributes:
        data_path: Path to input file (CSV or JSONL)
        collection_name: Name for ChromaDB collection
        chunk_size: Maximum chunk size in characters
        chunk_overlap: Overlap between chunks
        batch_size: Number of documents to embed at once
        persist_directory: ChromaDB storage location
        force_rebuild: If True, delete existing collection first
    """
    data_path: Path
    collection_name: str = "jira_tickets"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    batch_size: int = 100
    persist_directory: Optional[Path] = None
    force_rebuild: bool = False
    
    def __post_init__(self):
        """Validate and convert paths."""
        self.data_path = Path(self.data_path)
        if self.persist_directory:
            self.persist_directory = Path(self.persist_directory)
        else:
            self.persist_directory = Path(settings.chroma.persist_directory)


@dataclass
class IndexingResult:
    """
    Results from indexing pipeline execution.
    
    Attributes:
        success: Whether indexing completed successfully
        documents_loaded: Number of documents from source
        chunks_created: Number of chunks after splitting
        chunks_indexed: Number of chunks added to vector store
        collection_total: Total documents in collection
        duration_seconds: Time taken for indexing
        errors: List of any errors encountered
    """
    success: bool
    documents_loaded: int = 0
    chunks_created: int = 0
    chunks_indexed: int = 0
    collection_total: int = 0
    duration_seconds: float = 0.0
    errors: list[str] = field(default_factory=list)


# =============================================================================
# SECTION 2: INDEXING PIPELINE
# =============================================================================


class IndexingPipeline:
    """
    End-to-end document indexing pipeline.
    
    WORKFLOW:
    ---------
    1. Load documents from CSV or JSONL
    2. Split into semantic chunks
    3. Add stable IDs to chunks
    4. Create/update vector store
    
    Example:
        >>> config = IndexingConfig(
        ...     data_path="data/processed/jira_issues.jsonl",
        ...     collection_name="jira_tickets"
        ... )
        >>> pipeline = IndexingPipeline(config)
        >>> result = pipeline.run()
        >>> print(f"Indexed {result.chunks_indexed} chunks")
    """
    
    def __init__(self, config: IndexingConfig):
        """
        Initialize the indexing pipeline.
        
        Args:
            config: IndexingConfig with pipeline settings
        """
        self.config = config
        self._embeddings = None
    
    @property
    def embeddings(self):
        """Lazy-load embeddings."""
        if self._embeddings is None:
            self._embeddings = get_embeddings()
        return self._embeddings
    
    def _load_documents(self) -> list[Document]:
        """
        Load documents from the data source.
        
        Returns:
            List of LangChain Documents
        """
        logger.info(f"Loading documents from {self.config.data_path}")
        
        if not self.config.data_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {self.config.data_path}"
            )
        
        loader = create_loader(self.config.data_path)
        documents = list(loader.lazy_load())
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents
    
    def _split_documents(
        self,
        documents: list[Document],
    ) -> list[Document]:
        """
        Split documents into chunks with stable IDs.
        
        Args:
            documents: List of documents to split
            
        Returns:
            List of chunked documents with IDs
        """
        logger.info(
            f"Splitting documents (chunk_size={self.config.chunk_size}, "
            f"overlap={self.config.chunk_overlap})"
        )
        
        # split_documents_with_ids returns (ids, chunks) tuple
        chunk_ids, chunks = split_documents_with_ids(
            documents=documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap,
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _prepare_collection(self) -> None:
        """
        Prepare the collection, optionally deleting if force_rebuild.
        """
        if self.config.force_rebuild:
            logger.warning(
                f"Force rebuild: deleting collection '{self.config.collection_name}'"
            )
            try:
                delete_collection(self.config.collection_name)
            except Exception as e:
                logger.warning(f"Could not delete collection: {e}")
    
    def _index_chunks(self, chunks: list[Document]) -> int:
        """
        Index chunks into the vector store.
        
        BATCHING:
        ---------
        We process in batches to:
        1. Avoid memory issues with large datasets
        2. Provide progress feedback
        3. Enable partial recovery on failure
        
        Args:
            chunks: List of document chunks to index
            
        Returns:
            Number of chunks indexed
        """
        logger.info(f"Indexing {len(chunks)} chunks to ChromaDB")
        
        # Process in batches
        total_indexed = 0
        batch_size = self.config.batch_size
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_num = i // batch_size + 1
            total_batches = (len(chunks) + batch_size - 1) // batch_size
            
            logger.debug(f"Processing batch {batch_num}/{total_batches}")
            
            try:
                # Extract IDs from chunk metadata
                ids = [
                    chunk.metadata.get("chunk_id", f"chunk_{i+j}")
                    for j, chunk in enumerate(batch)
                ]
                
                # Get or create vector store
                vector_store = create_vector_store(
                    collection_name=self.config.collection_name,
                    embeddings=self.embeddings,
                    persist_directory=str(self.config.persist_directory),
                )
                
                # Add documents with IDs
                vector_store.add_documents(
                    documents=batch,
                    ids=ids,
                )
                
                total_indexed += len(batch)
                
            except Exception as e:
                logger.error(f"Failed to index batch {batch_num}: {e}")
                raise
        
        return total_indexed
    
    def run(self) -> IndexingResult:
        """
        Execute the full indexing pipeline.
        
        Returns:
            IndexingResult with statistics
        """
        start_time = time.time()
        errors: list[str] = []
        
        logger.info("=" * 60)
        logger.info("Starting Indexing Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Prepare collection
            self._prepare_collection()
            
            # Step 2: Load documents
            documents = self._load_documents()
            
            # Step 3: Split into chunks
            chunks = self._split_documents(documents)
            
            # Step 4: Index chunks
            chunks_indexed = self._index_chunks(chunks)
            
            # Get final count
            collection_total = get_collection_count(
                
            )
            
            duration = time.time() - start_time
            
            logger.info("=" * 60)
            logger.info("Indexing Complete!")
            logger.info(f"Documents loaded: {len(documents)}")
            logger.info(f"Chunks created: {len(chunks)}")
            logger.info(f"Chunks indexed: {chunks_indexed}")
            logger.info(f"Collection total: {collection_total}")
            logger.info(f"Duration: {duration:.2f}s")
            logger.info("=" * 60)
            
            return IndexingResult(
                success=True,
                documents_loaded=len(documents),
                chunks_created=len(chunks),
                chunks_indexed=chunks_indexed,
                collection_total=collection_total,
                duration_seconds=duration,
                errors=errors,
            )
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = f"Indexing failed: {str(e)}"
            logger.error(error_msg)
            errors.append(error_msg)
            
            return IndexingResult(
                success=False,
                duration_seconds=duration,
                errors=errors,
            )


# =============================================================================
# SECTION 3: CONVENIENCE FUNCTION
# =============================================================================


def run_indexing_pipeline(
    data_path: str | Path,
    collection_name: str = "jira_tickets",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    force_rebuild: bool = False,
) -> IndexingResult:
    """
    Run the indexing pipeline (convenience function).
    
    Args:
        data_path: Path to input file
        collection_name: ChromaDB collection name
        chunk_size: Maximum chunk size
        chunk_overlap: Chunk overlap
        force_rebuild: Delete existing collection first
        
    Returns:
        IndexingResult with statistics
        
    Example:
        >>> result = run_indexing_pipeline(
        ...     data_path="data/processed/jira_issues.jsonl",
        ...     collection_name="jira_tickets",
        ...     force_rebuild=True
        ... )
        >>> if result.success:
        ...     print(f"Indexed {result.chunks_indexed} chunks")
    """
    config = IndexingConfig(
        data_path=Path(data_path),
        collection_name=collection_name,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        force_rebuild=force_rebuild,
    )
    
    pipeline = IndexingPipeline(config)
    return pipeline.run()


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. End-to-end indexing workflow
# 2. Batch processing for large datasets
# 3. Idempotent indexing with stable IDs
# 4. Progress logging and error handling
#
# KEY PATTERNS:
# - Configuration dataclass for settings
# - Result dataclass for outputs
# - Batch processing for scalability
# - Lazy loading of expensive resources
#
# INTERVIEW TALKING POINTS:
# - "Stable chunk IDs enable idempotent re-indexing"
# - "Batch processing handles large document sets"
# - "Configuration objects make pipelines testable"
# - "Detailed logging enables production debugging"
#
# =============================================================================
