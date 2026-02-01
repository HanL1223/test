"""
================================================================================
TUTORIAL: LangChain Text Splitters for Jira Documents
================================================================================

WHY SPLIT TEXT?
---------------
Embedding models have token limits (typically 512-8192 tokens).
Long documents must be split into smaller chunks that:
  1. Fit within the embedding model's context window
  2. Are semantically coherent (don't split mid-sentence)
  3. Have some overlap for context continuity

LANGCHAIN TEXT SPLITTERS:
-------------------------
LangChain provides several splitters:
  - CharacterTextSplitter: Splits on character count
  - RecursiveCharacterTextSplitter: Tries multiple separators
  - TokenTextSplitter: Splits based on actual token count
  - MarkdownHeaderTextSplitter: Respects markdown structure

We use RecursiveCharacterTextSplitter because:
  - It tries to split on paragraphs first, then sentences, then words
  - This preserves semantic coherence
  - It's the recommended default in LangChain

JIRA-SPECIFIC CUSTOMIZATION:
----------------------------
Our Jira documents have structure (Summary, Description, etc.).
We customize the separators to respect this structure.

================================================================================
"""

import hashlib
import logging
from typing import Optional

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: JIRA-AWARE TEXT SPLITTER
# =============================================================================


def create_jira_text_splitter(
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> RecursiveCharacterTextSplitter:
    """
    Create a text splitter configured for Jira ticket content.
    
    SPLITTER BEHAVIOR:
    RecursiveCharacterTextSplitter tries separators in order:
      1. "\n\n---\n\n" - Our section divider (Summary → Description)
      2. "\n\n" - Paragraph breaks
      3. "\n" - Line breaks
      4. " " - Words
      5. "" - Characters (last resort)
    
    This means chunks will preferentially break at:
      - Section boundaries (best)
      - Paragraph boundaries (good)
      - Line boundaries (acceptable)
      - Word boundaries (not great)
    
    CHUNK SIZE GUIDANCE:
    - Too small: Loses context, more chunks to process
    - Too large: May exceed embedding limits, less precise retrieval
    - Sweet spot: 1000-1500 chars for Jira tickets
    
    Args:
        chunk_size: Maximum chunk size in characters (default from settings)
        chunk_overlap: Overlap between chunks (default from settings)
        
    Returns:
        Configured RecursiveCharacterTextSplitter
    
    Example:
        >>> splitter = create_jira_text_splitter()
        >>> chunks = splitter.split_documents(documents)
    """
    # Use settings defaults if not specified
    chunk_size = chunk_size or settings.rag.chunk_size
    chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
    
    # Validate overlap < size
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than "
            f"chunk_size ({chunk_size})"
        )
    
    # Custom separators for Jira content structure
    # Order matters - splitter tries each in sequence
    jira_separators = [
        "\n\n---\n\n",  # Our section divider (Summary/Description/etc.)
        "---",          # Markdown horizontal rule
        "\nh2. ",       # Jira h2 heading
        "\nh3. ",       # Jira h3 heading
        "\n## ",        # Markdown h2
        "\n### ",       # Markdown h3
        "\n\n",         # Paragraph break
        "\n",           # Line break
        ". ",           # Sentence break
        " ",            # Word break
        "",             # Character break (last resort)
    ]
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=jira_separators,
        length_function=len,  # Use character count
        is_separator_regex=False,  # Separators are literal strings
        keep_separator=True,  # Include separator at start of chunk
    )


# =============================================================================
# SECTION 2: CHUNK ID GENERATION
# =============================================================================


def generate_chunk_id(issue_key: str, chunk_index: int, chunk_text: str) -> str:
    """
    Generate a stable, unique ID for a document chunk.
    
    WHY STABLE IDS?
    Vector stores use IDs to:
      - Deduplicate on re-indexing
      - Update existing chunks
      - Reference in results
    
    We create IDs that are:
      - Deterministic (same input → same ID)
      - Unique (issue_key + index + content hash)
      - Human-readable prefix (for debugging)
    
    FORMAT:
    {issue_key}::chunk={index}::{content_hash}
    Example: "CSCI-123::chunk=0::a1b2c3d4"
    
    Args:
        issue_key: The Jira issue key (e.g., "CSCI-123")
        chunk_index: Index of this chunk within the issue
        chunk_text: The actual chunk content
        
    Returns:
        Stable chunk ID string
    """
    # Create content hash (first 16 chars of SHA-256)
    content_hash = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
    
    return f"{issue_key}::chunk={chunk_index}::{content_hash}"


# =============================================================================
# SECTION 3: DOCUMENT SPLITTING WITH METADATA
# =============================================================================


def split_documents_with_ids(
    documents: list[Document],
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> tuple[list[str], list[Document]]:
    """
    Split documents and generate stable chunk IDs.
    
    This is the main entry point for document splitting. It:
      1. Creates the text splitter
      2. Splits documents into chunks
      3. Generates stable IDs for each chunk
      4. Adds chunk metadata (index, id)
    
    METADATA PRESERVATION:
    LangChain's split_documents() preserves the original document's
    metadata on each chunk. We add:
      - chunk_id: Our stable ID
      - chunk_index: Position within original document
    
    Args:
        documents: List of LangChain Documents to split
        chunk_size: Maximum chunk size (optional, uses settings)
        chunk_overlap: Overlap between chunks (optional, uses settings)
        
    Returns:
        Tuple of (chunk_ids, chunk_documents)
    
    Example:
        >>> docs = loader.load()
        >>> ids, chunks = split_documents_with_ids(docs)
        >>> len(ids) == len(chunks)  # True
    """
    # Create splitter
    splitter = create_jira_text_splitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    
    # Track results
    all_ids: list[str] = []
    all_chunks: list[Document] = []
    
    # Process each document
    for doc in documents:
        # Get issue key from metadata
        issue_key = doc.metadata.get("issue_key", "UNKNOWN")
        
        # Split this document
        doc_chunks = splitter.split_documents([doc])
        
        # Generate IDs and update metadata for each chunk
        for chunk_idx, chunk in enumerate(doc_chunks):
            # Generate stable ID
            chunk_id = generate_chunk_id(
                issue_key=issue_key,
                chunk_index=chunk_idx,
                chunk_text=chunk.page_content,
            )
            
            # Add chunk-specific metadata
            chunk.metadata["chunk_id"] = chunk_id
            chunk.metadata["chunk_index"] = chunk_idx
            
            all_ids.append(chunk_id)
            all_chunks.append(chunk)
    
    logger.info(
        f"Split {len(documents)} documents into {len(all_chunks)} chunks "
        f"(avg {len(all_chunks)/max(len(documents),1):.1f} chunks/doc)"
    )
    
    return all_ids, all_chunks


# =============================================================================
# SECTION 4: CONVENIENCE FUNCTIONS
# =============================================================================


def split_document(
    doc: Document,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> list[Document]:
    """
    Split a single document into chunks.
    
    Convenience function for splitting one document at a time.
    
    Args:
        doc: Document to split
        chunk_size: Maximum chunk size
        chunk_overlap: Overlap between chunks
        
    Returns:
        List of chunk Documents
    """
    _, chunks = split_documents_with_ids(
        [doc],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return chunks


def estimate_chunks(
    text: str,
    chunk_size: Optional[int] = None,
    chunk_overlap: Optional[int] = None,
) -> int:
    """
    Estimate number of chunks for a given text.
    
    Useful for capacity planning without actually splitting.
    
    Args:
        text: Text to estimate
        chunk_size: Chunk size (default from settings)
        chunk_overlap: Overlap (default from settings)
        
    Returns:
        Estimated number of chunks
    """
    chunk_size = chunk_size or settings.rag.chunk_size
    chunk_overlap = chunk_overlap or settings.rag.chunk_overlap
    
    text_len = len(text.strip())
    if text_len <= chunk_size:
        return 1
    
    # Calculate with overlap
    effective_chunk = chunk_size - chunk_overlap
    return max(1, (text_len - chunk_overlap) // effective_chunk + 1)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. RecursiveCharacterTextSplitter for intelligent splitting
# 2. Custom separators for Jira document structure
# 3. Stable chunk ID generation for vector store operations
# 4. Metadata preservation through splitting
#
# LANGCHAIN INTEGRATION:
# - Uses LangChain's text splitters directly
# - Produces standard Document objects
# - Works with any LangChain vector store
#
# TUNING TIPS:
# - Larger chunks = more context but fewer, less precise matches
# - Smaller chunks = more precise but may lose context
# - More overlap = better continuity but more redundancy
#
# INTERVIEW TALKING POINTS:
# - "We customize separators to respect Jira's section structure"
# - "Stable IDs enable idempotent re-indexing"
# - "The splitter tries semantic boundaries before character breaks"
#
# =============================================================================
