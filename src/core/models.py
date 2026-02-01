"""
================================================================================
TUTORIAL: Domain Models for Jira RAG System
================================================================================

WHY DOMAIN MODELS?
------------------
Domain models represent the core entities in your system. They:
  1. Provide type safety and IDE autocompletion
  2. Document the data structure clearly
  3. Enable validation and transformation
  4. Decouple your code from external data formats

LANGCHAIN INTEGRATION:
----------------------
While LangChain has its own Document class, we keep our domain models
for several reasons:
  - Richer metadata specific to Jira
  - Clear separation of concerns
  - Easy conversion to/from LangChain Documents

================================================================================
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional

from langchain_core.documents import Document


# =============================================================================
# SECTION 1: JIRA ISSUE MODEL
# =============================================================================


@dataclass(frozen=True)
class JiraIssue:
    """
    Represents a normalized Jira issue record.
    
    This is the canonical representation used across:
      - Data ingestion (from CSV/JSONL)
      - Indexing (converting to chunks/vectors)
      - Retrieval (search results)
      - Generation (context for LLM)
    
    FROZEN DATACLASS:
    Using frozen=True makes instances immutable and hashable.
    This prevents accidental modifications and enables caching.
    
    FIELD DESCRIPTIONS:
    - issue_key: Unique identifier (e.g., "CSCI-123")
    - project_key: Project prefix (e.g., "CSCI")
    - issue_type: Task, Story, Bug, etc.
    - status: Current workflow state
    - summary: One-line title
    - description: Full description (may include Jira markup)
    - acceptance_criteria: Definition of done
    - comments: List of comment bodies
    - labels: Tags/categories
    - priority: Urgency level
    - created/updated: Timestamps
    - raw: Original data for debugging
    """
    # Required fields
    issue_key: str
    project_key: str
    issue_type: str
    status: str
    summary: str
    
    # Optional fields with defaults
    description: Optional[str] = None
    acceptance_criteria: Optional[str] = None
    comments: Optional[list[str]] = None
    labels: Optional[list[str]] = None
    priority: Optional[str] = None
    created: Optional[datetime] = None
    updated: Optional[datetime] = None
    raw: Optional[dict[str, Any]] = None
    
    def to_langchain_document(self, text_content: str) -> Document:
        """
        Convert to a LangChain Document.
        
        LANGCHAIN DOCUMENT:
        LangChain's Document class has two main fields:
          - page_content: The actual text content
          - metadata: Dictionary of associated data
        
        WHY CONVERT?
        LangChain components (text splitters, vector stores, retrievers)
        all work with Document objects. This method bridges our domain
        model to LangChain's ecosystem.
        
        Args:
            text_content: The processed text to use as page_content.
                         We take this as input rather than generating it
                         because text formatting may vary by use case.
        
        Returns:
            LangChain Document with Jira metadata
        """
        return Document(
            page_content=text_content,
            metadata={
                "issue_key": self.issue_key,
                "project_key": self.project_key,
                "issue_type": self.issue_type,
                "status": self.status,
                "priority": self.priority or "Unknown",
                "summary": self.summary,
                # Store dates as ISO strings for JSON serialization
                "created": self.created.isoformat() if self.created else None,
                "updated": self.updated.isoformat() if self.updated else None,
                "labels": ",".join(self.labels) if self.labels else "",
            }
        )
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "JiraIssue":
        """
        Create a JiraIssue from a dictionary.
        
        Handles both our processed format and raw Jira export format
        by checking for both key styles (snake_case and Title Case).
        
        Args:
            data: Dictionary with issue data
            
        Returns:
            JiraIssue instance
        """
        # Helper to get value with fallback keys
        def get_val(primary: str, *fallbacks: str) -> Any:
            if primary in data:
                return data[primary]
            for fb in fallbacks:
                if fb in data:
                    return data[fb]
            return None
        
        # Parse datetime strings
        def parse_dt(val: Any) -> Optional[datetime]:
            if not val:
                return None
            if isinstance(val, datetime):
                return val
            val = str(val).strip()
            # Try common formats
            for fmt in ("%d/%m/%Y %H:%M", "%d/%b/%y %I:%M %p", 
                       "%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S"):
                try:
                    return datetime.strptime(val, fmt)
                except ValueError:
                    continue
            return None
        
        # Parse labels (could be string or list)
        def parse_labels(val: Any) -> Optional[list[str]]:
            if not val:
                return None
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            s = str(val).strip()
            if not s:
                return None
            if "," in s:
                return [p.strip() for p in s.split(",") if p.strip()]
            return [p.strip() for p in s.split() if p.strip()]
        
        # Parse comments (could be string or list)
        def parse_comments(val: Any) -> Optional[list[str]]:
            if not val:
                return None
            if isinstance(val, list):
                return [str(x).strip() for x in val if str(x).strip()]
            s = str(val).strip()
            return [s] if s else None
        
        return cls(
            issue_key=str(get_val("issue_key", "Issue key") or "").strip(),
            project_key=str(get_val("project_key", "Project key") or "").strip(),
            issue_type=str(get_val("issue_type", "Issue Type") or "").strip(),
            status=str(get_val("status", "Status") or "").strip(),
            summary=str(get_val("summary", "Summary") or "").strip(),
            description=get_val("description", "Description"),
            acceptance_criteria=get_val(
                "acceptance_criteria", 
                "Custom field (Acceptance Criteria)"
            ),
            comments=parse_comments(get_val("comments")),
            labels=parse_labels(get_val("labels", "Labels")),
            priority=get_val("priority", "Priority"),
            created=parse_dt(get_val("created", "Created")),
            updated=parse_dt(get_val("updated", "Updated")),
            raw=data,
        )


# =============================================================================
# SECTION 2: GENERATION RESULT MODEL
# =============================================================================


@dataclass
class TicketGenerationResult:
    """
    Result of the ticket generation pipeline.
    
    This captures both the output and metadata about the generation process,
    which is useful for:
      - Debugging (did RAG work? how many chunks?)
      - Analytics (track usage patterns)
      - A/B testing (compare with/without RAG)
    """
    # The generated Jira markdown
    jira_markdown: str
    
    # Was RAG context used in generation?
    rag_context_used: bool = False
    
    # Number of chunks retrieved (0 if RAG disabled)
    retrieved_chunks: int = 0
    
    # Were refinement agents applied?
    agents_used: bool = False
    
    # Detected ticket style (brief/verbose)
    ticket_style: str = "verbose"
    
    # Optional: The retrieved context (for debugging)
    context: Optional[str] = None


# =============================================================================
# SECTION 3: RETRIEVED CHUNK MODEL
# =============================================================================


@dataclass
class RetrievedChunk:
    """
    A chunk retrieved from the vector store.
    
    LANGCHAIN NOTE:
    LangChain retrievers return Document objects. This class wraps
    that with additional retrieval-specific metadata (like distance).
    
    WHY WRAP?
    - LangChain Documents don't have a standard distance/score field
    - We can add our own methods for formatting
    - Clearer type in our code vs generic Document
    """
    # Unique identifier for this chunk
    chunk_id: str
    
    # The actual text content
    text: str
    
    # Metadata from the vector store
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # Similarity distance (lower = more similar for L2/cosine)
    distance: Optional[float] = None
    
    @classmethod
    def from_langchain_document(
        cls, 
        doc: Document, 
        chunk_id: str = "",
        distance: Optional[float] = None
    ) -> "RetrievedChunk":
        """
        Create from a LangChain Document.
        
        Args:
            doc: LangChain Document
            chunk_id: Optional ID (uses doc metadata if not provided)
            distance: Optional similarity distance
            
        Returns:
            RetrievedChunk instance
        """
        return cls(
            chunk_id=chunk_id or doc.metadata.get("chunk_id", "unknown"),
            text=doc.page_content,
            metadata=doc.metadata,
            distance=distance,
        )
    
    @property
    def issue_key(self) -> str:
        """Get the issue key from metadata."""
        return self.metadata.get("issue_key", "UNKNOWN")
    
    @property
    def similarity_score(self) -> Optional[float]:
        """
        Convert distance to similarity score (0-1).
        
        For L2 distance: similarity = 1 / (1 + distance)
        For cosine distance: similarity = 1 - distance
        """
        if self.distance is None:
            return None
        # Assuming cosine distance from Chroma
        return max(0.0, 1.0 - self.distance)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Frozen dataclasses for immutable domain objects
# 2. Conversion methods to/from LangChain Documents
# 3. Factory methods (from_dict) for flexible parsing
# 4. Computed properties for derived values
#
# DESIGN DECISIONS:
# - JiraIssue is frozen (immutable) for safety and caching
# - TicketGenerationResult is mutable (built up during pipeline)
# - RetrievedChunk bridges LangChain Documents to our domain
#
# INTERVIEW TALKING POINTS:
# - "We maintain our own models for richer domain semantics"
# - "Conversion methods enable clean integration with LangChain"
# - "Frozen dataclasses prevent subtle mutation bugs"
#
# =============================================================================
