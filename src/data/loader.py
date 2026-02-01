"""
================================================================================
TUTORIAL: LangChain Document Loaders for Jira Data
================================================================================

WHAT ARE DOCUMENT LOADERS?
--------------------------
LangChain Document Loaders convert various data sources into Document objects.
Built-in loaders exist for PDFs, web pages, databases, etc.

For our Jira data, we create custom loaders that:
  1. Read our processed JSONL/CSV files
  2. Convert each record to a LangChain Document
  3. Apply our text building/cleaning utilities

LANGCHAIN PATTERN:
------------------
All loaders implement the BaseLoader interface with two methods:
  - load(): Returns all documents at once
  - lazy_load(): Generator that yields documents one at a time

lazy_load() is preferred for large datasets (memory efficient).

WHY CUSTOM LOADERS?
-------------------
- Our data has specific structure (issue_key, description, etc.)
- We apply domain-specific text processing
- We attach rich metadata for filtering and display

================================================================================
"""

import csv
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterator, Optional

from langchain_core.documents import Document

from src.core.models import JiraIssue
from src.core.text_utils import build_issue_text

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: BASE LOADER INTERFACE
# =============================================================================


class JiraIssueLoader(ABC):
    """
    Abstract base class for Jira issue loaders.
    
    DESIGN PATTERN: Template Method
    The load() method is implemented in terms of lazy_load(),
    so subclasses only need to implement lazy_load().
    
    LANGCHAIN COMPATIBILITY:
    This follows the same pattern as LangChain's BaseLoader,
    making our loaders feel native to the ecosystem.
    """
    
    @abstractmethod
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load documents one at a time.
        
        This is a generator that yields Document objects.
        Memory efficient for large datasets.
        
        Yields:
            LangChain Document objects
        """
        raise NotImplementedError
    
    def load(self) -> list[Document]:
        """
        Load all documents into memory.
        
        Convenience method that collects all documents from lazy_load().
        Use lazy_load() for large datasets to avoid memory issues.
        
        Returns:
            List of all Document objects
        """
        return list(self.lazy_load())


# =============================================================================
# SECTION 2: JSONL LOADER
# =============================================================================


class JiraJsonlLoader(JiraIssueLoader):
    """
    Load Jira issues from a JSONL (JSON Lines) file.
    
    JSONL FORMAT:
    Each line is a complete JSON object:
        {"issue_key": "CSCI-123", "summary": "...", ...}
        {"issue_key": "CSCI-124", "summary": "...", ...}
    
    ADVANTAGES OF JSONL:
    - Stream-friendly (no need to parse entire file)
    - One record per line (easy to count, filter)
    - Compatible with big data tools (Spark, etc.)
    
    This loader reads our prepare_dataset.py output.
    """
    
    def __init__(
        self,
        path: Path | str,
        max_issues: Optional[int] = None,
        include_comments: bool = True,
    ):
        """
        Initialize the JSONL loader.
        
        Args:
            path: Path to the JSONL file
            max_issues: Maximum issues to load (None = all)
            include_comments: Whether to include comments in text
        """
        self.path = Path(path)
        self.max_issues = max_issues
        self.include_comments = include_comments
        
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {self.path}")
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load documents from JSONL file.
        
        PROCESS:
        1. Open file, read line by line
        2. Parse JSON to dict
        3. Create JiraIssue from dict
        4. Build structured text
        5. Convert to LangChain Document
        
        Yields:
            Document for each valid Jira issue
        """
        issue_count = 0
        
        with self.path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                # Check max issues limit
                if self.max_issues and issue_count >= self.max_issues:
                    logger.info(f"Reached max issues limit: {self.max_issues}")
                    break
                
                # Skip empty lines
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse JSON
                    data = json.loads(line)
                    
                    # Create domain model
                    issue = JiraIssue.from_dict(data)
                    
                    # Skip invalid issues
                    if not issue.issue_key or not issue.summary:
                        logger.warning(
                            f"Skipping line {line_num}: missing issue_key or summary"
                        )
                        continue
                    
                    # Build structured text for embedding
                    text = build_issue_text(
                        summary=issue.summary,
                        description=issue.description,
                        acceptance_criteria=issue.acceptance_criteria,
                        comments=issue.comments if self.include_comments else None,
                    )
                    
                    # Convert to LangChain Document
                    doc = issue.to_langchain_document(text)
                    
                    issue_count += 1
                    yield doc
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        logger.info(f"Loaded {issue_count} issues from {self.path}")


# =============================================================================
# SECTION 3: CSV LOADER
# =============================================================================


class JiraCsvLoader(JiraIssueLoader):
    """
    Load Jira issues from a CSV file.
    
    CSV HANDLING:
    CSV files from Jira exports can be messy:
      - Inconsistent column names
      - Quoted fields with commas
      - Unicode issues
    
    We use Python's csv.DictReader which handles most of this.
    The from_dict() method handles column name variations.
    """
    
    def __init__(
        self,
        path: Path | str,
        max_issues: Optional[int] = None,
        include_comments: bool = True,
    ):
        """
        Initialize the CSV loader.
        
        Args:
            path: Path to the CSV file
            max_issues: Maximum issues to load (None = all)
            include_comments: Whether to include comments in text
        """
        self.path = Path(path)
        self.max_issues = max_issues
        self.include_comments = include_comments
        
        if not self.path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.path}")
    
    def _extract_comments(self, row: dict) -> list[str]:
        """
        Extract comments from CSV row.
        
        Jira CSV exports have columns like:
          Comment, Comment.1, Comment.2, ...
        
        Each comment is formatted as:
          "date;author;body"
        
        We extract just the body (third part).
        """
        comments = []
        for key, value in row.items():
            if not key:
                continue
            if key.lower().startswith("comment") and value:
                value = str(value).strip()
                if value:
                    # Try to extract body from "date;author;body" format
                    parts = value.split(";", 2)
                    if len(parts) == 3:
                        body = parts[2].strip()
                    else:
                        body = value
                    if body:
                        comments.append(body)
        return comments
    
    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load documents from CSV file.
        
        Yields:
            Document for each valid Jira issue
        """
        issue_count = 0
        
        with self.path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            
            for row_num, row in enumerate(reader, start=1):
                # Check max issues limit
                if self.max_issues and issue_count >= self.max_issues:
                    break
                
                try:
                    # Create domain model
                    issue = JiraIssue.from_dict(row)
                    
                    # Skip invalid issues
                    if not issue.issue_key or not issue.summary:
                        logger.warning(
                            f"Skipping row {row_num}: missing issue_key or summary"
                        )
                        continue
                    
                    # Get comments from CSV columns
                    comments = None
                    if self.include_comments:
                        comments = self._extract_comments(row)
                    
                    # Build structured text
                    text = build_issue_text(
                        summary=issue.summary,
                        description=issue.description,
                        acceptance_criteria=issue.acceptance_criteria,
                        comments=comments,
                    )
                    
                    # Convert to LangChain Document
                    doc = issue.to_langchain_document(text)
                    
                    issue_count += 1
                    yield doc
                    
                except Exception as e:
                    logger.warning(f"Error processing row {row_num}: {e}")
        
        logger.info(f"Loaded {issue_count} issues from {self.path}")


# =============================================================================
# SECTION 4: FACTORY FUNCTION
# =============================================================================


def create_loader(
    path: Path | str,
    max_issues: Optional[int] = None,
    include_comments: bool = True,
) -> JiraIssueLoader:
    """
    Factory function to create the appropriate loader based on file extension.
    
    FACTORY PATTERN:
    Encapsulates the creation logic so callers don't need to know
    about specific loader classes.
    
    Args:
        path: Path to the data file
        max_issues: Maximum issues to load
        include_comments: Whether to include comments
        
    Returns:
        Appropriate loader instance
        
    Raises:
        ValueError: If file extension is not supported
    
    Example:
        >>> loader = create_loader("data/issues.jsonl")
        >>> docs = loader.load()
    """
    path = Path(path)
    extension = path.suffix.lower()
    
    if extension in {".jsonl", ".jsonlines"}:
        return JiraJsonlLoader(
            path=path,
            max_issues=max_issues,
            include_comments=include_comments,
        )
    
    if extension == ".csv":
        return JiraCsvLoader(
            path=path,
            max_issues=max_issues,
            include_comments=include_comments,
        )
    
    raise ValueError(
        f"Unsupported file extension: {extension}. "
        f"Supported: .jsonl, .jsonlines, .csv"
    )


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. LangChain loader pattern (load/lazy_load)
# 2. JSONL format for streaming data
# 3. CSV handling with DictReader
# 4. Factory pattern for loader creation
#
# LANGCHAIN INTEGRATION:
# - Our loaders produce standard LangChain Documents
# - Documents work with any LangChain text splitter, vector store, etc.
# - Metadata is preserved for filtering and display
#
# INTERVIEW TALKING POINTS:
# - "We use lazy_load() for memory-efficient processing of large exports"
# - "The factory pattern lets us swap data sources without changing code"
# - "Rich metadata enables filtering by issue_type, status, etc."
#
# =============================================================================
