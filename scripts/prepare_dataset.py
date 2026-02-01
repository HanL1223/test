#!/usr/bin/env python3
"""
================================================================================
TUTORIAL: Dataset Preparation Script
================================================================================

PURPOSE:
--------
This script converts raw Jira CSV exports into a clean JSONL format
suitable for RAG indexing. It preserves the original dataset preparation
logic while improving organization and documentation.

WORKFLOW:
---------
1. Load raw Jira CSV export
2. Validate required columns exist
3. Clean and normalize text fields
4. Redact PII (account IDs, etc.)
5. Build combined text for embedding
6. Deduplicate by issue key (keep most recent)
7. Write outputs: JSONL, minimal CSV, stats

INPUT FORMAT:
-------------
Jira CSV export with columns like:
- Issue key, Summary, Description
- Custom field (Acceptance Criteria)
- Comment, Comment.1, Comment.2, ...

OUTPUT FILES:
-------------
- jira_issues.jsonl: Processed issues for indexing
- jira_issues_min.csv: Cleaned CSV for inspection
- stats.json: Dataset statistics

USAGE:
------
    python scripts/prepare_dataset.py --input data/raw/CSCI.csv --out-dir data/processed

================================================================================
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


# =============================================================================
# SECTION 1: CONFIGURATION
# =============================================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
LOGGER = logging.getLogger(__name__)


# Required columns from Jira CSV export
# These must be present for processing to succeed
DEFAULT_REQUIRED_COLUMNS = [
    "Issue key",
    "Summary",
    "Issue Type",
    "Status",
    "Project key",
    "Project name",
    "Priority",
    "Created",
    "Updated",
    "Labels",
    "Description",
    "Custom field (Acceptance Criteria)",
]


# =============================================================================
# SECTION 2: DATA MODEL
# =============================================================================


@dataclass
class PreparedIssue:
    """
    A processed Jira issue ready for indexing.
    
    FIELDS:
    -------
    issue_key: Unique identifier (e.g., "CSCI-123")
    project_key: Project prefix (e.g., "CSCI")
    project_name: Full project name
    issue_type: Story, Bug, Task, etc.
    status: Done, In Progress, etc.
    priority: High, Medium, Low, etc.
    created: Creation timestamp
    updated: Last update timestamp
    labels: List of labels
    summary: Issue title
    description: Main description (cleaned)
    acceptance_criteria: AC field (cleaned)
    comments: Combined comment text
    text: Full text for embedding (all fields combined)
    source: Provenance metadata
    """
    issue_key: str
    project_key: Optional[str]
    project_name: Optional[str]
    issue_type: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    created: Optional[str]
    updated: Optional[str]
    labels: List[str]
    summary: str
    description: str
    acceptance_criteria: str
    comments: str
    text: str  # Combined text for embedding
    source: Dict[str, Any]


# =============================================================================
# SECTION 3: TEXT PROCESSING UTILITIES
# =============================================================================


def _safe_str(value: Any) -> str:
    """
    Safely convert any value to string.
    
    Handles None and NaN values gracefully.
    This ensures .strip() and other string methods won't fail.
    
    Args:
        value: Any value (string, float, None, etc.)
        
    Returns:
        String representation or empty string for None/NaN
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)


def _normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text.
    
    OPERATIONS:
    1. Convert Windows line endings to Unix
    2. Collapse multiple spaces/tabs to single space
    3. Collapse 3+ newlines to 2 newlines
    4. Strip leading/trailing whitespace
    
    Args:
        text: Raw text with inconsistent whitespace
        
    Returns:
        Cleaned text with normalized whitespace
    """
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse horizontal whitespace
    text = re.sub(r"[ \t]+", " ", text)
    
    # Collapse excessive newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# Regex patterns for PII redaction
_ACCOUNT_ID_PATTERN = re.compile(r"\[~accountid:[^\]]+\]")
_IMAGE_PATTERN = re.compile(r"!\S+?\|[^!]*!")
_SMART_LINK_PATTERN = re.compile(r"\|smart-link\]", re.IGNORECASE)


def _redact_jira_tokens(text: str) -> str:
    """
    Redact PII and normalize Jira-specific tokens.
    
    REDACTIONS:
    1. Account IDs: [~accountid:abc123] → @user
    2. Embedded images: !image.png|alt=text! → (removed)
    3. Smart links: |smart-link] → ]
    
    WHY REDACT?
    - Account IDs are PII (privacy concern)
    - Image references don't add semantic value
    - Smart links add noise to embeddings
    
    Args:
        text: Raw Jira text with markup
        
    Returns:
        Text with PII removed
    """
    text = _ACCOUNT_ID_PATTERN.sub("@user", text)
    text = _IMAGE_PATTERN.sub("", text)
    text = _SMART_LINK_PATTERN.sub("]", text)
    return text


def _parse_labels(raw: str) -> List[str]:
    """
    Parse labels from Jira export format.
    
    Labels can be space or comma separated.
    
    Args:
        raw: Raw labels string (e.g., "label1, label2 label3")
        
    Returns:
        List of individual labels
    """
    raw = _normalize_whitespace(_safe_str(raw))
    if not raw:
        return []
    
    # Split on whitespace or comma
    tokens = re.split(r"[\s,]", raw)
    return [t for t in (tok.strip() for tok in tokens) if t]


def _extract_comment_bodies(row: pd.Series, comment_cols: List[str]) -> List[str]:
    """
    Extract comment bodies from Jira export columns.
    
    JIRA COMMENT FORMAT:
    Comments are exported as "timestamp;author;body" in each column.
    We extract just the body portion.
    
    Args:
        row: DataFrame row
        comment_cols: List of comment column names
        
    Returns:
        List of cleaned comment bodies
    """
    bodies: List[str] = []
    
    for col in comment_cols:
        raw = _safe_str(row.get(col, "")).strip()
        if not raw:
            continue
        
        # Split by semicolon - format is "timestamp;author;body"
        parts = raw.split(";", 2)
        if len(parts) == 3:
            body = parts[2]  # Extract just the body
        else:
            body = raw  # Fallback to full text
        
        body = _normalize_whitespace(_redact_jira_tokens(body))
        if body:
            bodies.append(body)
    
    return bodies


def _build_text(
    summary: str,
    description: str,
    acceptance_criteria: str,
    comments: str,
) -> str:
    """
    Build combined text for embedding.
    
    STRUCTURE:
    ```
    Summary
    <summary text>
    
    ---
    
    Description
    <description text>
    
    ---
    
    Acceptance Criteria
    <ac text>
    
    ---
    
    Comments
    <comments text>
    ```
    
    This structured format helps:
    1. Semantic chunking to split at section boundaries
    2. Retrieval to find relevant sections
    3. Generation to understand ticket structure
    
    Args:
        summary: Issue summary
        description: Issue description
        acceptance_criteria: Acceptance criteria text
        comments: Combined comments
        
    Returns:
        Structured combined text
    """
    sections: List[Tuple[str, str]] = [
        ("Summary", summary),
        ("Description", description),
        ("Acceptance Criteria", acceptance_criteria),
        ("Comments", comments),
    ]
    
    chunks: List[str] = []
    for title, content in sections:
        content = _normalize_whitespace(content)
        if not content:
            continue
        chunks.append(f"{title}\n{content}")
    
    return _normalize_whitespace("\n\n---\n\n".join(chunks))


# =============================================================================
# SECTION 4: DATA LOADING AND VALIDATION
# =============================================================================


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    """
    Validate that required columns exist in DataFrame.
    
    Args:
        df: Loaded DataFrame
        required: List of required column names
        
    Raises:
        ValueError: If any required columns are missing
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load Jira CSV export with proper encoding.
    
    Uses UTF-8 with BOM (utf-8-sig) which handles:
    - Standard UTF-8 files
    - Files with BOM marker (common in Windows exports)
    
    Args:
        csv_path: Path to CSV file
        
    Returns:
        Loaded DataFrame
    """
    df = pd.read_csv(
        csv_path,
        encoding="utf-8-sig",
        encoding_errors="replace",
    )
    return df


# =============================================================================
# SECTION 5: ISSUE PREPARATION
# =============================================================================


def prepare_issues(
    df: pd.DataFrame,
    source_name: str,
) -> List[PreparedIssue]:
    """
    Process DataFrame rows into PreparedIssue objects.
    
    PROCESSING STEPS:
    1. Extract and clean each field
    2. Redact PII from text fields
    3. Build combined text for embedding
    4. Deduplicate by issue key (keep most recent)
    
    Args:
        df: Loaded DataFrame
        source_name: Name of source file (for provenance)
        
    Returns:
        List of prepared issues (deduplicated)
    """
    # Find comment columns (Comment, Comment.1, Comment.2, etc.)
    comment_cols = [c for c in df.columns if c.startswith("Comment")]
    
    prepared: List[PreparedIssue] = []
    
    for idx, row in df.iterrows():
        # Extract and validate required fields
        issue_key = _normalize_whitespace(_safe_str(row.get("Issue key")))
        summary = _normalize_whitespace(_safe_str(row.get("Summary")))
        
        if not issue_key or not summary:
            LOGGER.warning(f"Skipping row {idx}: missing issue key or summary")
            continue
        
        # Clean text fields
        description = _normalize_whitespace(
            _redact_jira_tokens(_safe_str(row.get("Description")))
        )
        acceptance_criteria = _normalize_whitespace(
            _redact_jira_tokens(
                _safe_str(row.get("Custom field (Acceptance Criteria)"))
            )
        )
        
        # Extract and combine comments
        comment_bodies = _extract_comment_bodies(row, comment_cols)
        comments = _normalize_whitespace("\n\n".join(comment_bodies))
        
        # Build combined text for embedding
        text = _build_text(summary, description, acceptance_criteria, comments)
        
        # Parse labels
        labels = _parse_labels(_safe_str(row.get("Labels")))
        
        prepared.append(
            PreparedIssue(
                issue_key=issue_key,
                project_key=_normalize_whitespace(_safe_str(row.get("Project key"))),
                project_name=_normalize_whitespace(_safe_str(row.get("Project name"))),
                issue_type=_normalize_whitespace(_safe_str(row.get("Issue Type"))),
                status=_normalize_whitespace(_safe_str(row.get("Status"))),
                priority=_normalize_whitespace(_safe_str(row.get("Priority"))),
                created=_normalize_whitespace(_safe_str(row.get("Created"))),
                updated=_normalize_whitespace(_safe_str(row.get("Updated"))),
                labels=labels,
                summary=summary,
                description=description,
                acceptance_criteria=acceptance_criteria,
                comments=comments,
                text=text,
                source={
                    "source_name": source_name,
                    "row_index": idx,
                },
            )
        )
    
    # Deduplicate by issue key (keep most recent or longest)
    by_key: Dict[str, PreparedIssue] = {}
    for item in prepared:
        existing = by_key.get(item.issue_key)
        if existing is None:
            by_key[item.issue_key] = item
            continue
        
        # Prefer more recently updated
        if (item.updated or "") >= (existing.updated or ""):
            by_key[item.issue_key] = item
        # Or prefer longer text (more content)
        elif len(item.text) >= len(existing.text):
            by_key[item.issue_key] = item
    
    LOGGER.info(
        f"Deduplicated: {len(prepared)} → {len(by_key)} issues"
    )
    
    return list(by_key.values())


# =============================================================================
# SECTION 6: OUTPUT WRITERS
# =============================================================================


def write_jsonl(items: List[PreparedIssue], out_path: Path) -> None:
    """
    Write issues as JSON Lines file.
    
    JSONL format has one JSON object per line, which is:
    - Easy to stream/process line by line
    - Appendable (can add new issues)
    - Human readable
    
    Args:
        items: List of prepared issues
        out_path: Output file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            payload = asdict(item)
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def write_min_csv(items: List[PreparedIssue], out_path: Path) -> None:
    """
    Write minimal CSV for inspection.
    
    This CSV contains cleaned data without the full text field,
    making it easier to inspect in spreadsheet software.
    
    Args:
        items: List of prepared issues
        out_path: Output file path
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    rows = [
        {
            "issue_key": item.issue_key,
            "project_key": item.project_key,
            "project_name": item.project_name,
            "issue_type": item.issue_type,
            "status": item.status,
            "priority": item.priority,
            "created": item.created,
            "updated": item.updated,
            "labels": " ".join(item.labels),
            "summary": item.summary,
            "description": item.description[:500] if item.description else "",
            "acceptance_criteria": item.acceptance_criteria[:500] if item.acceptance_criteria else "",
            "comments": item.comments[:500] if item.comments else "",
        }
        for item in items
    ]
    
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")


def write_stats(items: List[PreparedIssue], out_path: Path) -> None:
    """
    Write dataset statistics.
    
    Stats include:
    - Total issue count
    - Breakdown by project, type, status
    
    Args:
        items: List of prepared issues
        out_path: Output file path
    """
    def _count(key_fn):
        out: Dict[str, int] = {}
        for item in items:
            k = key_fn(item) or "UNKNOWN"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    
    # Calculate text length statistics
    text_lengths = [len(item.text) for item in items]
    avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
    
    stats = {
        "total_prepared": len(items),
        "avg_text_length": round(avg_length, 1),
        "by_project_key": _count(lambda x: x.project_key),
        "by_issue_type": _count(lambda x: x.issue_type),
        "by_status": _count(lambda x: x.status),
        "by_priority": _count(lambda x: x.priority),
    }
    
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)


# =============================================================================
# SECTION 7: MAIN ENTRY POINT
# =============================================================================


def main() -> None:
    """
    Main entry point for dataset preparation.
    
    Parses command line arguments and runs the preparation pipeline.
    """
    parser = argparse.ArgumentParser(
        description="Prepare Jira CSV export for RAG indexing.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/prepare_dataset.py --input data/raw/CSCI.csv
  python scripts/prepare_dataset.py --input data/raw/CSCI.csv --out-dir data/processed
        """,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to Jira CSV export",
    )
    parser.add_argument(
        "--out-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Configure logging level
    logging.getLogger().setLevel(args.log_level)
    
    csv_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()
    
    # Load and validate
    LOGGER.info(f"Loading CSV: {csv_path}")
    df = load_csv(csv_path)
    LOGGER.info(f"Loaded {df.shape[0]} rows, {df.shape[1]} columns")
    
    validate_columns(df, DEFAULT_REQUIRED_COLUMNS)
    
    # Process
    LOGGER.info("Processing issues...")
    items = prepare_issues(df, source_name=csv_path.name)
    
    # Write outputs
    jsonl_path = out_dir / "jira_issues.jsonl"
    min_csv_path = out_dir / "jira_issues_min.csv"
    stats_path = out_dir / "stats.json"
    
    LOGGER.info(f"Writing JSONL: {jsonl_path}")
    write_jsonl(items, jsonl_path)
    
    LOGGER.info(f"Writing minimal CSV: {min_csv_path}")
    write_min_csv(items, min_csv_path)
    
    LOGGER.info(f"Writing stats: {stats_path}")
    write_stats(items, stats_path)
    
    LOGGER.info(f"Done! Prepared {len(items)} issues.")
    print(f"\nOutput files:")
    print(f"  JSONL:     {jsonl_path}")
    print(f"  CSV:       {min_csv_path}")
    print(f"  Stats:     {stats_path}")


if __name__ == "__main__":
    main()


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
# 1. Loads Jira CSV exports (handles encoding issues)
# 2. Cleans and normalizes text
# 3. Redacts PII (account IDs, images)
# 4. Builds structured text for embedding
# 5. Deduplicates by issue key
# 6. Writes JSONL for indexing
#
# KEY DESIGN DECISIONS:
# - Structured text format aids semantic chunking
# - PII redaction protects privacy
# - Deduplication keeps most recent version
#
# INTERVIEW TALKING POINT:
# "Data quality is critical for RAG. We clean, normalize, and redact
# before indexing to ensure embeddings are high quality and PII-safe."
#
# =============================================================================
