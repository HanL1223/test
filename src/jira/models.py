"""
================================================================================
TUTORIAL: Jira Data Models
================================================================================

PURPOSE:
--------
These dataclasses define the structure for Jira API requests and responses.
Using typed models:
- Provides IDE autocompletion
- Catches errors at development time
- Documents the expected data format

ATLASSIAN DOCUMENT FORMAT (ADF):
--------------------------------
Jira Cloud API v3 requires descriptions in ADF, not plain text.
ADF is a JSON structure that represents rich text:

    {
        "type": "doc",
        "version": 1,
        "content": [
            {
                "type": "paragraph",
                "content": [
                    {"type": "text", "text": "Hello world"}
                ]
            }
        ]
    }

We provide helpers to convert plain text → ADF and Jira markup → ADF.

================================================================================
"""

import re
from dataclasses import dataclass, field
from typing import Optional, Any
from datetime import date


# =============================================================================
# SECTION 1: REQUEST MODELS
# =============================================================================


@dataclass
class JiraIssueCreate:
    """
    Data for creating a new Jira issue.
    
    REQUIRED FIELDS:
    ----------------
    summary: Issue title (required by Jira)
    
    OPTIONAL FIELDS:
    ----------------
    description: Issue description (plain text, will be converted to ADF)
    issue_type: Task, Story, Bug, etc. (defaults to "Task")
    assignee_account_id: Jira account ID to assign to
    labels: List of labels to apply
    start_date: Start date (YYYY-MM-DD format)
    custom_fields: Dict of custom field ID → value
    
    USAGE:
        issue = JiraIssueCreate(
            summary="Implement OAuth2",
            description="Add support for OAuth2 authentication...",
            issue_type="Story",
            labels=["backend", "security"],
        )
    """
    summary: str
    description: Optional[str] = None
    issue_type: str = "Task"
    assignee_account_id: Optional[str] = None
    labels: list[str] = field(default_factory=list)
    start_date: Optional[str] = None  # YYYY-MM-DD
    custom_fields: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate after initialization."""
        if not self.summary or not self.summary.strip():
            raise ValueError("Summary is required")
        self.summary = self.summary.strip()
        
        # Default start_date to today if not provided
        if self.start_date is None:
            self.start_date = date.today().isoformat()


@dataclass
class JiraIssueUpdate:
    """
    Data for updating an existing Jira issue.
    
    FIELDS:
    -------
    issue_key: The Jira issue key (e.g., "PROJ-123")
    description: New description (optional)
    summary: New summary (optional)
    labels: New labels (optional, replaces existing)
    custom_fields: Custom fields to update
    
    Only non-None fields will be updated.
    """
    issue_key: str
    description: Optional[str] = None
    summary: Optional[str] = None
    labels: Optional[list[str]] = None
    custom_fields: dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate after initialization."""
        if not self.issue_key or not self.issue_key.strip():
            raise ValueError("Issue key is required")
        self.issue_key = self.issue_key.strip().upper()


# =============================================================================
# SECTION 2: RESPONSE MODELS
# =============================================================================


@dataclass
class JiraCreateResult:
    """
    Result from creating a Jira issue.
    
    FIELDS:
    -------
    success: Whether creation succeeded
    key: Issue key (e.g., "PROJ-123")
    id: Issue ID (internal Jira ID)
    url: Browse URL for the issue
    error: Error message if failed
    """
    success: bool
    key: Optional[str] = None
    id: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "key": self.key,
            "id": self.id,
            "url": self.url,
            "error": self.error,
        }


@dataclass
class JiraUpdateResult:
    """
    Result from updating a Jira issue.
    
    FIELDS:
    -------
    success: Whether update succeeded
    key: Issue key that was updated
    url: Browse URL for the issue
    error: Error message if failed
    """
    success: bool
    key: Optional[str] = None
    url: Optional[str] = None
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "success": self.success,
            "key": self.key,
            "url": self.url,
            "error": self.error,
        }


# =============================================================================
# SECTION 3: ADF CONVERSION HELPERS
# =============================================================================


def text_to_adf(text: str) -> Optional[dict]:
    """
    Convert plain text to Atlassian Document Format (ADF).
    
    Handles:
    - Multiple paragraphs (separated by blank lines)
    - Basic newlines within paragraphs
    
    Args:
        text: Plain text content
        
    Returns:
        ADF document dict, or None if text is empty
        
    Example:
        >>> adf = text_to_adf("Hello\\n\\nWorld")
        >>> # Returns ADF with two paragraphs
    """
    if not text or not text.strip():
        return None
    
    # Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Split into paragraphs on blank lines
    blocks = []
    for block in text.split("\n\n"):
        block = block.strip()
        if not block:
            continue
        
        # Convert single newlines to hard breaks within paragraph
        lines = block.split("\n")
        content = []
        for i, line in enumerate(lines):
            if line.strip():
                content.append({
                    "type": "text",
                    "text": line,
                })
                # Add hard break between lines (not after last)
                if i < len(lines) - 1:
                    content.append({"type": "hardBreak"})
        
        if content:
            blocks.append({
                "type": "paragraph",
                "content": content,
            })
    
    if not blocks:
        return None
    
    return {
        "type": "doc",
        "version": 1,
        "content": blocks,
    }


def jira_markup_to_adf(markup: str) -> Optional[dict]:
    """
    Convert Jira wiki markup to ADF.
    
    Handles common patterns:
    - h2. Header → heading level 2
    - *bold* → strong mark
    - _italic_ → em mark
    - {code}...{code} → code block
    - - item → bullet list
    - # item → numbered list
    - [link|url] → link
    
    For complex markup, this is a best-effort conversion.
    Consider using Jira's own markup-to-ADF API for full fidelity.
    
    Args:
        markup: Jira wiki markup text
        
    Returns:
        ADF document dict, or None if empty
    """
    if not markup or not markup.strip():
        return None
    
    # Normalize line endings
    markup = markup.replace("\r\n", "\n").replace("\r", "\n")
    
    blocks = []
    lines = markup.split("\n")
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
        
        # Headers: h1. through h6.
        header_match = re.match(r'^h([1-6])\.\s*(.+)$', line)
        if header_match:
            level = int(header_match.group(1))
            text = header_match.group(2)
            blocks.append({
                "type": "heading",
                "attrs": {"level": level},
                "content": [{"type": "text", "text": text}],
            })
            i += 1
            continue
        
        # Code blocks: {code}...{code}
        if line.startswith("{code"):
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("{code}"):
                code_lines.append(lines[i])
                i += 1
            i += 1  # Skip closing {code}
            
            blocks.append({
                "type": "codeBlock",
                "content": [{"type": "text", "text": "\n".join(code_lines)}],
            })
            continue
        
        # Bullet list items
        if line.startswith("- ") or line.startswith("* "):
            list_items = []
            while i < len(lines) and (lines[i].strip().startswith("- ") or lines[i].strip().startswith("* ")):
                item_text = lines[i].strip()[2:]
                list_items.append({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": _parse_inline_markup(item_text),
                    }],
                })
                i += 1
            
            blocks.append({
                "type": "bulletList",
                "content": list_items,
            })
            continue
        
        # Numbered list items
        if line.startswith("# "):
            list_items = []
            while i < len(lines) and lines[i].strip().startswith("# "):
                item_text = lines[i].strip()[2:]
                list_items.append({
                    "type": "listItem",
                    "content": [{
                        "type": "paragraph",
                        "content": _parse_inline_markup(item_text),
                    }],
                })
                i += 1
            
            blocks.append({
                "type": "orderedList",
                "content": list_items,
            })
            continue
        
        # Regular paragraph
        blocks.append({
            "type": "paragraph",
            "content": _parse_inline_markup(line),
        })
        i += 1
    
    if not blocks:
        return None
    
    return {
        "type": "doc",
        "version": 1,
        "content": blocks,
    }


def _parse_inline_markup(text: str) -> list[dict]:
    """
    Parse inline Jira markup (bold, italic, links) to ADF content.
    
    This is a simplified parser. For full fidelity, use Jira's API.
    """
    content = []
    
    # Simple approach: just return plain text for now
    # A full implementation would parse *bold*, _italic_, [links|url], etc.
    # For MVP, we preserve the text as-is
    if text:
        content.append({"type": "text", "text": text})
    
    return content


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Dataclasses for typed API request/response models
# 2. Atlassian Document Format (ADF) for rich text
# 3. Converting plain text and markup to ADF
#
# WHY ADF?
# - Jira Cloud API v3 requires ADF for description fields
# - ADF is a structured JSON format for rich text
# - Enables consistent rendering across Jira clients
#
# INTERVIEW TALKING POINTS:
# - "We use dataclasses for type-safe API contracts"
# - "ADF conversion handles the Jira API's rich text requirement"
# - "Models provide validation at construction time"
#
# =============================================================================
