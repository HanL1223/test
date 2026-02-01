"""
================================================================================
TUTORIAL: Text Processing Utilities for Jira RAG
================================================================================

WHY TEXT UTILS?
---------------
Raw Jira data is messy:
  - Inconsistent whitespace
  - Account IDs that leak PII
  - Image/attachment references that add noise
  - Varying formats for links

These utilities clean and normalize text before it's embedded.
Clean text → Better embeddings → Better retrieval → Better generation.

LANGCHAIN RELEVANCE:
--------------------
While LangChain has text splitters, it doesn't do content-aware cleaning.
We apply these utilities BEFORE passing text to LangChain components.

================================================================================
"""

import re
from typing import Optional


# =============================================================================
# SECTION 1: WHITESPACE NORMALIZATION
# =============================================================================


def normalize_whitespace(text: str) -> str:
    """
    Normalize whitespace in text for consistent processing.
    
    WHY THIS MATTERS:
    -----------------
    Jira exports have inconsistent whitespace:
      - Windows line endings (\\r\\n)
      - Multiple spaces from copy-paste
      - Excessive blank lines
    
    Consistent whitespace means:
      - Consistent token counts
      - Better embedding similarity
      - Cleaner display
    
    OPERATIONS:
    1. Convert all line endings to \\n
    2. Collapse multiple spaces/tabs to single space
    3. Collapse 3+ newlines to double newline (preserve paragraphs)
    4. Strip leading/trailing whitespace
    
    Args:
        text: Raw text to normalize
        
    Returns:
        Normalized text
    
    Example:
        >>> normalize_whitespace("Hello  \\r\\n\\r\\n\\r\\nWorld")
        'Hello\\n\\nWorld'
    """
    # Convert line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    
    # Collapse horizontal whitespace (spaces, tabs)
    text = re.sub(r"[ \t]+", " ", text)
    
    # Collapse excessive blank lines (keep paragraph breaks)
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    return text.strip()


# =============================================================================
# SECTION 2: JIRA-SPECIFIC CLEANING
# =============================================================================

# Compiled regex patterns for efficiency (compiled once, used many times)
_ACCOUNT_ID_PATTERN = re.compile(r"\[~accountid:[^\]]+\]")
_IMAGE_PATTERN = re.compile(r"!\S+?\|[^!]*!")
_SMART_LINK_PATTERN = re.compile(r"\|smart-link\]", re.IGNORECASE)
_ATTACHMENT_PATTERN = re.compile(r"\[(\^[^\]]+)\]")


def redact_jira_tokens(text: str) -> str:
    """
    Remove or redact Jira-specific tokens that add noise.
    
    WHAT WE CLEAN:
    --------------
    1. Account IDs: [~accountid:abc123] → @user
       - Jira exports contain internal account IDs
       - These leak PII and add no semantic value
       - We replace with generic @user
    
    2. Inline images: !image.png|thumbnail! → (removed)
       - Image references can't be embedded meaningfully
       - They add noise to the text
    
    3. Smart links: |smart-link] → ]
       - Jira's smart link format breaks normal link parsing
       - We clean it to standard format
    
    4. Attachment references: [^file.pdf] → (cleaned)
       - Similar to images, references to attachments
    
    Args:
        text: Text with Jira markup
        
    Returns:
        Cleaned text
    
    Example:
        >>> redact_jira_tokens("[~accountid:123] uploaded !file.png|thumb!")
        '@user uploaded '
    """
    # Replace account IDs with generic @user
    text = _ACCOUNT_ID_PATTERN.sub("@user", text)
    
    # Remove inline image references
    text = _IMAGE_PATTERN.sub("", text)
    
    # Clean smart link format
    text = _SMART_LINK_PATTERN.sub("]", text)
    
    # Clean attachment references (keep the filename)
    text = _ATTACHMENT_PATTERN.sub(r"[attachment: \1]", text)
    
    return text


# =============================================================================
# SECTION 3: STRUCTURED TEXT BUILDING
# =============================================================================


def build_issue_text(
    *,  # Keyword-only arguments for clarity
    summary: str,
    description: Optional[str] = None,
    acceptance_criteria: Optional[str] = None,
    comments: Optional[list[str]] = None,
    include_labels: bool = False,
    labels: Optional[list[str]] = None,
) -> str:
    """
    Build structured text from Jira issue fields.
    
    WHY STRUCTURED FORMAT?
    ----------------------
    Just concatenating fields loses structure:
        "Fix bug We need to check... - [ ] Works in DEV"
    
    With structure, the model understands field boundaries:
        "Summary: Fix bug
         Description: We need to check...
         Acceptance Criteria: - [ ] Works in DEV"
    
    EMBEDDING BENEFIT:
    When we search for "acceptance criteria for feature X",
    the structured format helps match the right section.
    
    Args:
        summary: Issue title (required)
        description: Full description
        acceptance_criteria: Definition of done
        comments: List of comment texts
        include_labels: Whether to include labels section
        labels: Issue labels/tags
        
    Returns:
        Structured text ready for embedding
    
    Example:
        >>> build_issue_text(
        ...     summary="Configure auth",
        ...     description="Set up OAuth",
        ...     acceptance_criteria="- [ ] Works in DEV"
        ... )
        'Summary:\\nConfigure auth\\n\\n---\\n\\nDescription:\\nSet up OAuth...'
    """
    parts: list[str] = []
    
    # Summary is always included
    parts.append(f"Summary:\n{summary.strip()}")
    
    # Description (if present and non-empty)
    if description and description.strip():
        cleaned_desc = redact_jira_tokens(description.strip())
        if cleaned_desc:
            parts.append(f"Description:\n{cleaned_desc}")
    
    # Acceptance Criteria (if present)
    if acceptance_criteria and acceptance_criteria.strip():
        cleaned_ac = redact_jira_tokens(acceptance_criteria.strip())
        if cleaned_ac:
            parts.append(f"Acceptance Criteria:\n{cleaned_ac}")
    
    # Comments (limited to most recent for relevance)
    if comments:
        cleaned_comments = []
        for comment in comments[:15]:  # Limit to 15 most recent
            if comment and comment.strip():
                cleaned = redact_jira_tokens(comment.strip())
                if cleaned:
                    cleaned_comments.append(cleaned)
        
        if cleaned_comments:
            joined = "\n---\n".join(cleaned_comments)
            parts.append(f"Comments:\n{joined}")
    
    # Labels (optional)
    if include_labels and labels:
        label_str = ", ".join(labels)
        parts.append(f"Labels: {label_str}")
    
    # Join with section dividers and normalize
    return normalize_whitespace("\n\n---\n\n".join(parts))


# =============================================================================
# SECTION 4: CONTEXT FORMATTING
# =============================================================================


def format_retrieved_context(
    chunks: list,
    include_metadata: bool = True,
    max_chunks: Optional[int] = None,
) -> str:
    """
    Format retrieved chunks into context for the LLM.
    
    WHY FORMAT CONTEXT?
    -------------------
    Raw chunks are just text blobs. Formatting adds:
      - Clear delineation between chunks
      - Metadata (issue key, status) for context
      - Ranking information (Match 1, Match 2, etc.)
    
    This helps the LLM understand it's looking at multiple
    historical examples, not one continuous document.
    
    Args:
        chunks: List of RetrievedChunk objects
        include_metadata: Whether to include issue metadata
        max_chunks: Maximum chunks to include (None = all)
        
    Returns:
        Formatted context string
    """
    if not chunks:
        return "No relevant historical tickets found."
    
    # Limit chunks if specified
    if max_chunks:
        chunks = chunks[:max_chunks]
    
    lines: list[str] = ["## Retrieved Historical Tickets (Top Matches)\n"]
    
    for rank, chunk in enumerate(chunks, start=1):
        # Get metadata (handle both dict and object access)
        if hasattr(chunk, 'metadata'):
            meta = chunk.metadata
        elif hasattr(chunk, 'meta_data'):
            meta = chunk.meta_data
        else:
            meta = {}
        
        # Get text content
        if hasattr(chunk, 'text'):
            text = chunk.text
        elif hasattr(chunk, 'page_content'):
            text = chunk.page_content
        else:
            text = str(chunk)
        
        # Build header
        if include_metadata:
            issue_key = meta.get("issue_key", "UNKNOWN")
            status = meta.get("status", "Unknown")
            priority = meta.get("priority", "Unknown")
            
            # Get distance/similarity if available
            distance = getattr(chunk, 'distance', None)
            dist_str = f", distance={distance:.4f}" if distance is not None else ""
            
            header = f"### Match {rank}: {issue_key} (status={status}, priority={priority}{dist_str})"
        else:
            header = f"### Match {rank}"
        
        lines.append(f"\n{header}\n{text.strip()}\n")
    
    return "\n".join(lines)


# =============================================================================
# SECTION 5: LINK EXTRACTION
# =============================================================================


def extract_jira_links(text: str) -> list[dict[str, str]]:
    """
    Extract Jira-format links from text.
    
    JIRA LINK FORMAT:
    Jira uses [Display Text|URL] format for links.
    
    WHY EXTRACT LINKS?
    - Preserve important references in generated tickets
    - Show the LLM which links to include
    - Enable link validation/updating
    
    Args:
        text: Text containing Jira links
        
    Returns:
        List of {"display": ..., "url": ...} dicts
    
    Example:
        >>> extract_jira_links("[Docs|https://example.com]")
        [{'display': 'Docs', 'url': 'https://example.com'}]
    """
    pattern = r'\[([^\]|]+)\|([^\]]+)\]'
    matches = re.findall(pattern, text)
    return [{"display": m[0], "url": m[1]} for m in matches]


def format_context_with_links(
    chunks: list,
    include_link_summary: bool = True,
) -> str:
    """
    Format context with explicit link extraction.
    
    This variant of context formatting:
    1. Formats chunks normally
    2. Extracts all links from chunks
    3. Adds a summary section of links to preserve
    
    WHY?
    LLMs sometimes lose links during generation.
    Explicit link lists remind them to preserve these.
    
    Args:
        chunks: Retrieved chunks
        include_link_summary: Whether to add link summary section
        
    Returns:
        Formatted context with link preservation hints
    """
    if not chunks:
        return "No relevant historical tickets found."
    
    # Format the main context
    lines: list[str] = ["## Similar Historical Tickets (use as your template)\n"]
    all_links: list[dict] = []
    
    for rank, chunk in enumerate(chunks, start=1):
        # Get text
        if hasattr(chunk, 'text'):
            text = chunk.text
        elif hasattr(chunk, 'page_content'):
            text = chunk.page_content
        else:
            text = str(chunk)
        
        # Get metadata
        meta = getattr(chunk, 'metadata', getattr(chunk, 'meta_data', {}))
        issue_key = meta.get("issue_key", "UNKNOWN")
        
        # Get similarity info
        distance = getattr(chunk, 'distance', None)
        dist_str = f", similarity={1-distance:.2f}" if distance else ""
        
        lines.append(f"### Ticket {rank}: {issue_key}{dist_str}")
        lines.append(text.strip())
        lines.append("")
        
        # Extract links
        links = extract_jira_links(text)
        all_links.extend(links)
    
    # Add link preservation section
    if include_link_summary and all_links:
        lines.append("\n## Links to PRESERVE in your ticket:")
        seen_urls: set[str] = set()
        for link in all_links:
            if link["url"] not in seen_urls:
                lines.append(f"- [{link['display']}|{link['url']}]")
                seen_urls.add(link["url"])
    
    return "\n".join(lines)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Text normalization for consistent processing
# 2. Jira-specific cleaning (redacting PII, removing noise)
# 3. Structured text building for better embeddings
# 4. Context formatting for LLM consumption
# 5. Link extraction and preservation
#
# QUALITY IMPACT:
# - Clean text → Better embeddings → Better retrieval
# - Structured format → Model understands field boundaries
# - Link preservation → Generated tickets maintain references
#
# INTERVIEW TALKING POINTS:
# - "Text preprocessing is often overlooked but critical for RAG quality"
# - "Jira exports have PII (account IDs) that we redact"
# - "Explicit link extraction improves preservation in generation"
#
# =============================================================================
