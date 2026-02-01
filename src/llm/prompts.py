"""
================================================================================
TUTORIAL: LangChain Prompt Templates for Jira Ticket Generation
================================================================================

WHAT ARE PROMPT TEMPLATES?
--------------------------
Prompt templates are structured formats for LLM prompts that:
  1. Separate static instructions from dynamic content
  2. Validate required variables
  3. Enable reuse and composition

LANGCHAIN PROMPT TEMPLATES:
---------------------------
LangChain provides several template types:

1. PromptTemplate: Simple string formatting
   "Translate {text} to {language}"

2. ChatPromptTemplate: For chat models
   [SystemMessage("You are..."), HumanMessage("{input}")]

3. MessagesPlaceholder: For conversation history
   Allows inserting a list of messages dynamically

STYLE-ADAPTIVE PROMPTS:
-----------------------
Our prompts are designed to MATCH historical ticket style, not invent.
Key principles:
  - Preserve links and references from similar tickets
  - Match the length of similar tickets
  - Don't add sections that don't exist in examples

================================================================================
"""

from dataclasses import dataclass

from langchain_core.prompts import ChatPromptTemplate, PromptTemplate


# =============================================================================
# SECTION 1: PROMPT BUNDLE DATACLASS
# =============================================================================


@dataclass(frozen=True)
class PromptBundle:
    """
    Collection of prompts used across the generation pipeline.
    
    WHY A BUNDLE?
    All prompts are related and should be versioned together.
    This makes it easy to:
      - Swap prompt sets for A/B testing
      - Load from configuration files
      - Ensure consistency across pipeline steps
    """
    system: str
    draft: str
    draft_brief: str
    product_refine: str
    technical_refine: str
    qa_refine: str


# =============================================================================
# SECTION 2: DEFAULT PROMPTS
# =============================================================================


def get_default_prompts() -> PromptBundle:
    """
    Get the default prompt templates.
    
    DESIGN PRINCIPLES:
    ------------------
    1. MATCH STYLE: Model should mimic historical tickets, not create new styles
    2. PRESERVE CONTENT: Links, references must be copied exactly
    3. MATCH LENGTH: Brief tickets stay brief
    4. DON'T INVENT: Only add what's in similar tickets
    
    INTERVIEW TALKING POINT:
    "I discovered that 'create' and 'improve' language causes over-generation.
    The fix was style-adaptive prompting that instructs the model to MATCH
    the format of retrieved similar tickets."
    
    Returns:
        PromptBundle with all prompt templates
    """
    return PromptBundle(
        # =====================================================================
        # SYSTEM PROMPT
        # =====================================================================
        # Sets the overall behavior for all generations
        system=(
            "You are an expert at writing Jira tickets that match organizational standards.\n\n"
            
            "CORE PRINCIPLES:\n"
            "1. MATCH STYLE: Your PRIMARY goal is to produce tickets that look like "
            "existing tickets in this project.\n"
            "2. PRESERVE CONTENT: Copy all URLs, links, and specific references from "
            "similar tickets exactly.\n"
            "3. MATCH LENGTH: If similar tickets are brief (under 500 chars), yours must be too.\n"
            "4. DON'T INVENT: Never add sections (like 'Acceptance Criteria') if similar "
            "tickets don't have them.\n"
            "5. USE SAME FORMAT: If similar tickets use 'h3.' headers, use 'h3.' headers. "
            "If they use '##', use '##'.\n\n"
            
            "Output MUST be valid Jira markdown.\n"
        ),
        
        # =====================================================================
        # DRAFT PROMPT (Verbose Style)
        # =====================================================================
        # For tickets that should be detailed
        draft=(
            "Generate a Jira ticket that MATCHES THE STYLE of similar historical tickets.\n\n"
            
            "USER REQUEST:\n{user_request}\n\n"
            
            "SIMILAR HISTORICAL TICKETS (use these as your TEMPLATE):\n{rag_context}\n\n"
            
            "CRITICAL RULES:\n"
            "1. STRUCTURE: Use the EXACT same sections as the similar tickets above.\n"
            "   - If they have 'h2. Overview' use that. If they have '## Overview' use that.\n"
            "   - If they DON'T have 'Acceptance Criteria', DON'T add it.\n"
            "2. LENGTH: Match the length of similar tickets. If they're 10 lines, yours is ~10 lines.\n"
            "3. LINKS: Preserve ALL links from similar tickets that are relevant.\n"
            "   - Copy the exact format: [Display Text|URL]\n"
            "   - If similar tickets reference Confluence/SharePoint, include those links.\n"
            "4. SPECIFICITY: Use the same level of detail as similar tickets.\n"
            "   - If they just list bullet points, you list bullet points.\n"
            "   - If they have detailed steps, you have detailed steps.\n"
            "5. NO PLACEHOLDERS: Never use [Date], [Version], or similar placeholders.\n"
            "   - Either include real values or omit the field entirely.\n\n"
            
            "Generate the ticket now, matching the style of the similar tickets above:"
        ),
        
        # =====================================================================
        # DRAFT PROMPT (Brief Style)
        # =====================================================================
        # For concise tickets like data modeling tasks
        draft_brief=(
            "Generate a BRIEF Jira ticket matching the concise style of similar tickets.\n\n"
            
            "USER REQUEST:\n{user_request}\n\n"
            
            "SIMILAR HISTORICAL TICKETS:\n{rag_context}\n\n"
            
            "RULES FOR BRIEF TICKETS:\n"
            "1. Keep it SHORT - similar tickets are brief, yours must be too.\n"
            "2. Use bullet points or numbered lists, not elaborate prose.\n"
            "3. PRESERVE ALL LINKS from similar tickets exactly as written.\n"
            "4. NO 'Overview' or 'Acceptance Criteria' sections unless similar tickets have them.\n"
            "5. Focus on WHAT to do, not WHY (brief tickets assume reader knows context).\n\n"
            
            "Generate a brief ticket matching the style above:"
        ),
        
        # =====================================================================
        # PRODUCT AGENT
        # =====================================================================
        # Preserves brevity, ensures business context is clear
        product_refine=(
            "You are the PRODUCT agent reviewing this Jira ticket.\n\n"
            
            "Your job is to ensure business context is CLEAR, not to ADD content.\n\n"
            
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            
            "RULES:\n"
            "1. If the draft is already brief and clear, RETURN IT UNCHANGED.\n"
            "2. Only add context if the ticket is confusing without it.\n"
            "3. NEVER make a short ticket longer - brevity is a feature.\n"
            "4. PRESERVE all links exactly as written.\n"
            "5. Keep the SAME structure and sections.\n\n"
            
            "Return the ticket (modified only if necessary):"
        ),
        
        # =====================================================================
        # TECHNICAL AGENT
        # =====================================================================
        # Respects original scope, doesn't invent details
        technical_refine=(
            "You are the TECHNICAL agent reviewing this Jira ticket.\n\n"
            
            "Your job is to ensure technical clarity, not to INVENT implementation details.\n\n"
            
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            
            "RULES:\n"
            "1. If the draft is a design/modeling task, DON'T add implementation steps.\n"
            "2. If the draft is brief, KEEP IT BRIEF.\n"
            "3. Only add technical details if they're MISSING and NECESSARY.\n"
            "4. NEVER invent systems, APIs, or tools not mentioned in the request.\n"
            "5. PRESERVE all links exactly as written.\n\n"
            
            "Return the ticket (modified only if necessary):"
        ),
        
        # =====================================================================
        # QA AGENT
        # =====================================================================
        # Conditional on whether AC exists
        qa_refine=(
            "You are the QA agent reviewing this Jira ticket.\n\n"
            
            "Your job is to ensure testability IF acceptance criteria exist.\n\n"
            
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            
            "RULES:\n"
            "1. If the draft has NO Acceptance Criteria section, DON'T ADD ONE.\n"
            "   - Many valid tickets (especially design tasks) don't need AC.\n"
            "2. If AC exists, ensure each item is testable (avoid vague words).\n"
            "3. Use '- [ ]' checkbox format only if original draft uses it.\n"
            "4. PRESERVE all links exactly as written.\n"
            "5. Keep the SAME length and structure.\n\n"
            
            "Return the ticket (modified only if necessary):"
        ),
    )


# =============================================================================
# SECTION 3: LANGCHAIN PROMPT TEMPLATES
# =============================================================================


def create_draft_prompt() -> ChatPromptTemplate:
    """
    Create a LangChain ChatPromptTemplate for draft generation.
    
    LANGCHAIN ADVANTAGE:
    ChatPromptTemplate validates variables and formats messages properly.
    It can be composed with other components using LCEL (| operator).
    
    Returns:
        ChatPromptTemplate for draft generation
        
    Example:
        >>> prompt = create_draft_prompt()
        >>> messages = prompt.invoke({
        ...     "user_request": "Create ticket for...",
        ...     "rag_context": "Similar tickets..."
        ... })
    """
    prompts = get_default_prompts()
    
    return ChatPromptTemplate.from_messages([
        ("system", prompts.system),
        ("human", prompts.draft),
    ])


def create_brief_draft_prompt() -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for brief-style drafts.
    
    Returns:
        ChatPromptTemplate for brief draft generation
    """
    prompts = get_default_prompts()
    
    return ChatPromptTemplate.from_messages([
        ("system", prompts.system),
        ("human", prompts.draft_brief),
    ])


def create_refinement_prompt(agent_type: str) -> ChatPromptTemplate:
    """
    Create a ChatPromptTemplate for agent refinement.
    
    Args:
        agent_type: One of "product", "technical", "qa"
        
    Returns:
        ChatPromptTemplate for the specified agent
        
    Raises:
        ValueError: If agent_type is not recognized
    """
    prompts = get_default_prompts()
    
    agent_prompts = {
        "product": prompts.product_refine,
        "technical": prompts.technical_refine,
        "qa": prompts.qa_refine,
    }
    
    if agent_type not in agent_prompts:
        raise ValueError(
            f"Unknown agent type: {agent_type}. "
            f"Must be one of: {list(agent_prompts.keys())}"
        )
    
    return ChatPromptTemplate.from_messages([
        ("system", prompts.system),
        ("human", agent_prompts[agent_type]),
    ])


# =============================================================================
# SECTION 4: STYLE DETECTION
# =============================================================================


def detect_ticket_style(chunks: list, user_request: str = "") -> str:
    """
    Analyze retrieved chunks to determine brief or verbose style.
    
    STYLE DETECTION LOGIC:
    1. Calculate average length of retrieved tickets
    2. Check for brief-style keywords
    3. Check for verbose structure markers
    
    Args:
        chunks: Retrieved document chunks
        user_request: The user's request text
        
    Returns:
        "brief" or "verbose"
    """
    if not chunks:
        return "verbose"  # Default when no context
    
    # Calculate average length
    total_length = sum(
        len(getattr(ch, 'text', getattr(ch, 'page_content', str(ch))))
        for ch in chunks
    )
    avg_length = total_length / len(chunks)
    
    # Brief if chunks are short
    if avg_length < 600:
        return "brief"
    
    # Check for brief-style keywords
    brief_keywords = [
        "data model", "schema design", "lucid chart", "diagram",
        "column names", "column types", "primary key", "foreign key"
    ]
    
    combined_text = user_request.lower()
    for ch in chunks:
        text = getattr(ch, 'text', getattr(ch, 'page_content', str(ch)))
        combined_text += " " + text.lower()
    
    if any(kw in combined_text for kw in brief_keywords):
        if avg_length < 1000:
            return "brief"
    
    # Check for verbose structure markers
    has_overview = any(
        "overview" in getattr(ch, 'text', getattr(ch, 'page_content', str(ch))).lower()
        for ch in chunks
    )
    has_acceptance = any(
        "acceptance" in getattr(ch, 'text', getattr(ch, 'page_content', str(ch))).lower()
        for ch in chunks
    )
    
    if has_overview and has_acceptance:
        return "verbose"
    
    return "brief" if avg_length < 800 else "verbose"


def should_skip_refinement(draft: str, style: str) -> bool:
    """
    Determine if multi-agent refinement should be skipped.
    
    SKIP REFINEMENT WHEN:
    1. Draft is already brief (under 800 chars) and style is "brief"
    2. Draft is very short with links (don't over-process)
    
    Args:
        draft: The generated draft text
        style: "brief" or "verbose"
        
    Returns:
        True if refinement should be skipped
    """
    if style == "brief":
        # For brief tickets, only refine if draft got too long
        return len(draft) < 800
    
    # For verbose, only skip if draft is very short with links
    if len(draft) < 400 and "http" in draft:
        return True
    
    return False


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. LangChain ChatPromptTemplate for structured prompts
# 2. Style-adaptive prompting (brief vs verbose)
# 3. Multi-agent refinement prompts
# 4. Prompt bundle pattern for versioning
#
# KEY INSIGHT:
# "Different LLM models require model-specific prompt engineering.
# Developing with one model and deploying with another is problematic."
#
# INTERVIEW TALKING POINTS:
# - "Our prompts instruct the model to MATCH, not CREATE"
# - "Style detection prevents over-elaboration of brief tickets"
# - "Multi-agent refinement is conditional based on ticket complexity"
#
# =============================================================================
