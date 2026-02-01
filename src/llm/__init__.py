"""
================================================================================
LLM Package - Language Model Integration
================================================================================

This package provides LangChain-based integration with Google's Gemini models.

COMPONENTS:
- gemini: ChatGoogleGenerativeAI wrapper for text generation
- prompts: Prompt templates for ticket generation and refinement

================================================================================
"""

from src.llm.gemini import (
    create_llm,
    get_llm,
    clear_cache,
    generate,
    agenerate,
    generate_with_messages,
    stream,
    astream,
)

from src.llm.prompts import (
    PromptBundle,
    create_draft_prompt,
    create_brief_draft_prompt,
    create_refinement_prompt,
    detect_ticket_style,
    should_skip_refinement,
)

__all__ = [
    # Gemini LLM
    "create_llm",
    "get_llm",
    "clear_cache",
    "generate",
    "agenerate",
    "generate_with_messages",
    "stream",
    "astream",
    "PromptBundle",
    "create_draft_prompt",
    "create_brief_draft_prompt",
    "create_refinement_prompt",
    "detect_ticket_style",
    "should_skip_refinement",
]
