"""
================================================================================
TUTORIAL: LangChain Gemini LLM Integration
================================================================================

LANGCHAIN LLM ABSTRACTIONS:
---------------------------
LangChain provides two main abstractions for language models:

1. LLM (legacy): Simple text-in, text-out
   - Input: String prompt
   - Output: String response

2. ChatModel (preferred): Message-based
   - Input: List of messages (system, human, ai)
   - Output: AI message
   - Supports: System prompts, conversation history, structured output

CHATGOOGLEGENERATIVEAI:
-----------------------
This is LangChain's chat model wrapper for Gemini:
  - Supports system instructions
  - Handles message formatting
  - Built-in retry logic
  - Streaming support

WHY CHAT MODELS?
----------------
Chat models are preferred because:
  1. System prompts set model behavior
  2. Conversation history enables multi-turn
  3. Role separation (human vs ai) is explicit
  4. Better for instruction-following tasks

GEMINI 2.5 FLASH:
-----------------
We use gemini-2.5-flash-preview-05-20 which is:
  - The cheapest Gemini 2.5 model
  - Fast inference (flash variant)
  - Good balance of cost and capability

================================================================================
"""

import logging
from functools import lru_cache
from typing import Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: LLM FACTORY
# =============================================================================


def create_llm(
    api_key: Optional[str] = None,
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> ChatGoogleGenerativeAI:
    """
    Create a configured Gemini chat model.
    
    CHATGOOGLEGENERATIVEAI FEATURES:
    --------------------------------
    - model: The Gemini model to use
    - temperature: Randomness (0=deterministic, 1=creative)
    - max_output_tokens: Response length limit
    - convert_system_message_to_human: Convert system to human (for models
      that don't support system messages natively)
    
    Args:
        api_key: Google API key (optional, uses settings)
        model: Model name (optional, uses settings)
        temperature: Generation temperature (optional, uses settings)
        max_tokens: Max output tokens (optional, uses settings)
        
    Returns:
        Configured ChatGoogleGenerativeAI instance
        
    Example:
        >>> llm = create_llm()
        >>> response = llm.invoke("What is Python?")
        >>> print(response.content)
    """
    # Use settings defaults if not provided
    api_key = api_key or settings.google_api_key
    model = model or settings.llm.model
    temperature = temperature if temperature is not None else settings.llm.temperature
    max_tokens = max_tokens or settings.llm.max_tokens
    
    if not api_key:
        raise ValueError(
            "Google API key is required. "
            "Set GOOGLE_API_KEY environment variable."
        )
    
    logger.debug(
        f"Creating Gemini LLM: model={model}, "
        f"temp={temperature}, max_tokens={max_tokens}"
    )
    
    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=temperature,
        max_output_tokens=max_tokens,
        # Don't convert system messages - Gemini supports them
        convert_system_message_to_human=False,
    )


# =============================================================================
# SECTION 2: CACHED SINGLETON
# =============================================================================


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    """
    Get a cached singleton LLM instance.
    
    WHY CACHE?
    LLM instances are stateless but have setup overhead.
    Caching avoids repeated initialization.
    
    Returns:
        Cached ChatGoogleGenerativeAI instance
    """
    return create_llm()


def clear_cache():
    """Clear the cached LLM instance."""
    get_llm.cache_clear()


# =============================================================================
# SECTION 3: CONVENIENCE FUNCTIONS
# =============================================================================


def generate(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Generate a response from a prompt.
    
    This is the simplest way to get a response from the LLM.
    For more control, use the LLM directly with message objects.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system instruction
        
    Returns:
        Generated text response
        
    Example:
        >>> response = generate(
        ...     prompt="Write a haiku about Python",
        ...     system_prompt="You are a poet."
        ... )
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""
    
    llm = get_llm()
    
    # Build messages
    messages: list = []
    
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    
    messages.append(HumanMessage(content=prompt))
    
    # Generate response
    response = llm.invoke(messages)
    
    return response.content


async def agenerate(
    prompt: str,
    system_prompt: Optional[str] = None,
) -> str:
    """
    Async version of generate.
    
    Use in async contexts (FastAPI handlers, etc.).
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system instruction
        
    Returns:
        Generated text response
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return ""
    
    llm = get_llm()
    
    messages: list = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    response = await llm.ainvoke(messages)
    return response.content


def generate_with_messages(
    messages: list[dict[str, str]],
) -> str:
    """
    Generate from a list of message dicts.
    
    Useful when you have pre-formatted messages.
    
    Args:
        messages: List of {"role": "...", "content": "..."} dicts
                 Roles: "system", "human"/"user", "ai"/"assistant"
        
    Returns:
        Generated text response
        
    Example:
        >>> messages = [
        ...     {"role": "system", "content": "You are helpful."},
        ...     {"role": "human", "content": "Hello!"}
        ... ]
        >>> response = generate_with_messages(messages)
    """
    llm = get_llm()
    
    # Convert dicts to LangChain messages
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "human").lower()
        content = msg.get("content", "")
        
        if role in ("system",):
            lc_messages.append(SystemMessage(content=content))
        elif role in ("human", "user"):
            lc_messages.append(HumanMessage(content=content))
        elif role in ("ai", "assistant"):
            lc_messages.append(AIMessage(content=content))
        else:
            # Default to human
            lc_messages.append(HumanMessage(content=content))
    
    response = llm.invoke(lc_messages)
    return response.content


# =============================================================================
# SECTION 4: STREAMING SUPPORT
# =============================================================================


def stream(
    prompt: str,
    system_prompt: Optional[str] = None,
):
    """
    Stream response tokens.
    
    STREAMING:
    Instead of waiting for the full response, streaming
    yields tokens as they're generated. This improves
    perceived latency for long responses.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system instruction
        
    Yields:
        Response chunks (strings)
        
    Example:
        >>> for chunk in stream("Write a long story"):
        ...     print(chunk, end="", flush=True)
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return
    
    llm = get_llm()
    
    messages: list = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    for chunk in llm.stream(messages):
        if chunk.content:
            yield chunk.content


async def astream(
    prompt: str,
    system_prompt: Optional[str] = None,
):
    """
    Async streaming version.
    
    Args:
        prompt: The user prompt
        system_prompt: Optional system instruction
        
    Yields:
        Response chunks (strings)
    """
    prompt = (prompt or "").strip()
    if not prompt:
        return
    
    llm = get_llm()
    
    messages: list = []
    if system_prompt:
        messages.append(SystemMessage(content=system_prompt))
    messages.append(HumanMessage(content=prompt))
    
    async for chunk in llm.astream(messages):
        if chunk.content:
            yield chunk.content


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. ChatGoogleGenerativeAI for Gemini chat models
# 2. Message-based API (system, human, ai messages)
# 3. Sync and async generation methods
# 4. Streaming for improved UX
#
# LANGCHAIN INTEGRATION:
# - Uses official langchain-google-genai package
# - Compatible with LangChain chains and LCEL
# - Supports tool calling (for future agents)
#
# MODEL SELECTION:
# - gemini-2.5-flash-preview-05-20: Cheapest, fast, good quality
# - gemini-2.5-pro-preview-05-06: More capable, higher cost
#
# INTERVIEW TALKING POINTS:
# - "We use chat models for better instruction following"
# - "System prompts set the model's persona and constraints"
# - "Streaming improves perceived latency for users"
#
# =============================================================================
