"""
================================================================================
TUTORIAL: LCEL Generation Chain for Jira Tickets
================================================================================

LANGCHAIN EXPRESSION LANGUAGE (LCEL):
-------------------------------------
LCEL is LangChain's declarative way to build chains. The key concept is
composability using the pipe operator (|):

    chain = prompt | llm | output_parser

Each component in the chain:
1. Receives input from the previous component
2. Processes it
3. Passes output to the next component

RUNNABLE PROTOCOL:
------------------
All LCEL components implement the Runnable protocol:
- invoke(input): Synchronous execution
- ainvoke(input): Async execution
- stream(input): Streaming execution
- batch(inputs): Batch processing

RUNNABLEPASSTHROUGH:
--------------------
RunnablePassthrough passes input unchanged, useful for:
- Including original input in chain output
- Combining with other Runnables using RunnableParallel

RUNNABLELAMBDA:
---------------
Wraps arbitrary functions as Runnables:
    process = RunnableLambda(my_function)
    chain = input | process | output

OUR GENERATION CHAIN:
---------------------
The ticket generation chain:
1. Takes user request and retrieved context
2. Detects ticket style (brief vs verbose)
3. Selects appropriate prompt template
4. Generates draft ticket with LLM
5. Parses and validates output

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
    RunnablePassthrough,
    RunnableParallel,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.llm import (
    create_llm,
    get_llm,
    create_draft_prompt,
    create_brief_draft_prompt,
    detect_ticket_style,
    PROMPTS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: DATA CLASSES
# =============================================================================


@dataclass
class GenerationInput:
    """
    Input for the generation chain.
    
    Attributes:
        user_request: The user's description of the ticket they want
        retrieved_context: Formatted context from similar tickets
        project_key: Jira project key (e.g., "CSCI")
        issue_type: Type of issue (e.g., "Story", "Task", "Bug")
    """
    user_request: str
    retrieved_context: str
    project_key: str = "CSCI"
    issue_type: str = "Story"
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for chain input."""
        return {
            "user_request": self.user_request,
            "context": self.retrieved_context,
            "project_key": self.project_key,
            "issue_type": self.issue_type,
        }


@dataclass
class GenerationOutput:
    """
    Output from the generation chain.
    
    Attributes:
        draft: The generated ticket draft
        style: Detected style (brief/verbose)
        prompt_used: Which prompt template was used
        metadata: Additional generation metadata
    """
    draft: str
    style: str = "standard"
    prompt_used: str = "draft"
    metadata: dict = field(default_factory=dict)


# =============================================================================
# SECTION 2: CHAIN CONSTRUCTION
# =============================================================================


class TicketGenerationChain:
    """
    LCEL chain for generating Jira ticket drafts.
    
    ARCHITECTURE:
    -------------
    The chain is style-adaptive:
    
    1. If retrieved tickets are brief → Use brief prompt
       - Preserves concise style
       - Minimal elaboration
    
    2. If retrieved tickets are verbose → Use standard prompt
       - Full context utilization
       - Detailed descriptions
    
    This ensures generated tickets match the project's style.
    
    Example:
        >>> chain = TicketGenerationChain()
        >>> result = chain.invoke(GenerationInput(
        ...     user_request="Add user authentication",
        ...     retrieved_context="[Similar tickets here]"
        ... ))
        >>> print(result.draft)
    """
    
    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        """
        Initialize the generation chain.
        
        Args:
            llm: Optional LLM instance (uses singleton if not provided)
        """
        self.llm = llm or get_llm()
        self._chain: Optional[Runnable] = None
        
        # Pre-build prompts
        self._draft_prompt = create_draft_prompt()
        self._brief_prompt = create_brief_draft_prompt()
    
    def _detect_style(self, context: str) -> str:
        """
        Detect the style from retrieved context.
        
        Returns: "brief" or "verbose"
        """
        return detect_ticket_style(context)
    
    def _select_prompt(self, style: str) -> ChatPromptTemplate:
        """
        Select appropriate prompt based on style.
        
        Args:
            style: "brief" or "verbose"
            
        Returns:
            The appropriate ChatPromptTemplate
        """
        if style == "brief":
            return self._brief_prompt
        return self._draft_prompt
    
    def _build_chain(self, prompt: ChatPromptTemplate) -> Runnable:
        """
        Build an LCEL chain for the given prompt.
        
        LCEL BREAKDOWN:
        ---------------
        prompt | self.llm | StrOutputParser()
        
        1. prompt: Formats input variables into messages
        2. self.llm: Generates response from messages
        3. StrOutputParser(): Extracts string content from AIMessage
        
        Args:
            prompt: The ChatPromptTemplate to use
            
        Returns:
            Runnable chain
        """
        return prompt | self.llm | StrOutputParser()
    
    def invoke(self, input_data: GenerationInput) -> GenerationOutput:
        """
        Generate a ticket draft synchronously.
        
        Args:
            input_data: Generation input with request and context
            
        Returns:
            GenerationOutput with draft and metadata
        """
        logger.info("Starting ticket generation")
        
        # Detect style from context
        style = self._detect_style(input_data.retrieved_context)
        logger.debug(f"Detected style: {style}")
        
        # Select prompt based on style
        prompt = self._select_prompt(style)
        prompt_name = "brief" if style == "brief" else "draft"
        
        # Build and run chain
        chain = self._build_chain(prompt)
        
        try:
            draft = chain.invoke(input_data.to_dict())
            
            return GenerationOutput(
                draft=draft.strip(),
                style=style,
                prompt_used=prompt_name,
                metadata={
                    "model": settings.llm.model,
                    "context_length": len(input_data.retrieved_context),
                }
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise
    
    async def ainvoke(self, input_data: GenerationInput) -> GenerationOutput:
        """
        Generate a ticket draft asynchronously.
        
        Args:
            input_data: Generation input with request and context
            
        Returns:
            GenerationOutput with draft and metadata
        """
        logger.info("Starting async ticket generation")
        
        style = self._detect_style(input_data.retrieved_context)
        prompt = self._select_prompt(style)
        prompt_name = "brief" if style == "brief" else "draft"
        
        chain = self._build_chain(prompt)
        
        try:
            draft = await chain.ainvoke(input_data.to_dict())
            
            return GenerationOutput(
                draft=draft.strip(),
                style=style,
                prompt_used=prompt_name,
                metadata={
                    "model": settings.llm.model,
                    "context_length": len(input_data.retrieved_context),
                }
            )
            
        except Exception as e:
            logger.error(f"Async generation failed: {e}")
            raise


# =============================================================================
# SECTION 3: FACTORY FUNCTIONS
# =============================================================================


def create_generation_chain(
    llm: Optional[ChatGoogleGenerativeAI] = None,
) -> TicketGenerationChain:
    """
    Create a ticket generation chain.
    
    Args:
        llm: Optional LLM instance
        
    Returns:
        Configured TicketGenerationChain
        
    Example:
        >>> chain = create_generation_chain()
        >>> result = chain.invoke(input_data)
    """
    return TicketGenerationChain(llm=llm)


# =============================================================================
# SECTION 4: CONVENIENCE FUNCTIONS
# =============================================================================


def generate_ticket_draft(
    user_request: str,
    retrieved_context: str,
    project_key: str = "CSCI",
    issue_type: str = "Story",
) -> GenerationOutput:
    """
    Generate a ticket draft (convenience function).
    
    This is the simplest way to generate a ticket. For more control,
    use the TicketGenerationChain class directly.
    
    Args:
        user_request: User's ticket description
        retrieved_context: Context from similar tickets
        project_key: Jira project key
        issue_type: Issue type
        
    Returns:
        GenerationOutput with the draft
        
    Example:
        >>> result = generate_ticket_draft(
        ...     user_request="Add login button",
        ...     retrieved_context="[Similar tickets...]"
        ... )
        >>> print(result.draft)
    """
    chain = create_generation_chain()
    
    input_data = GenerationInput(
        user_request=user_request,
        retrieved_context=retrieved_context,
        project_key=project_key,
        issue_type=issue_type,
    )
    
    return chain.invoke(input_data)


async def agenerate_ticket_draft(
    user_request: str,
    retrieved_context: str,
    project_key: str = "CSCI",
    issue_type: str = "Story",
) -> GenerationOutput:
    """
    Generate a ticket draft asynchronously.
    
    Args:
        user_request: User's ticket description
        retrieved_context: Context from similar tickets
        project_key: Jira project key
        issue_type: Issue type
        
    Returns:
        GenerationOutput with the draft
    """
    chain = create_generation_chain()
    
    input_data = GenerationInput(
        user_request=user_request,
        retrieved_context=retrieved_context,
        project_key=project_key,
        issue_type=issue_type,
    )
    
    return await chain.ainvoke(input_data)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. LCEL basics: prompt | llm | parser
# 2. Runnable protocol: invoke, ainvoke
# 3. Style-adaptive generation
# 4. Chain encapsulation in a class
#
# KEY PATTERNS:
# - Detect style from context, not from user request
# - Use appropriate prompt for the detected style
# - Return structured output (dataclass) not raw strings
# - Log at appropriate levels (info for start, debug for details)
#
# INTERVIEW TALKING POINTS:
# - "LCEL provides declarative chain composition"
# - "Style-adaptive generation prevents over-verbose tickets"
# - "The Runnable protocol enables both sync and async execution"
# - "We detect style from historical data, not user input"
#
# NEXT STEP:
# The refinement chain (refinement.py) takes the draft and improves it
# through multiple agent perspectives (Product, Technical, QA).
#
# =============================================================================
