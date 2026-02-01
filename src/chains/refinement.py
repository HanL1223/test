"""
================================================================================
TUTORIAL: LCEL Multi-Agent Refinement Chain
================================================================================

MULTI-AGENT REFINEMENT:
-----------------------
Instead of a single LLM pass, we use multiple "agent" perspectives:

1. PRODUCT OWNER: Ensures business value is clear
   - User stories follow proper format
   - Acceptance criteria are testable
   - Business context is captured

2. TECHNICAL LEAD: Adds implementation details
   - Technical requirements are specific
   - Dependencies are identified
   - Non-functional requirements noted

3. QA ENGINEER: Validates testability
   - Edge cases are considered
   - Test scenarios are identified
   - Acceptance criteria are measurable

SEQUENTIAL REFINEMENT:
----------------------
Each agent refines the output of the previous:

    Draft → Product → Technical → QA → Final

This iterative approach produces higher quality tickets.

CONDITIONAL REFINEMENT:
-----------------------
Not all tickets need full refinement:
- Brief tickets should stay brief
- Simple tasks don't need QA review

We check should_skip_refinement() before each agent.

LCEL PATTERN: RunnableSequence
------------------------------
We chain multiple refinement steps:

    chain = (
        product_refine 
        | RunnableLambda(extract_content)
        | technical_refine
        | RunnableLambda(extract_content)
        | qa_refine
    )

================================================================================
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import (
    Runnable,
    RunnableLambda,
)
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import settings
from src.llm import (
    get_llm,
    create_refinement_prompt,
    should_skip_refinement,
    PROMPTS,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: ENUMS AND DATA CLASSES
# =============================================================================


class RefinerRole(str, Enum):
    """
    Refinement agent roles.
    
    Each role brings a different perspective to ticket quality.
    """
    PRODUCT = "product"
    TECHNICAL = "technical"
    QA = "qa"
    
    @property
    def display_name(self) -> str:
        """Human-readable role name."""
        return {
            RefinerRole.PRODUCT: "Product Owner",
            RefinerRole.TECHNICAL: "Technical Lead",
            RefinerRole.QA: "QA Engineer",
        }[self]


@dataclass
class RefinementInput:
    """
    Input for the refinement chain.
    
    Attributes:
        draft: The ticket draft to refine
        context: Retrieved context for reference
        style: Detected ticket style (brief/verbose)
        roles: Which refinement roles to apply
    """
    draft: str
    context: str
    style: str = "standard"
    roles: list[RefinerRole] = field(
        default_factory=lambda: [
            RefinerRole.PRODUCT,
            RefinerRole.TECHNICAL,
            RefinerRole.QA,
        ]
    )
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for chain input."""
        return {
            "draft": self.draft,
            "context": self.context,
        }


@dataclass
class RefinementStep:
    """
    Record of a single refinement step.
    
    Attributes:
        role: Which agent performed the refinement
        input_text: Text before refinement
        output_text: Text after refinement
        skipped: Whether this step was skipped
        reason: Reason for skipping (if skipped)
    """
    role: RefinerRole
    input_text: str
    output_text: str
    skipped: bool = False
    reason: Optional[str] = None


@dataclass
class RefinementOutput:
    """
    Output from the refinement chain.
    
    Attributes:
        final_ticket: The fully refined ticket
        steps: List of refinement steps taken
        total_steps: Number of non-skipped steps
        metadata: Additional refinement metadata
    """
    final_ticket: str
    steps: list[RefinementStep] = field(default_factory=list)
    total_steps: int = 0
    metadata: dict = field(default_factory=dict)
    
    @property
    def was_refined(self) -> bool:
        """Whether any refinement was applied."""
        return self.total_steps > 0


# =============================================================================
# SECTION 2: SINGLE ROLE REFINER
# =============================================================================


class RoleRefiner:
    """
    A single-role refiner using LCEL.
    
    Each RoleRefiner wraps one perspective (Product/Technical/QA)
    and applies that refinement to a draft.
    
    Example:
        >>> refiner = RoleRefiner(RefinerRole.PRODUCT)
        >>> output = refiner.invoke({"draft": "...", "context": "..."})
    """
    
    def __init__(
        self,
        role: RefinerRole,
        llm: Optional[ChatGoogleGenerativeAI] = None,
    ):
        """
        Initialize a role refiner.
        
        Args:
            role: The refinement perspective
            llm: Optional LLM instance
        """
        self.role = role
        self.llm = llm or get_llm()
        
        # Create the refinement prompt for this role
        self.prompt = create_refinement_prompt(role.value)
        
        # Build the LCEL chain
        self._chain = self.prompt | self.llm | StrOutputParser()
    
    def invoke(self, input_dict: dict[str, str]) -> str:
        """
        Apply refinement synchronously.
        
        Args:
            input_dict: Dict with "draft" and "context" keys
            
        Returns:
            Refined text
        """
        logger.debug(f"Applying {self.role.display_name} refinement")
        return self._chain.invoke(input_dict)
    
    async def ainvoke(self, input_dict: dict[str, str]) -> str:
        """
        Apply refinement asynchronously.
        
        Args:
            input_dict: Dict with "draft" and "context" keys
            
        Returns:
            Refined text
        """
        logger.debug(f"Applying {self.role.display_name} refinement (async)")
        return await self._chain.ainvoke(input_dict)


# =============================================================================
# SECTION 3: MULTI-ROLE REFINEMENT CHAIN
# =============================================================================


class RefinementChain:
    """
    Multi-agent refinement chain.
    
    WORKFLOW:
    ---------
    1. Check if refinement should be skipped (brief style)
    2. For each enabled role:
       a. Check if this specific role should be skipped
       b. Apply refinement if not skipped
       c. Record the step
    3. Return final refined ticket with step history
    
    CONDITIONAL LOGIC:
    ------------------
    - Brief tickets skip elaboration agents
    - Simple tasks may skip QA review
    - Each step can be independently controlled
    
    Example:
        >>> chain = RefinementChain()
        >>> result = chain.invoke(RefinementInput(
        ...     draft="Initial draft...",
        ...     context="Similar tickets...",
        ...     style="verbose"
        ... ))
        >>> print(result.final_ticket)
    """
    
    def __init__(
        self,
        llm: Optional[ChatGoogleGenerativeAI] = None,
        roles: Optional[list[RefinerRole]] = None,
    ):
        """
        Initialize the refinement chain.
        
        Args:
            llm: Optional LLM instance
            roles: List of roles to apply (default: all three)
        """
        self.llm = llm or get_llm()
        self.roles = roles or [
            RefinerRole.PRODUCT,
            RefinerRole.TECHNICAL,
            RefinerRole.QA,
        ]
        
        # Create refiners for each role
        self._refiners = {
            role: RoleRefiner(role, self.llm)
            for role in self.roles
        }
    
    def _should_skip_role(
        self,
        role: RefinerRole,
        current_text: str,
        style: str,
    ) -> tuple[bool, Optional[str]]:
        """
        Determine if a role should be skipped.
        
        SKIP LOGIC:
        -----------
        - Brief style: Skip Product (would add verbosity)
        - Very short text: Skip QA (not enough to validate)
        - Already detailed: Maybe skip Technical
        
        Args:
            role: The role to check
            current_text: Current ticket text
            style: Detected style
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Check general refinement skip
        if should_skip_refinement(current_text):
            return True, "Text too brief for refinement"
        
        # Brief style specific skips
        if style == "brief":
            if role == RefinerRole.PRODUCT:
                return True, "Brief style: skip product elaboration"
            if role == RefinerRole.QA:
                return True, "Brief style: skip QA elaboration"
        
        # Word count based skips
        word_count = len(current_text.split())
        if word_count < 20 and role == RefinerRole.QA:
            return True, f"Too brief ({word_count} words) for QA review"
        
        return False, None
    
    def invoke(self, input_data: RefinementInput) -> RefinementOutput:
        """
        Apply multi-agent refinement synchronously.
        
        Args:
            input_data: RefinementInput with draft and context
            
        Returns:
            RefinementOutput with final ticket and step history
        """
        logger.info(f"Starting refinement with {len(input_data.roles)} roles")
        
        current_text = input_data.draft
        steps: list[RefinementStep] = []
        total_applied = 0
        
        for role in input_data.roles:
            if role not in self._refiners:
                logger.warning(f"Role {role} not configured, skipping")
                continue
            
            # Check if we should skip this role
            should_skip, skip_reason = self._should_skip_role(
                role, current_text, input_data.style
            )
            
            if should_skip:
                logger.debug(f"Skipping {role.display_name}: {skip_reason}")
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=current_text,
                    skipped=True,
                    reason=skip_reason,
                ))
                continue
            
            # Apply refinement
            refiner = self._refiners[role]
            input_dict = {
                "draft": current_text,
                "context": input_data.context,
            }
            
            try:
                refined_text = refiner.invoke(input_dict)
                
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=refined_text.strip(),
                    skipped=False,
                ))
                
                current_text = refined_text.strip()
                total_applied += 1
                
                logger.debug(f"Applied {role.display_name} refinement")
                
            except Exception as e:
                logger.error(f"Refinement failed for {role.display_name}: {e}")
                # Continue with current text on failure
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=current_text,
                    skipped=True,
                    reason=f"Error: {str(e)}",
                ))
        
        return RefinementOutput(
            final_ticket=current_text,
            steps=steps,
            total_steps=total_applied,
            metadata={
                "model": settings.llm.model,
                "initial_style": input_data.style,
                "roles_requested": [r.value for r in input_data.roles],
            }
        )
    
    async def ainvoke(self, input_data: RefinementInput) -> RefinementOutput:
        """
        Apply multi-agent refinement asynchronously.
        
        Args:
            input_data: RefinementInput with draft and context
            
        Returns:
            RefinementOutput with final ticket and step history
        """
        logger.info(f"Starting async refinement with {len(input_data.roles)} roles")
        
        current_text = input_data.draft
        steps: list[RefinementStep] = []
        total_applied = 0
        
        for role in input_data.roles:
            if role not in self._refiners:
                continue
            
            should_skip, skip_reason = self._should_skip_role(
                role, current_text, input_data.style
            )
            
            if should_skip:
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=current_text,
                    skipped=True,
                    reason=skip_reason,
                ))
                continue
            
            refiner = self._refiners[role]
            input_dict = {
                "draft": current_text,
                "context": input_data.context,
            }
            
            try:
                refined_text = await refiner.ainvoke(input_dict)
                
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=refined_text.strip(),
                    skipped=False,
                ))
                
                current_text = refined_text.strip()
                total_applied += 1
                
            except Exception as e:
                logger.error(f"Async refinement failed for {role.display_name}: {e}")
                steps.append(RefinementStep(
                    role=role,
                    input_text=current_text,
                    output_text=current_text,
                    skipped=True,
                    reason=f"Error: {str(e)}",
                ))
        
        return RefinementOutput(
            final_ticket=current_text,
            steps=steps,
            total_steps=total_applied,
            metadata={
                "model": settings.llm.model,
                "initial_style": input_data.style,
                "roles_requested": [r.value for r in input_data.roles],
            }
        )


# =============================================================================
# SECTION 4: FACTORY AND CONVENIENCE FUNCTIONS
# =============================================================================


def create_refinement_chain(
    llm: Optional[ChatGoogleGenerativeAI] = None,
    roles: Optional[list[RefinerRole]] = None,
) -> RefinementChain:
    """
    Create a refinement chain.
    
    Args:
        llm: Optional LLM instance
        roles: List of roles to include
        
    Returns:
        Configured RefinementChain
    """
    return RefinementChain(llm=llm, roles=roles)


def refine_ticket(
    draft: str,
    context: str,
    style: str = "standard",
    roles: Optional[list[RefinerRole]] = None,
) -> RefinementOutput:
    """
    Refine a ticket draft (convenience function).
    
    Args:
        draft: The ticket draft to refine
        context: Retrieved context for reference
        style: Detected ticket style
        roles: Roles to apply (default: all)
        
    Returns:
        RefinementOutput with refined ticket
        
    Example:
        >>> result = refine_ticket(
        ...     draft="Add login feature",
        ...     context="[Similar tickets...]",
        ...     style="verbose"
        ... )
        >>> print(result.final_ticket)
    """
    roles = roles or [
        RefinerRole.PRODUCT,
        RefinerRole.TECHNICAL,
        RefinerRole.QA,
    ]
    
    chain = create_refinement_chain(roles=roles)
    
    input_data = RefinementInput(
        draft=draft,
        context=context,
        style=style,
        roles=roles,
    )
    
    return chain.invoke(input_data)


async def arefine_ticket(
    draft: str,
    context: str,
    style: str = "standard",
    roles: Optional[list[RefinerRole]] = None,
) -> RefinementOutput:
    """
    Refine a ticket draft asynchronously.
    
    Args:
        draft: The ticket draft to refine
        context: Retrieved context for reference
        style: Detected ticket style
        roles: Roles to apply (default: all)
        
    Returns:
        RefinementOutput with refined ticket
    """
    roles = roles or [
        RefinerRole.PRODUCT,
        RefinerRole.TECHNICAL,
        RefinerRole.QA,
    ]
    
    chain = create_refinement_chain(roles=roles)
    
    input_data = RefinementInput(
        draft=draft,
        context=context,
        style=style,
        roles=roles,
    )
    
    return await chain.ainvoke(input_data)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Multi-agent refinement pattern
# 2. Conditional processing based on content
# 3. Step-by-step refinement with history
# 4. Error handling that preserves progress
#
# KEY PATTERNS:
# - Each role brings unique perspective
# - Conditional skipping prevents over-processing
# - Step history enables debugging and tracing
# - Async support for production use
#
# INTERVIEW TALKING POINTS:
# - "Multi-agent refinement improves ticket quality"
# - "Conditional processing adapts to content style"
# - "We track each refinement step for observability"
# - "Brief tickets skip elaboration to maintain style"
#
# ARCHITECTURE NOTE:
# This sequential approach is simple but effective. For complex
# workflows, consider:
# - Parallel refinement with consensus
# - Iterative refinement until quality threshold
# - Human-in-the-loop validation
#
# =============================================================================
