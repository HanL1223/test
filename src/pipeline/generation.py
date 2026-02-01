"""
================================================================================
TUTORIAL: Agentic Generation Pipeline with ReAct Agent
================================================================================

AGENT VS WORKFLOW:
------------------
This module has been refactored from a WORKFLOW to an AGENT approach.

WORKFLOW (before):
    Fixed sequence: Search → Detect → Generate → Refine → Output
    Every request follows the same path regardless of intermediate results.

AGENT (now):
    Dynamic decisions: The agent REASONS about what to do next.
    It adapts based on observations and self-corrects until done.

THE REACT PATTERN:
------------------
The agent follows a Thought → Action → Observation loop:

    while not done:
        thought = "What should I do next?"
        action = choose_tool(thought)  
        observation = execute_tool(action)
        if observation indicates done:
            return final_answer

KEY DIFFERENCES:
----------------
1. Agent DECIDES which tools to use (not predetermined)
2. Agent ADAPTS based on intermediate results
3. Agent SELF-CORRECTS when validation fails
4. Execution path varies per request

PIPELINE CLASS RETAINED:
------------------------
We keep the JiraTicketPipeline class for backward compatibility,
but internally it now uses the ReAct agent.

================================================================================
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from src.config import settings
from src.core.models import RetrievedChunk
from src.agents import (
    JiraTicketAgent,
    AgentConfig,
    AgentResult,
    create_jira_agent,
    get_jira_agent,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: PIPELINE CONFIGURATION
# =============================================================================


@dataclass
class PipelineConfig:
    """
    Configuration for the generation pipeline.
    
    NOTE: This now configures the underlying ReAct agent.
    
    PARAMETERS:
    -----------
    max_iterations: Maximum agent reasoning steps
        - Higher = more thorough but slower/costlier
        - Typical: 5-10 iterations
        
    max_execution_time: Timeout in seconds
        - Prevents runaway costs
        
    verbose: Print agent reasoning to console
        - Useful for debugging
        
    use_agent: If True, use ReAct agent (default)
               If False, fallback to simple chain (legacy)
    """
    max_iterations: int = 10
    max_execution_time: float = 120.0
    verbose: bool = False
    use_agent: bool = True  # Enable agentic behavior
    
    # Legacy config (for backward compatibility)
    top_k: int = 5
    score_threshold: float = 0.3
    enable_refinement: bool = True
    fast_mode: bool = False
    auto_detect_style: bool = True


@dataclass
class GenerationResult:
    """
    Complete result from the generation pipeline.
    
    ENHANCED FOR AGENTS:
    Now includes reasoning_trace to show agent's decision-making.
    
    FIELDS:
    -------
    ticket_text: The final generated ticket content
    draft_text: Initial draft (may differ from final if refined)
    retrieved_chunks: Similar tickets used as context
    style_detected: "brief" or "verbose" (agent decides)
    refinement_applied: Whether refinement tools were used
    reasoning_trace: List of agent's thoughts and actions
    iterations: Number of reasoning steps taken
    tools_used: Which tools agent chose to use
    metadata: Additional info (timings, model, etc.)
    """
    ticket_text: str
    draft_text: str = ""
    retrieved_chunks: list = field(default_factory=list)
    style_detected: str = "auto"
    refinement_applied: bool = False
    reasoning_trace: list = field(default_factory=list)
    iterations: int = 0
    tools_used: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for API responses."""
        return {
            "ticket_text": self.ticket_text,
            "draft_text": self.draft_text,
            "retrieved_chunks": [
                {
                    "text": getattr(chunk, 'text', str(chunk))[:500],
                    "issue_key": getattr(chunk, 'issue_key', 'unknown'),
                    "score": getattr(chunk, 'score', 0.0),
                }
                for chunk in self.retrieved_chunks
            ],
            "style_detected": self.style_detected,
            "refinement_applied": self.refinement_applied,
            "reasoning_trace": self.reasoning_trace,
            "iterations": self.iterations,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
        }


# =============================================================================
# SECTION 2: AGENTIC PIPELINE CLASS
# =============================================================================


class JiraTicketPipeline:
    """
    Orchestrates Jira ticket generation using a ReAct Agent.
    
    THIS IS NOW AN AGENT, NOT A WORKFLOW:
    -------------------------------------
    The pipeline class is retained for backward compatibility,
    but internally it delegates to a ReAct agent that:
    
    1. REASONS about what to do next
    2. CHOOSES tools dynamically
    3. OBSERVES results and adapts
    4. SELF-CORRECTS based on validation
    
    The agent decides:
    - Whether to search once or multiple times
    - What style to generate (based on retrieved examples)
    - Whether to refine (based on validation score)
    - Which refinement focus to apply
    - When the ticket is "done"
    
    USAGE:
        pipeline = JiraTicketPipeline()
        result = pipeline.generate("Create ticket for data migration")
        print(result.ticket_text)
        print(result.reasoning_trace)  # See agent's decisions!
    """
    
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
    ):
        """
        Initialize the agentic pipeline.
        
        Args:
            config: Pipeline/agent configuration
        """
        self.config = config or PipelineConfig()
        self._agent: Optional[JiraTicketAgent] = None
        
        logger.info(
            f"Initialized JiraTicketPipeline (AGENTIC): "
            f"max_iterations={self.config.max_iterations}, "
            f"verbose={self.config.verbose}"
        )
    
    @property
    def agent(self) -> JiraTicketAgent:
        """Lazy-load the agent."""
        if self._agent is None:
            agent_config = AgentConfig(
                max_iterations=self.config.max_iterations,
                max_execution_time=self.config.max_execution_time,
                verbose=False,
            )
            self._agent = create_jira_agent(config=agent_config)
        return self._agent
    
    def generate(
        self,
        user_request: str,
        force_style: Optional[str] = None,
    ) -> GenerationResult:
        """
        Generate a Jira ticket using the ReAct agent.
        
        AGENT EXECUTION:
        ----------------
        1. Agent receives user request
        2. Agent thinks: "I should search for similar tickets"
        3. Agent executes search_similar_tickets tool
        4. Agent observes: "Found 5 tickets about auth..."
        5. Agent thinks: "These are detailed tickets, I'll generate verbose"
        6. Agent executes generate_draft_ticket tool
        7. Agent observes the draft
        8. Agent thinks: "Let me validate this"
        9. Agent executes validate_ticket tool
        10. Agent observes: "Score 7/10, needs technical detail"
        11. Agent thinks: "I should refine with technical focus"
        12. Agent executes refine_ticket(focus='technical')
        13. Agent re-validates
        14. Agent observes: "Score 9/10, ready"
        15. Agent returns final ticket
        
        The path VARIES based on observations!
        
        Args:
            user_request: Natural language description of the ticket
            force_style: Hint for the agent (it may still adapt)
            
        Returns:
            GenerationResult with ticket and reasoning trace
        """
        import time
        start_time = time.time()
        
        user_request = (user_request or "").strip()
        if not user_request:
            return GenerationResult(
                ticket_text="",
                metadata={"error": "Empty request"},
            )
        
        # Add style hint to request if provided
        if force_style:
            user_request = f"{user_request}\n\n(Hint: Use {force_style} style)"
        
        logger.info(f"Agent starting for: {user_request[:100]}...")
        
        # -----------------------------------------------------------------
        # RUN THE REACT AGENT
        # -----------------------------------------------------------------
        # This is where the magic happens - the agent DECIDES what to do
        # -----------------------------------------------------------------
        
        agent_result: AgentResult = self.agent.run(user_request)
        
        # -----------------------------------------------------------------
        # Convert agent result to pipeline result format
        # -----------------------------------------------------------------
        
        elapsed = time.time() - start_time
        
        # Determine if refinement was used by checking tools
        refinement_applied = agent_result.tools_used.get("refine_ticket", 0) > 0
        
        # Detect style from agent's actions (heuristic)
        style_detected = "auto"
        for step in agent_result.reasoning_trace:
            action_input = step.get("action_input", "")
            if "brief" in action_input.lower():
                style_detected = "brief"
                break
            elif "verbose" in action_input.lower():
                style_detected = "verbose"
                break
        
        result = GenerationResult(
            ticket_text=agent_result.ticket_text,
            draft_text=agent_result.ticket_text,  # Agent refines in place
            retrieved_chunks=[],  # Would need to extract from trace
            style_detected=style_detected,
            refinement_applied=refinement_applied,
            reasoning_trace=agent_result.reasoning_trace,
            iterations=agent_result.iterations,
            tools_used=agent_result.tools_used,
            metadata={
                "elapsed_seconds": round(elapsed, 2),
                "model": settings.llm.model,
                "agent_success": agent_result.success,
                "agent_error": agent_result.error,
            },
        )
        
        logger.info(
            f"Agent complete: {agent_result.iterations} iterations, "
            f"tools: {agent_result.tools_used}, "
            f"time: {elapsed:.2f}s"
        )
        
        return result
    
    async def agenerate(
        self,
        user_request: str,
        force_style: Optional[str] = None,
    ) -> GenerationResult:
        """
        Async version of generate.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.generate(user_request, force_style)
        )


# =============================================================================
# SECTION 3: FACTORY FUNCTIONS
# =============================================================================


def create_pipeline(
    config: Optional[PipelineConfig] = None,
) -> JiraTicketPipeline:
    """
    Create a configured agentic pipeline.
    
    Args:
        config: Pipeline configuration
        
    Returns:
        Configured JiraTicketPipeline (with ReAct agent)
    """
    return JiraTicketPipeline(config=config)


# Cached singleton
_pipeline_instance: Optional[JiraTicketPipeline] = None


def get_pipeline() -> JiraTicketPipeline:
    """
    Get a cached singleton pipeline.
    
    Returns:
        Shared JiraTicketPipeline instance
    """
    global _pipeline_instance
    if _pipeline_instance is None:
        _pipeline_instance = create_pipeline()
    return _pipeline_instance


def clear_pipeline_cache():
    """Clear the cached pipeline instance."""
    global _pipeline_instance
    _pipeline_instance = None


# =============================================================================
# SECTION 4: CONVENIENCE FUNCTIONS
# =============================================================================


def generate_ticket(
    user_request: str,
    verbose: bool = False,
) -> str:
    """
    Simple function to generate a ticket using the agent.
    
    Args:
        user_request: What the ticket should be about
        verbose: Print agent reasoning
        
    Returns:
        Generated ticket text
        
    Example:
        >>> ticket = generate_ticket(
        ...     "Add caching layer for API responses"
        ... )
        >>> print(ticket)
    """
    config = PipelineConfig(verbose=verbose)
    pipeline = create_pipeline(config)
    result = pipeline.generate(user_request)
    return result.ticket_text


def generate_ticket_with_context(
    user_request: str,
    verbose: bool = False,
) -> dict:
    """
    Generate a ticket with full reasoning trace.
    
    Returns the ticket plus agent's reasoning process.
    Useful for debugging and demonstrating agent behavior.
    
    Args:
        user_request: What the ticket should be about
        verbose: Print agent reasoning
        
    Returns:
        Dict with ticket_text, reasoning_trace, iterations, tools_used
    """
    config = PipelineConfig(verbose=verbose)
    pipeline = create_pipeline(config)
    result = pipeline.generate(user_request)
    return result.to_dict()


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT CHANGED FROM WORKFLOW TO AGENT:
#
# BEFORE (Workflow):
#   1. Always retrieve
#   2. Always detect style
#   3. Always generate draft
#   4. Maybe refine (based on config)
#   5. Return result
#   Path: Fixed, determined by code
#
# AFTER (Agent):
#   1. Agent thinks: "What should I do?"
#   2. Agent chooses tool based on reasoning
#   3. Agent observes result
#   4. Agent adapts next action based on observation
#   5. Agent self-corrects if validation fails
#   6. Agent decides when done
#   Path: Dynamic, determined by agent reasoning
#
# KEY AGENT BEHAVIORS:
# - May search multiple times if first results aren't good
# - May skip refinement if validation passes immediately
# - May refine multiple times with different focuses
# - Chooses refinement focus based on validation feedback
#
# INTERVIEW TALKING POINTS:
# - "The agent REASONS about strategy, not just executes steps"
# - "Validation creates a feedback loop for self-improvement"
# - "The reasoning trace shows WHY the agent made each decision"
# - "Execution path varies per request based on content and quality"
#
# =============================================================================
