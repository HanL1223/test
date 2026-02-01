"""
================================================================================
TUTORIAL: Agent Tools for Jira Ticket Generation
================================================================================

WHAT ARE TOOLS?
---------------
In LangChain agents, tools are functions the agent can decide to call.
Unlike a workflow where steps are fixed, an agent REASONS about which
tool to use and when.

REACT PATTERN:
--------------
The agent follows a Thought → Action → Observation loop:

1. THOUGHT: "I need to find similar tickets first"
2. ACTION: Call search_similar_tickets tool
3. OBSERVATION: "Found 5 similar tickets about data migration"
4. THOUGHT: "Now I have context, I should generate a draft"
5. ACTION: Call generate_draft tool
6. OBSERVATION: "Draft created: h2. Overview..."
7. THOUGHT: "Let me validate if this is good enough"
8. ACTION: Call validate_ticket tool
9. OBSERVATION: "Score: 7/10 - needs more technical detail"
10. THOUGHT: "I should refine with technical focus"
11. ACTION: Call refine_ticket tool with focus="technical"
12. OBSERVATION: "Refined ticket with technical details"
13. THOUGHT: "This looks complete now"
14. FINAL ANSWER: Return the ticket

TOOL DESIGN PRINCIPLES:
-----------------------
1. Clear descriptions help the agent decide when to use each tool
2. Tools should be atomic - do one thing well
3. Return structured output the agent can reason about
4. Include enough context in observations for next decisions

================================================================================
"""

import json
import logging
from typing import Optional, Type

from langchain_core.tools import BaseTool, tool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from src.config import settings
from src.retrieval import create_retriever, JiraIssueRetriever
from src.llm import get_llm, create_draft_prompt, create_brief_draft_prompt
from src.llm.prompts import create_refinement_prompt, detect_ticket_style

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: TOOL INPUT SCHEMAS
# =============================================================================
# Pydantic models define what arguments each tool accepts.
# This helps the agent understand how to call tools correctly.


class SearchInput(BaseModel):
    """Input schema for search tool."""
    query: str = Field(
        description="The search query to find similar historical Jira tickets. "
                    "Should describe the type of ticket you want to create."
    )
    top_k: int = Field(
        default=5,
        description="Number of similar tickets to retrieve (1-10)"
    )


class GenerateInput(BaseModel):
    """Input schema for generate tool."""
    user_request: str = Field(
        description="The original user request describing what ticket to create"
    )
    context: str = Field(
        description="Retrieved similar tickets to use as context/examples. "
                    "Should come from the search_similar_tickets tool."
    )
    style: str = Field(
        default="auto",
        description="Ticket style: 'brief', 'verbose', or 'auto' to detect from context"
    )


class RefineInput(BaseModel):
    """Input schema for refine tool."""
    draft: str = Field(
        description="The current draft ticket text to refine"
    )
    user_request: str = Field(
        description="The original user request for context"
    )
    focus: str = Field(
        default="all",
        description="Refinement focus: 'product' (business clarity), "
                    "'technical' (implementation details), 'qa' (testability), "
                    "or 'all' for comprehensive review"
    )


class ValidateInput(BaseModel):
    """Input schema for validate tool."""
    ticket: str = Field(
        description="The ticket text to validate"
    )
    user_request: str = Field(
        description="The original user request to check alignment"
    )


class CreateJiraInput(BaseModel):
    """Input schema for create Jira ticket tool."""
    ticket_text: str = Field(
        description="The complete ticket text to create in Jira. "
                    "Should be the validated, final ticket content."
    )
    summary: str = Field(
        description="The ticket summary/title. Should be concise (under 100 chars)."
    )
    issue_type: str = Field(
        default="Task",
        description="Jira issue type: Task, Story, Bug, Epic, etc."
    )
    labels: list[str] = Field(
        default=[],
        description="Labels to apply to the ticket (optional)"
    )


# =============================================================================
# SECTION 2: SEARCH TOOL
# =============================================================================


class SearchSimilarTicketsTool(BaseTool):
    """
    Tool to search for similar historical Jira tickets.
    
    WHY A TOOL CLASS?
    Using a class instead of @tool decorator allows:
    - Stateful components (retriever instance)
    - Custom initialization
    - Better type hints
    
    AGENT USAGE:
    The agent calls this FIRST to gather context before generating.
    It may call multiple times with different queries to find
    the best matching tickets.
    """
    
    name: str = "search_similar_tickets"
    description: str = (
        "Search for similar historical Jira tickets in the database. "
        "Use this FIRST to find examples before generating a new ticket. "
        "Returns formatted context with issue keys and content. "
        "The search query should describe the type of ticket you want to create."
    )
    args_schema: Type[BaseModel] = SearchInput
    
    # Instance attributes
    _retriever: Optional[JiraIssueRetriever] = None
    
    def __init__(self, retriever: Optional[JiraIssueRetriever] = None, **kwargs):
        super().__init__(**kwargs)
        self._retriever = retriever
    
    @property
    def retriever(self) -> JiraIssueRetriever:
        """Lazy-load retriever."""
        if self._retriever is None:
            self._retriever = create_retriever()
        return self._retriever
    
    def _run(
        self,
        query: str,
        top_k: int = 5,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Execute the search.
        
        Returns:
            Formatted string with search results for the agent to observe
        """
        logger.info(f"[TOOL] Searching for: {query[:100]}...")
        
        try:
            # Clamp top_k
            top_k = max(1, min(10, top_k))
            
            # Create retriever with specified top_k
            retriever = create_retriever(top_k=top_k)
            chunks = retriever.retrieve(query)
            
            if not chunks:
                return (
                    "No similar tickets found. You may need to:\n"
                    "1. Try a different search query\n"
                    "2. Generate a ticket without examples (less accurate)"
                )
            
            # Format results for agent observation
            results = []
            for i, chunk in enumerate(chunks, 1):
                results.append(
                    f"[{i}] {chunk.issue_key} (similarity: {chunk.similarity_score or 0:.2f})\n"
                    f"{chunk.text[:800]}{'...' if len(chunk.text) > 800 else ''}"
                )
            
            context = "\n\n---\n\n".join(results)
            
            return (
                f"Found {len(chunks)} similar tickets:\n\n"
                f"{context}\n\n"
                f"Use this context when generating the draft ticket."
            )
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return f"Search failed: {str(e)}. Try with a different query."


# =============================================================================
# SECTION 3: GENERATE TOOL
# =============================================================================


class GenerateDraftTool(BaseTool):
    """
    Tool to generate an initial draft ticket.
    
    AGENT USAGE:
    Call this AFTER searching for similar tickets.
    The context parameter should contain the search results.
    The agent decides what style to use based on the context.
    """
    
    name: str = "generate_draft_ticket"
    description: str = (
        "Generate an initial draft Jira ticket using RAG context. "
        "IMPORTANT: Call search_similar_tickets FIRST to get context. "
        "Pass the search results as the 'context' parameter. "
        "The style can be 'brief' (short tickets), 'verbose' (detailed), "
        "or 'auto' to detect from the context."
    )
    args_schema: Type[BaseModel] = GenerateInput
    
    def _run(
        self,
        user_request: str,
        context: str,
        style: str = "auto",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Generate a draft ticket.
        
        Returns:
            The generated draft with metadata for agent observation
        """
        logger.info(f"[TOOL] Generating draft, style={style}")
        
        try:
            llm = get_llm()
            
            # Auto-detect style if needed
            if style == "auto":
                # Simple heuristic based on context length
                avg_length = len(context) / max(1, context.count("---"))
                style = "brief" if avg_length < 600 else "verbose"
                logger.debug(f"Auto-detected style: {style}")
            
            # Select prompt based on style
            if style == "brief":
                prompt = create_brief_draft_prompt()
            else:
                prompt = create_draft_prompt()
            
            # Generate
            chain = prompt | llm
            response = chain.invoke({
                "user_request": user_request,
                "rag_context": context,
            })
            
            draft = response.content
            
            return (
                f"Draft generated ({style} style, {len(draft)} chars):\n\n"
                f"---DRAFT START---\n"
                f"{draft}\n"
                f"---DRAFT END---\n\n"
                f"You should now validate this draft or refine it if needed."
            )
            
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            return f"Generation failed: {str(e)}. Check the context and try again."


# =============================================================================
# SECTION 4: REFINE TOOL
# =============================================================================


class RefineTicketTool(BaseTool):
    """
    Tool to refine/improve a draft ticket.
    
    AGENT USAGE:
    Call this when validation indicates the ticket needs improvement.
    Choose a focus area based on what's missing:
    - 'product': Business context unclear
    - 'technical': Missing implementation details
    - 'qa': Acceptance criteria not testable
    - 'all': General improvement
    """
    
    name: str = "refine_ticket"
    description: str = (
        "Refine and improve a draft Jira ticket. "
        "Use after generate_draft_ticket if the ticket needs improvement. "
        "Choose focus: 'product' (business clarity), 'technical' (implementation), "
        "'qa' (testability), or 'all' (comprehensive review). "
        "The agent should decide the focus based on validation feedback."
    )
    args_schema: Type[BaseModel] = RefineInput
    
    def _run(
        self,
        draft: str,
        user_request: str,
        focus: str = "all",
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Refine the ticket.
        
        Returns:
            Refined ticket with change summary
        """
        logger.info(f"[TOOL] Refining ticket, focus={focus}")
        
        try:
            llm = get_llm()
            
            # Determine which refinement agents to run
            if focus == "all":
                focuses = ["product", "technical", "qa"]
            else:
                focuses = [focus]
            
            current_draft = draft
            changes_made = []
            
            for agent_focus in focuses:
                prompt = create_refinement_prompt(agent_focus)
                chain = prompt | llm
                
                response = chain.invoke({
                    "user_request": user_request,
                    "draft": current_draft,
                })
                
                new_draft = response.content
                
                # Track if changes were made
                if new_draft.strip() != current_draft.strip():
                    changes_made.append(agent_focus)
                    current_draft = new_draft
            
            change_summary = (
                f"Applied {', '.join(changes_made)} refinements" 
                if changes_made else "No changes needed"
            )
            
            return (
                f"Refinement complete ({change_summary}):\n\n"
                f"---REFINED TICKET---\n"
                f"{current_draft}\n"
                f"---END TICKET---\n\n"
                f"Validate the refined ticket to check if it meets quality standards."
            )
            
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return f"Refinement failed: {str(e)}. The original draft is still valid."


# =============================================================================
# SECTION 5: VALIDATE TOOL
# =============================================================================


class ValidateTicketTool(BaseTool):
    """
    Tool to validate ticket quality.
    
    AGENT USAGE:
    Call this to check if a ticket is ready for output.
    Based on the validation score and feedback, the agent decides:
    - Score >= 8: Ticket is good, return as final answer
    - Score < 8: Need refinement, use feedback to choose focus
    
    This creates the self-correcting loop that makes it a true agent.
    """
    
    name: str = "validate_ticket"
    description: str = (
        "Validate a Jira ticket against quality criteria. "
        "Returns a score (1-10) and specific feedback. "
        "Use this to decide if the ticket is ready or needs more refinement. "
        "Score >= 8 means ready to return. Score < 8 means needs improvement."
    )
    args_schema: Type[BaseModel] = ValidateInput
    
    def _run(
        self,
        ticket: str,
        user_request: str,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Validate the ticket.
        
        Returns:
            Validation score and detailed feedback
        """
        logger.info("[TOOL] Validating ticket...")
        
        try:
            llm = get_llm()
            
            validation_prompt = f"""Evaluate this Jira ticket against quality criteria.

ORIGINAL REQUEST:
{user_request}

TICKET TO VALIDATE:
{ticket}

EVALUATION CRITERIA:
1. ALIGNMENT (0-2): Does it address the user's request?
2. CLARITY (0-2): Is it clear what needs to be done?
3. STRUCTURE (0-2): Does it follow good Jira ticket format?
4. COMPLETENESS (0-2): Does it have necessary details without over-elaboration?
5. ACTIONABLE (0-2): Can someone start working on this immediately?

Respond in this EXACT JSON format:
{{
    "scores": {{
        "alignment": <0-2>,
        "clarity": <0-2>,
        "structure": <0-2>,
        "completeness": <0-2>,
        "actionable": <0-2>
    }},
    "total_score": <sum of above, 0-10>,
    "strengths": ["strength1", "strength2"],
    "improvements_needed": ["improvement1", "improvement2"],
    "recommended_focus": "<product|technical|qa|none>",
    "ready_for_output": <true if total_score >= 8, false otherwise>
}}
"""
            
            response = llm.invoke(validation_prompt)
            
            # Try to parse JSON response
            try:
                # Extract JSON from response
                content = response.content
                # Find JSON in response
                start = content.find("{")
                end = content.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = content[start:end]
                    result = json.loads(json_str)
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback: assume it's good enough if we can't parse
                logger.warning(f"Could not parse validation JSON: {e}")
                return (
                    "Validation complete:\n"
                    "Score: 8/10 (estimated)\n"
                    "The ticket appears to be ready. "
                    "If you're satisfied, return it as the final answer."
                )
            
            # Format validation result for agent
            score = result.get("total_score", 7)
            ready = result.get("ready_for_output", score >= 8)
            focus = result.get("recommended_focus", "none")
            strengths = result.get("strengths", [])
            improvements = result.get("improvements_needed", [])
            
            feedback_parts = [
                f"Validation Score: {score}/10",
                f"Ready for output: {'YES' if ready else 'NO'}",
            ]
            
            if strengths:
                feedback_parts.append(f"Strengths: {', '.join(strengths)}")
            
            if improvements:
                feedback_parts.append(f"Needs improvement: {', '.join(improvements)}")
            
            if not ready and focus != "none":
                feedback_parts.append(
                    f"Recommendation: Use refine_ticket with focus='{focus}'"
                )
            elif ready:
                feedback_parts.append(
                    "Recommendation: Return this ticket as the final answer"
                )
            
            return "\n".join(feedback_parts)
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return (
                "Validation encountered an error, but the ticket may still be usable.\n"
                f"Error: {str(e)}\n"
                "Consider returning the ticket or attempting one refinement."
            )


# =============================================================================
# SECTION 6: CREATE JIRA TICKET TOOL
# =============================================================================


class CreateJiraTicketTool(BaseTool):
    """
    Tool to create a ticket directly in Jira.
    
    AGENT USAGE:
    Call this AFTER the ticket is validated and ready.
    This completes the automation loop:
    
    User Request → Search → Generate → Validate → CREATE IN JIRA
    
    The agent should only use this when:
    1. Validation score >= 8
    2. User requested actual Jira creation (not just generation)
    
    REQUIREMENTS:
    Jira configuration must be set in environment:
    - JIRA_BASE_URL
    - JIRA_EMAIL
    - JIRA_API_TOKEN
    - JIRA_PROJECT_KEY
    """
    
    name: str = "create_jira_ticket"
    description: str = (
        "Create a ticket directly in Jira. "
        "Use this ONLY after the ticket has been validated (score >= 8). "
        "Requires: ticket_text (full content), summary (title), issue_type (Task/Story/Bug). "
        "Returns the Jira issue URL if successful. "
        "NOTE: Only use if user explicitly wants to create in Jira, not just generate."
    )
    args_schema: Type[BaseModel] = CreateJiraInput
    
    def _run(
        self,
        ticket_text: str,
        summary: str,
        issue_type: str = "Task",
        labels: list[str] = None,
        run_manager: Optional[CallbackManagerForToolRun] = None,
    ) -> str:
        """
        Create the ticket in Jira.
        
        Returns:
            Success message with Jira URL, or error message
        """
        logger.info(f"[TOOL] Creating Jira ticket: {summary[:50]}...")
        
        try:
            # Import here to avoid circular imports and allow optional Jira
            from src.jira import create_jira_client, JiraCreateResult
            from src.config import settings
            
            # Check if Jira is configured
            if not settings.jira.is_configured:
                return (
                    "Jira integration is not configured. "
                    "Set JIRA_BASE_URL, JIRA_EMAIL, JIRA_API_TOKEN, and JIRA_PROJECT_KEY. "
                    "Returning the ticket text as final answer instead."
                    f"\n\n---TICKET---\n{ticket_text}\n---END TICKET---"
                )
            
            # Create Jira client
            client = create_jira_client()
            
            # Create the issue
            result: JiraCreateResult = client.create_issue(
                summary=summary,
                description=ticket_text,
                issue_type=issue_type,
                labels=labels or [],
                use_jira_markup=True,  # Assume ticket has Jira markup
            )
            
            if result.success:
                return (
                    f"✅ Successfully created Jira ticket!\n\n"
                    f"Issue Key: {result.key}\n"
                    f"URL: {result.url}\n\n"
                    f"The ticket has been created in your Jira project. "
                    f"Return this information as the final answer."
                )
            else:
                return (
                    f"❌ Failed to create Jira ticket: {result.error}\n\n"
                    f"Returning the ticket text instead:\n\n"
                    f"---TICKET---\n{ticket_text}\n---END TICKET---"
                )
            
        except ImportError:
            return (
                "Jira client not available. "
                "Returning the ticket text as final answer.\n\n"
                f"---TICKET---\n{ticket_text}\n---END TICKET---"
            )
        except Exception as e:
            logger.error(f"Jira creation failed: {e}")
            return (
                f"Error creating Jira ticket: {str(e)}\n\n"
                f"Returning the ticket text instead:\n\n"
                f"---TICKET---\n{ticket_text}\n---END TICKET---"
            )


# =============================================================================
# SECTION 7: TOOL FACTORY
# =============================================================================


def create_jira_tools(
    retriever: Optional[JiraIssueRetriever] = None,
    include_jira_create: bool = True,
) -> list[BaseTool]:
    """
    Create all tools for the Jira ticket agent.
    
    FACTORY PATTERN:
    Centralizes tool creation and allows customization.
    
    Args:
        retriever: Optional pre-configured retriever
        include_jira_create: Whether to include the create_jira_ticket tool
                            Set to False if Jira integration is not configured
                            or if you only want ticket generation without creation
        
    Returns:
        List of tools for the agent
    """
    tools = [
        SearchSimilarTicketsTool(retriever=retriever),
        GenerateDraftTool(),
        RefineTicketTool(),
        ValidateTicketTool(),
    ]
    
    # Optionally add Jira creation tool
    if include_jira_create:
        tools.append(CreateJiraTicketTool())
    
    return tools


# =============================================================================
# SECTION 7: SIMPLE FUNCTION TOOLS (Alternative)
# =============================================================================
# These are simpler versions using the @tool decorator.
# Use these if you prefer functional style over classes.


@tool
def search_tickets(query: str, top_k: int = 5) -> str:
    """
    Search for similar historical Jira tickets.
    
    Args:
        query: Search query describing the ticket type
        top_k: Number of results (1-10)
    
    Returns:
        Formatted search results
    """
    tool_instance = SearchSimilarTicketsTool()
    return tool_instance._run(query, top_k)


@tool
def generate_ticket(user_request: str, context: str, style: str = "auto") -> str:
    """
    Generate a draft Jira ticket using context from search.
    
    Args:
        user_request: What ticket to create
        context: Similar tickets from search
        style: 'brief', 'verbose', or 'auto'
    
    Returns:
        Generated draft ticket
    """
    tool_instance = GenerateDraftTool()
    return tool_instance._run(user_request, context, style)


@tool  
def refine_ticket_content(draft: str, user_request: str, focus: str = "all") -> str:
    """
    Refine a draft ticket to improve quality.
    
    Args:
        draft: Current ticket text
        user_request: Original request
        focus: 'product', 'technical', 'qa', or 'all'
    
    Returns:
        Refined ticket
    """
    tool_instance = RefineTicketTool()
    return tool_instance._run(draft, user_request, focus)


@tool
def validate_ticket_quality(ticket: str, user_request: str) -> str:
    """
    Validate ticket quality and get improvement recommendations.
    
    Args:
        ticket: Ticket to validate
        user_request: Original request
    
    Returns:
        Score and feedback
    """
    tool_instance = ValidateTicketTool()
    return tool_instance._run(ticket, user_request)


@tool
def create_jira(ticket_text: str, summary: str, issue_type: str = "Task") -> str:
    """
    Create a ticket directly in Jira.
    
    Args:
        ticket_text: Full ticket content
        summary: Ticket title
        issue_type: Task, Story, Bug, etc.
    
    Returns:
        Success message with Jira URL or error
    """
    tool_instance = CreateJiraTicketTool()
    return tool_instance._run(ticket_text, summary, issue_type)


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Tools are functions the agent DECIDES to call (not hardcoded steps)
# 2. Clear descriptions help the agent reason about when to use each tool
# 3. Structured inputs/outputs enable the observe→think→act loop
# 4. The validate tool creates the self-correction capability
# 5. The create_jira_ticket tool completes the automation loop
#
# KEY AGENT BEHAVIORS ENABLED:
# - Agent searches multiple times if first results aren't good
# - Agent refines iteratively based on validation feedback
# - Agent chooses refinement focus based on what's missing
# - Agent decides when the ticket is "good enough"
# - Agent can CREATE the ticket in Jira when validated
#
# FULL AGENT LOOP:
#   User Request → Search → Generate → Validate → [Refine?] → Create in Jira
#
# INTERVIEW TALKING POINTS:
# - "Tools give the agent capabilities; the agent decides strategy"
# - "Validation creates a feedback loop for self-correction"
# - "Unlike workflows, the agent adapts based on intermediate results"
# - "Jira integration completes the automation - from idea to ticket in one step"
#
# =============================================================================
