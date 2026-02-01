"""
================================================================================
Agents Package - ReAct Agent for Jira Ticket Generation
================================================================================

This package provides the agentic (not workflow) approach to ticket generation.

KEY DIFFERENCE FROM PIPELINES:
------------------------------
Pipeline (src/pipeline/): Fixed sequence of steps
Agent (src/agents/): LLM reasons and decides which tools to use

COMPONENTS:
-----------
- tools.py: Individual capabilities the agent can use
  - search_similar_tickets: Find relevant examples
  - generate_draft_ticket: Create initial draft
  - refine_ticket: Improve based on feedback
  - validate_ticket: Check quality
  - create_jira_ticket: Create ticket in Jira (NEW!)

- jira_agent.py: The ReAct agent that orchestrates tools
  - Thinks about what to do
  - Chooses and executes tools
  - Observes results
  - Self-corrects until done

FULL AUTOMATION LOOP:
---------------------
User Request → Search → Generate → Validate → [Refine?] → Create in Jira

USAGE:
------
    from src.agents import JiraTicketAgent, generate_ticket_with_agent
    
    # Simple usage
    ticket = generate_ticket_with_agent("Create auth ticket")
    
    # With reasoning trace
    agent = JiraTicketAgent()
    result = agent.run("Create auth ticket")
    print(result.ticket_text)
    print(result.reasoning_trace)  # See agent's decisions

================================================================================
"""

# Lazy imports to avoid circular dependencies
def __getattr__(name):
    if name in [
        "JiraTicketAgent", "AgentConfig", "AgentResult",
        "create_jira_agent", "get_jira_agent", "clear_agent_cache",
        "generate_ticket_with_agent", "generate_with_reasoning",
    ]:
        from src.agents.jira_agent import (
            JiraTicketAgent, AgentConfig, AgentResult,
            create_jira_agent, get_jira_agent, clear_agent_cache,
            generate_ticket_with_agent, generate_with_reasoning,
        )
        return locals()[name]
    
    if name in [
        "SearchSimilarTicketsTool", "GenerateDraftTool", "RefineTicketTool",
        "ValidateTicketTool", "CreateJiraTicketTool", "create_jira_tools",
    ]:
        from src.agents.tools import (
            SearchSimilarTicketsTool, GenerateDraftTool, RefineTicketTool,
            ValidateTicketTool, CreateJiraTicketTool, create_jira_tools,
        )
        return locals()[name]
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "JiraTicketAgent", "AgentConfig", "AgentResult",
    "create_jira_agent", "get_jira_agent", "clear_agent_cache",
    "generate_ticket_with_agent", "generate_with_reasoning",
    "SearchSimilarTicketsTool", "GenerateDraftTool", "RefineTicketTool",
    "ValidateTicketTool", "CreateJiraTicketTool", "create_jira_tools",
]