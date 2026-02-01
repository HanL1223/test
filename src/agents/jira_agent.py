"""
================================================================================
TUTORIAL: ReAct Agent for Jira Ticket Generation
================================================================================

WHAT IS A REACT AGENT?
----------------------
ReAct (Reasoning + Acting) is an agent paradigm where the LLM:
1. REASONS about the current state and what to do next
2. ACTS by calling a tool
3. OBSERVES the result
4. Repeats until the task is complete

This is fundamentally different from a workflow:
- Workflow: Fixed sequence of steps (A → B → C → D)
- Agent: Dynamic decisions (A → B → A → C → B → D based on observations)

THE REACT LOOP:
---------------
```
while not done:
    thought = llm.think("Given observations, what should I do?")
    action = llm.choose_tool(thought)
    observation = execute_tool(action)
    if llm.thinks_done(observation):
        done = True
        return final_answer
```

AGENT VS WORKFLOW COMPARISON:
-----------------------------
WORKFLOW (what we had before):
    1. Always search
    2. Always detect style
    3. Always generate
    4. Always refine (all three agents)
    5. Return result

AGENT (what we're building):
    1. Think: "I should search for similar tickets"
    2. Search → Observe results
    3. Think: "These are brief tickets, I'll generate brief style"
    4. Generate → Observe draft
    5. Think: "Let me validate this"
    6. Validate → Observe: "Score 6/10, needs technical detail"
    7. Think: "I should refine with technical focus"
    8. Refine(technical) → Observe improved draft
    9. Validate → Observe: "Score 9/10, ready"
    10. Return final answer

The agent ADAPTS based on what it observes!

================================================================================
"""

"""
ReAct Agent for Jira Ticket Generation (LangChain 1.x Compatible)
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

from langchain_core.messages import HumanMessage
from langchain_core.tools import BaseTool

# LangChain 1.x uses langgraph for ReAct agents
from langgraph.prebuilt import create_react_agent as create_langgraph_agent

from src.config import settings
from src.llm import get_llm

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    max_iterations: int = 10
    max_execution_time: Optional[float] = 120.0
    verbose: bool = False


@dataclass
class AgentResult:
    ticket_text: str
    reasoning_trace: list = field(default_factory=list)
    iterations: int = 0
    tools_used: dict = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    jira_key: Optional[str] = None
    jira_url: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "ticket_text": self.ticket_text,
            "reasoning_trace": self.reasoning_trace,
            "iterations": self.iterations,
            "tools_used": self.tools_used,
            "success": self.success,
            "error": self.error,
            "jira_key": self.jira_key,
            "jira_url": self.jira_url,
        }


class JiraTicketAgent:
    def __init__(self, config: Optional[AgentConfig] = None, tools: Optional[list] = None):
        self.config = config or AgentConfig()
        self._tools = tools
        self._agent = None
        logger.info(f"Initialized JiraTicketAgent")
    
    @property
    def tools(self):
        if self._tools is None:
            from src.agents.tools import create_jira_tools
            self._tools = create_jira_tools()
        return self._tools
    
    @property
    def agent(self):
        if self._agent is None:
            llm = get_llm()
            self._agent = create_langgraph_agent(model=llm, tools=self.tools)
        return self._agent
    
    def run(self, user_request: str) -> AgentResult:
        logger.info(f"Agent starting: {user_request[:100]}...")
        
        if not user_request or not user_request.strip():
            return AgentResult(ticket_text="", success=False, error="Empty request")
        
        try:
            result = self.agent.invoke({"messages": [HumanMessage(content=user_request)]})
            
            messages = result.get("messages", [])
            output = ""
            for msg in reversed(messages):
                if hasattr(msg, 'content') and msg.content:
                    content = msg.content
                    # Handle if content is a list of message parts
                    if isinstance(content, list):
                        text_parts = []
                        for part in content:
                            if isinstance(part, dict) and part.get('type') == 'text':
                                text_parts.append(part.get('text', ''))
                            elif isinstance(part, str):
                                text_parts.append(part)
                        output = '\n'.join(text_parts)
                    else:
                        output = str(content)
                    break
            
            reasoning_trace = []
            tools_used = {}
            
            for msg in messages:
                msg_type = type(msg).__name__
                if msg_type == "ToolMessage":
                    tool_name = getattr(msg, 'name', 'unknown')
                    tools_used[tool_name] = tools_used.get(tool_name, 0) + 1
                    reasoning_trace.append({
                        "thought": "",
                        "action": tool_name,
                        "observation": str(msg.content)[:500],
                    })
            
            logger.info(f"Agent complete: tools={tools_used}")
            
            return AgentResult(
                ticket_text=output,
                reasoning_trace=reasoning_trace,
                iterations=len(reasoning_trace),
                tools_used=tools_used,
                success=True,
            )
            
        except Exception as e:
            logger.exception(f"Agent failed: {e}")
            return AgentResult(ticket_text="", success=False, error=str(e))


def create_jira_agent(config: Optional[AgentConfig] = None) -> JiraTicketAgent:
    return JiraTicketAgent(config=config)


_agent_instance: Optional[JiraTicketAgent] = None


def get_jira_agent() -> JiraTicketAgent:
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = create_jira_agent()
    return _agent_instance


def clear_agent_cache():
    global _agent_instance
    _agent_instance = None


def generate_ticket_with_agent(user_request: str, verbose: bool = False) -> str:
    config = AgentConfig(verbose=verbose)
    agent = create_jira_agent(config)
    result = agent.run(user_request)
    if not result.success:
        raise RuntimeError(f"Agent failed: {result.error}")
    return result.ticket_text


def generate_with_reasoning(user_request: str) -> dict:
    agent = get_jira_agent()
    result = agent.run(user_request)
    return result.to_dict()