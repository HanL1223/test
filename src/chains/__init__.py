"""
================================================================================
Chains Package - LangChain Expression Language (LCEL) Chains
================================================================================

This package contains LCEL chains for ticket generation workflow.

COMPONENTS:
- generation: Draft ticket generation chain
- refinement: Multi-agent refinement chain

LCEL OVERVIEW:
--------------
LangChain Expression Language (LCEL) is a declarative way to compose chains.
It uses the pipe operator (|) to connect components:

    chain = prompt | llm | output_parser

Advantages of LCEL:
1. Declarative composition - easy to read and modify
2. Streaming support built-in
3. Async support built-in
4. Type hints and validation
5. Easy to debug and trace

================================================================================
"""

from src.chains.generation import (
    TicketGenerationChain,
    create_generation_chain,
    generate_ticket_draft,
    agenerate_ticket_draft,
)

from src.chains.refinement import (
    RefinementChain,
    create_refinement_chain,
    refine_ticket,
    arefine_ticket,
    RefinerRole,
)

__all__ = [
    # Generation
    "TicketGenerationChain",
    "create_generation_chain",
    "generate_ticket_draft",
    "agenerate_ticket_draft",
    # Refinement
    "RefinementChain",
    "create_refinement_chain",
    "refine_ticket",
    "arefine_ticket",
    "RefinerRole",
]
