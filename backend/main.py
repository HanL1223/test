"""
================================================================================
TUTORIAL: FastAPI Backend for Jira Ticket RAG
================================================================================

ARCHITECTURE:
-------------
This backend provides a REST API for the Jira ticket generation system.
It uses FastAPI for the web framework and integrates with the LangChain
pipeline components.

ENDPOINTS:
----------
POST /api/generate     - Generate a Jira ticket
POST /api/search       - Search for similar tickets
GET  /api/health       - Health check

DESIGN PATTERNS:
----------------
1. Dependency Injection: Pipeline is injected via FastAPI depends
2. Pydantic Models: Request/response validation
3. Async Handlers: Non-blocking I/O for scalability
4. CORS Middleware: Allow frontend access

================================================================================
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings
from src.pipeline.generation import (
    JiraTicketPipeline,
    PipelineConfig,
    GenerationResult,
    get_pipeline
)
from src.retrieval import create_retriever


# =============================================================================
# SECTION 1: LOGGING CONFIGURATION
# =============================================================================


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 2: PYDANTIC MODELS (Request/Response)
# =============================================================================


class GenerateRequest(BaseModel):
    """
    Request body for ticket generation.
    
    FIELDS:
    -------
    request: Natural language description of the ticket
    fast_mode: Skip refinement for faster generation
    force_style: Override style detection ("brief" or "verbose")
    """
    request: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Natural language ticket description",
        examples=["Create a ticket for implementing user authentication"],
    )
    fast_mode: bool = Field(
        default=False,
        description="Skip multi-agent refinement for faster generation",
    )
    force_style: Optional[str] = Field(
        default=None,
        description="Force 'brief' or 'verbose' style",
    )


class GenerateResponse(BaseModel):
    """
    Response body for ticket generation.
    
    ENHANCED FOR AGENTIC GENERATION:
    --------------------------------
    Now includes reasoning_trace to show the agent's decision-making process.
    This is what makes it an AGENT, not just a workflow!
    
    FIELDS:
    -------
    ticket_text: The generated ticket content
    draft_text: Initial draft (before refinement)
    style_detected: "brief" or "verbose" (agent decides)
    refinement_applied: Whether agent used refinement tools
    retrieved_chunks: Similar tickets found by agent
    reasoning_trace: List of agent's thoughts, actions, and observations
    iterations: Number of reasoning steps the agent took
    tools_used: Which tools the agent chose to use (and how many times)
    metadata: Generation metadata (timing, model, etc.)
    """
    ticket_text: str
    draft_text: str
    style_detected: str
    refinement_applied: bool
    retrieved_chunks: list[dict]
    reasoning_trace: list[dict] = []  # NEW: Agent's decision trail
    iterations: int = 0  # NEW: How many reasoning steps
    tools_used: dict = {}  # NEW: Which tools and how often
    metadata: dict


class SearchRequest(BaseModel):
    """
    Request body for ticket search.
    
    FIELDS:
    -------
    query: Search query
    top_k: Number of results to return
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of results to return",
    )


class SearchResponse(BaseModel):
    """Response body for ticket search."""
    results: list[dict]
    query: str
    count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    model: str
    collection: str


# =============================================================================
# SECTION 3: DEPENDENCY INJECTION
# =============================================================================


def get_generation_pipeline() -> JiraTicketPipeline:
    """
    Dependency that provides the generation pipeline.
    
    DEPENDENCY INJECTION:
    FastAPI calls this function and passes the result to handlers.
    This pattern allows easy testing (swap with mock) and configuration.
    """
    return get_pipeline()


# =============================================================================
# SECTION 4: APPLICATION LIFECYCLE
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle manager."""
    logger.info("Starting Jira Ticket RAG API...")
    
    # Validate API key
    if not settings.google_api_key:
        logger.warning("GOOGLE_API_KEY not set!")
    
    logger.info(f"API ready - Model: {settings.llm.model}")
    
    yield  # Application runs here
    
    logger.info("Shutting down Jira Ticket RAG API...")


# =============================================================================
# SECTION 5: FASTAPI APPLICATION
# =============================================================================


app = FastAPI(
    title="Jira Ticket RAG API",
    description="Generate Jira tickets using RAG with LangChain",
    version="1.0.0",
    lifespan=lifespan,
)


# CORS Middleware - Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure properly in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# SECTION 6: API ENDPOINTS
# =============================================================================


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint.
    
    Returns the current status and configuration.
    """
    return HealthResponse(
        status="healthy",
        model=settings.llm.model,
        collection=settings.chroma.collection_name,
    )


@app.post("/api/generate", response_model=GenerateResponse)
async def generate_ticket(
    body: GenerateRequest,
    pipeline: JiraTicketPipeline = Depends(get_generation_pipeline),
):
    """
    Generate a Jira ticket using the ReAct Agent.
    
    THIS IS AN AGENT, NOT A WORKFLOW:
    ---------------------------------
    The agent REASONS about what to do, CHOOSES tools, OBSERVES results,
    and ADAPTS its strategy. The reasoning_trace shows this process!
    
    AGENT PROCESS:
    1. Agent thinks: "I should search for similar tickets"
    2. Agent calls search tool → observes results
    3. Agent thinks: "I'll generate a draft with this context"
    4. Agent calls generate tool → observes draft
    5. Agent thinks: "Let me validate quality"
    6. Agent calls validate tool → observes score
    7. If score < 8: Agent refines and re-validates
    8. Agent returns final ticket when satisfied
    
    PARAMETERS:
    -----------
    body.request: Natural language description
    body.fast_mode: Reduce agent iterations for speed
    body.force_style: Hint for the agent (it may adapt)
    
    RETURNS:
    --------
    Generated ticket with reasoning_trace showing agent's decisions
    """
    logger.info(f"Generate request: {body.request[:100]}...")
    
    try:
        # Configure pipeline/agent for this request
        if body.fast_mode:
            # Fewer iterations for faster response
            fast_config = PipelineConfig(
                max_iterations=5,  # Reduced from default 10
                verbose=False,
            )
            pipeline = JiraTicketPipeline(config=fast_config)
        
        # Run the agent
        result: GenerationResult = await pipeline.agenerate(
            user_request=body.request,
            force_style=body.force_style,
        )
        
        logger.info(
            f"Agent complete: iterations={result.iterations}, "
            f"tools={result.tools_used}, "
            f"time={result.metadata.get('elapsed_seconds', 0):.2f}s"
        )
        
        return GenerateResponse(
            ticket_text=result.ticket_text,
            draft_text=result.draft_text,
            style_detected=result.style_detected,
            refinement_applied=result.refinement_applied,
            retrieved_chunks=[
                {
                    "text": getattr(chunk, 'text', str(chunk))[:500],
                    "issue_key": getattr(chunk, 'issue_key', 'unknown'),
                    "score": getattr(chunk, 'score', 0.0),
                }
                for chunk in result.retrieved_chunks
            ],
            reasoning_trace=result.reasoning_trace,  # NEW: Agent's decisions
            iterations=result.iterations,  # NEW: Reasoning steps
            tools_used=result.tools_used,  # NEW: Tool usage
            metadata=result.metadata,
        )
        
    except Exception as e:
        logger.exception(f"Generation failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}",
        )


@app.post("/api/search", response_model=SearchResponse)
async def search_tickets(body: SearchRequest):
    """
    Search for similar tickets.
    
    Useful for:
    - Finding existing tickets before creating new ones
    - Understanding ticket patterns
    - Debugging retrieval quality
    
    PARAMETERS:
    -----------
    body.query: Search query
    body.top_k: Number of results (1-20)
    
    RETURNS:
    --------
    List of similar tickets with scores
    """
    logger.info(f"Search request: {body.query[:100]}...")
    
    try:
        retriever = create_retriever(top_k=body.top_k)
        chunks = retriever.retrieve(body.query)
        
        return SearchResponse(
            results=[
                {
                    "text": chunk.text,
                    "issue_key": chunk.issue_key,
                    "score": chunk.score,
                    "metadata": chunk.metadata,
                }
                for chunk in chunks
            ],
            query=body.query,
            count=len(chunks),
        )
        
    except Exception as e:
        logger.exception(f"Search failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}",
        )


# =============================================================================
# SECTION 7: DEVELOPMENT SERVER
# =============================================================================


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting development server on {host}:{port}")
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=True,  # Auto-reload on code changes
        log_level="info",
    )


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT THIS BACKEND PROVIDES:
# 1. REST API for ticket generation
# 2. Search endpoint for retrieval testing
# 3. Health check for monitoring
# 4. CORS support for frontend integration
#
# FASTAPI FEATURES USED:
# - Pydantic models for validation
# - Dependency injection for pipeline
# - Async handlers for scalability
# - Lifespan management for startup/shutdown
#
# INTERVIEW TALKING POINTS:
# - "FastAPI provides automatic OpenAPI docs at /docs"
# - "Dependency injection makes the handlers testable"
# - "Async handlers allow concurrent request processing"
#
# =============================================================================
