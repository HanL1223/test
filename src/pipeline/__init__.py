"""
================================================================================
Pipeline Package - End-to-End Workflows
================================================================================

This package provides complete pipelines for:
- Indexing: Load, process, embed, and store documents
- Generation: Retrieve context, generate draft, refine ticket

PIPELINE ARCHITECTURE:
----------------------
Pipeline A (Indexing):
    Raw Data → Load → Clean → Chunk → Embed → Store

Pipeline B (Generation):
    User Request → Retrieve → Generate → Refine → Final Ticket

================================================================================
"""

"""
Pipeline Package - Indexing and Generation Pipelines
"""

# Only import indexing by default - generation has agent dependencies
from src.pipeline.indexing import (
    IndexingPipeline,
    IndexingConfig,
    IndexingResult,
    run_indexing_pipeline,
)

__all__ = [
    "IndexingPipeline",
    "IndexingConfig", 
    "IndexingResult",
    "run_indexing_pipeline",
]

# Generation imports available on demand:
# from src.pipeline.generation import JiraTicketPipeline, generate_ticket