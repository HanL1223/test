#!/usr/bin/env python3
"""
================================================================================
TUTORIAL: Document Indexing Script
================================================================================

PURPOSE:
--------
This script indexes prepared Jira issues into ChromaDB for RAG retrieval.
It uses the LangChain-based indexing pipeline.

WORKFLOW:
---------
1. Load prepared JSONL file (from prepare_dataset.py)
2. Split documents into semantic chunks
3. Generate embeddings using Gemini
4. Store in ChromaDB with metadata

USAGE:
------
    # Basic usage
    python scripts/index_documents.py --input data/processed/jira_issues.jsonl
    
    # With custom settings
    python scripts/index_documents.py \\
        --input data/processed/jira_issues.jsonl \\
        --collection jira_tickets \\
        --chunk-size 1500 \\
        --force

REQUIREMENTS:
-------------
- GOOGLE_API_KEY environment variable set
- Prepared JSONL file (run prepare_dataset.py first)

================================================================================
"""

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str) -> None:
    """Configure logging for the script."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )


def check_environment() -> bool:
    """Check that required environment variables are set."""
    if not os.getenv("GOOGLE_API_KEY"):
        print("ERROR: GOOGLE_API_KEY environment variable not set.")
        print("Set it with: export GOOGLE_API_KEY=your-api-key")
        return False
    return True


def main() -> None:
    """Main entry point for indexing."""
    parser = argparse.ArgumentParser(
        description="Index Jira issues into ChromaDB for RAG retrieval.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index with defaults
  python scripts/index_documents.py --input data/processed/jira_issues.jsonl
  
  # Force re-index (delete existing collection)
  python scripts/index_documents.py --input data/processed/jira_issues.jsonl --force
  
  # Custom collection name
  python scripts/index_documents.py --input data/processed/jira_issues.jsonl --collection my_tickets
        """,
    )
    
    parser.add_argument(
        "--input",
        required=True,
        help="Path to prepared JSONL file",
    )
    parser.add_argument(
        "--collection",
        default="jira_issues",
        help="ChromaDB collection name (default: jira_issues)",
    )
    parser.add_argument(
        "--persist-dir",
        default="data/chroma",
        help="ChromaDB persistence directory (default: data/chroma)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1500,
        help="Maximum chunk size in characters (default: 1500)",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between chunks in characters (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Embedding batch size (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-index by deleting existing collection",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Validate input file
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        logger.error(f"Input file not found: {input_path}")
        sys.exit(1)
    
    # Import pipeline components (after environment check)
    from src.pipeline.indexing import (
        IndexingPipeline,
        IndexingConfig,
        IndexingResult,
    )
    from src.vectorstore import delete_collection, collection_exists
    
    # Handle force re-index
    if args.force:
        if collection_exists():
            logger.info(f"Force flag set - deleting existing collection: {args.collection}")
            delete_collection(args.collection, args.persist_dir)
    
    # Configure pipeline
    config = IndexingConfig(
        data_path=str(input_path),
        collection_name=args.collection,
        persist_directory=args.persist_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        batch_size=args.batch_size,
    )
    
    logger.info("=" * 60)
    logger.info("INDEXING CONFIGURATION")
    logger.info("=" * 60)
    logger.info(f"Input file:     {config.data_path}")
    logger.info(f"Collection:     {config.collection_name}")
    logger.info(f"Persist dir:    {config.persist_directory}")
    logger.info(f"Chunk size:     {config.chunk_size}")
    logger.info(f"Chunk overlap:  {config.chunk_overlap}")
    logger.info(f"Batch size:     {config.batch_size}")
    logger.info("=" * 60)
    
    # Run pipeline
    pipeline = IndexingPipeline(config)
    
    try:
        result = pipeline.run()
        
        # Print results
        logger.info("=" * 60)
        logger.info("INDEXING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Documents loaded:  {result.documents_loaded}")
        logger.info(f"Chunks created:    {result.chunks_created}")
        logger.info(f"Chunks indexed:    {result.chunks_indexed}")
        logger.info(f"Elapsed time:      {result.duration_seconds:.2f}s")
        logger.info("=" * 60)
        
        if result.errors:
            logger.warning(f"Errors encountered: {len(result.errors)}")
            for error in result.errors[:5]:  # Show first 5
                logger.warning(f"  - {error}")
        
        print(f"\n✓ Successfully indexed {result.chunks_indexed} chunks")
        print(f"  Collection: {args.collection}")
        print(f"  Location:   {args.persist_dir}")
        
    except Exception as e:
        logger.error(f"Indexing failed: {e}")
        raise


if __name__ == "__main__":
    main()


# =============================================================================
# TUTORIAL NOTES
# =============================================================================
#
# WHAT THIS SCRIPT DOES:
# 1. Validates environment (API key)
# 2. Configures indexing pipeline
# 3. Runs the full indexing workflow
# 4. Reports results and errors
#
# KEY FEATURES:
# - --force flag for re-indexing
# - Configurable chunk sizes
# - Detailed logging
# - Error reporting
#
# INTERVIEW TALKING POINT:
# "The indexing script is idempotent - you can run it multiple times
# without creating duplicates, thanks to stable chunk IDs."
#
# =============================================================================
