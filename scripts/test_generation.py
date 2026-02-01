#!/usr/bin/env python3
"""
================================================================================
TUTORIAL: ReAct Agent Testing Script
================================================================================

PURPOSE:
--------
This script tests the Jira ticket generation AGENT (not workflow!).
It's useful for:
- Observing agent reasoning and decision-making
- Demonstrating agentic behavior to stakeholders
- Quick ticket generation without running the full API

THIS IS AN AGENT, NOT A WORKFLOW:
---------------------------------
The agent REASONS about what to do, CHOOSES tools, OBSERVES results,
and ADAPTS its strategy. Use --verbose to see the reasoning trace!

USAGE:
------
    # Interactive mode
    python scripts/test_generation.py
    
    # Single request with reasoning trace
    python scripts/test_generation.py --request "Create a ticket for data migration" --verbose
    
    # Fast mode (fewer iterations)
    python scripts/test_generation.py --request "Add user auth" --fast

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
        format="%(asctime)s - %(levelname)s - %(message)s",
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


def print_divider(char: str = "=", width: int = 70) -> None:
    """Print a divider line."""
    print(char * width)


def print_header(text: str) -> None:
    """Print a section header."""
    print_divider()
    print(text)
    print_divider()


def generate_single(request: str, fast_mode: bool, verbose: bool) -> None:
    """
    Generate a single ticket using the ReAct Agent and display results.
    
    Args:
        request: User's ticket request
        fast_mode: Reduce agent iterations
        verbose: Show detailed output including reasoning trace
    """
    from src.pipeline.generation import (
        JiraTicketPipeline,
        PipelineConfig,
        GenerationResult,
        generate_ticket,
    )
    
    print_header("🤖 JIRA TICKET AGENT")
    print(f"Request: {request}")
    print(f"Mode: {'Fast (fewer iterations)' if fast_mode else 'Full (complete reasoning)'}")
    print_divider("-")
    
    # Configure agent
    config = PipelineConfig(
        max_iterations=5 if fast_mode else 10,
        verbose=verbose,
    )
    
    pipeline = JiraTicketPipeline(config=config)
    
    print("⏳ Agent is reasoning...")
    result = pipeline.generate(request)
    
    # Show agent reasoning trace if verbose
    if verbose and result.reasoning_trace:
        print_header("🧠 AGENT REASONING TRACE")
        print("This shows the agent's decision-making process:\n")
        for i, step in enumerate(result.reasoning_trace, 1):
            print(f"Step {i}:")
            thought = step.get("thought", "")
            if thought:
                # Extract just the thought part, not the full log
                thought_lines = thought.split("\n")
                for line in thought_lines[:3]:  # Show first 3 lines
                    if line.strip():
                        print(f"  💭 Thought: {line.strip()[:100]}")
            print(f"  🔧 Action: {step.get('action', 'unknown')}")
            action_input = step.get("action_input", "")
            if action_input:
                print(f"  📥 Input: {action_input[:100]}...")
            observation = step.get("observation", "")
            if observation:
                print(f"  👁️ Observation: {observation[:150]}...")
            print()
        print_divider("-")
    
    # Show tools used
    if result.tools_used:
        print_header("🛠️ TOOLS USED BY AGENT")
        for tool, count in result.tools_used.items():
            print(f"  {tool}: {count}x")
        print_divider("-")
    
    # Show final result
    print_header("✅ GENERATED TICKET")
    print(result.ticket_text)
    
    # Show metadata
    print_divider("-")
    print(f"Style detected:     {result.style_detected}")
    print(f"Refinement applied: {result.refinement_applied}")
    print(f"Agent iterations:   {result.iterations}")
    print(f"Tools used:         {result.tools_used}")
    print(f"Generation time:    {result.metadata.get('elapsed_seconds', 0):.2f}s")
    print_divider()


def interactive_mode(fast_mode: bool, verbose: bool) -> None:
    """
    Run in interactive mode with the ReAct Agent.
    
    Args:
        fast_mode: Reduce agent iterations
        verbose: Show detailed reasoning trace
    """
    from src.pipeline import (
        JiraTicketPipeline,
        PipelineConfig,
    )
    
    print_header("🤖 JIRA TICKET AGENT - INTERACTIVE MODE")
    print("This is a ReAct AGENT that reasons and decides, not a fixed workflow!")
    print("\nType your ticket request and press Enter.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'help' for commands.")
    print_divider("-")
    
    config = PipelineConfig(
        max_iterations=5 if fast_mode else 10,
        verbose=verbose,
    )
    
    pipeline = JiraTicketPipeline(config=config)
    
    while True:
        try:
            request = input("\n📝 Request: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break
        
        if not request:
            continue
        
        if request.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break
        
        if request.lower() == "help":
            print("\nCommands:")
            print("  quit/exit/q  - Exit interactive mode")
            print("  fast         - Toggle fast mode (fewer iterations)")
            print("  verbose      - Toggle verbose output (show reasoning)")
            print("  trace        - Toggle reasoning trace display")
            print("  help         - Show this help")
            print("\nOr type any ticket request for the agent to process.")
            continue
        
        if request.lower() == "fast":
            fast_mode = not fast_mode
            config = PipelineConfig(
                max_iterations=5 if fast_mode else 10,
                verbose=verbose,
            )
            pipeline = JiraTicketPipeline(config=config)
            print(f"Fast mode: {'ON (5 iterations)' if fast_mode else 'OFF (10 iterations)'}")
            continue
        
        if request.lower() in ("verbose", "trace"):
            verbose = not verbose
            print(f"Verbose/trace mode: {'ON' if verbose else 'OFF'}")
            continue
        
        # Run agent
        try:
            print("\n⏳ Agent is reasoning...")
            result = pipeline.generate(request)
            
            # Show reasoning trace if verbose
            if verbose and result.reasoning_trace:
                print("\n🧠 Agent reasoning:")
                for i, step in enumerate(result.reasoning_trace[:5], 1):  # Show first 5
                    action = step.get("action", "unknown")
                    print(f"  {i}. {action}")
            
            # Show tools used
            if result.tools_used:
                tools_str = ", ".join(f"{t}:{c}" for t, c in result.tools_used.items())
                print(f"\n🛠️ Tools used: {tools_str}")
            
            print("\n" + "=" * 50)
            print("✅ GENERATED TICKET")
            print("=" * 50)
            print(result.ticket_text)
            print("=" * 50)
            print(f"Style: {result.style_detected} | "
                  f"Iterations: {result.iterations} | "
                  f"Time: {result.metadata.get('elapsed_seconds', 0):.2f}s")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
            logging.getLogger().exception("Generation failed")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test Jira ticket generation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python scripts/test_generation.py
  
  # Single request
  python scripts/test_generation.py --request "Create login feature"
  
  # Fast mode with verbose output
  python scripts/test_generation.py --request "Fix bug" --fast --verbose
        """,
    )
    
    parser.add_argument(
        "--request", "-r",
        help="Ticket request (if not provided, runs in interactive mode)",
    )
    parser.add_argument(
        "--fast", "-f",
        action="store_true",
        help="Fast mode - skip multi-agent refinement",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output - show retrieved context and draft",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: WARNING)",
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check environment
    if not check_environment():
        sys.exit(1)
    
    # Run generation
    if args.request:
        generate_single(args.request, args.fast, args.verbose)
    else:
        interactive_mode(args.fast, args.verbose)


if __name__ == "__main__":
    main()


# =============================================================================
# TUTORIAL NOTES
# =============================================================================
#
# WHAT THIS SCRIPT PROVIDES:
# 1. Quick testing without API server
# 2. Interactive mode for exploration
# 3. Verbose mode for debugging
# 4. Fast mode for speed testing
#
# INTERVIEW TALKING POINT:
# "Having CLI tools separate from the API makes development faster -
# you can test generation without spinning up the full server."
#
# =============================================================================
