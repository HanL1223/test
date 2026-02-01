# 🎫 Jira Ticket RAG System

> AI-powered Jira ticket generation using a **ReAct Agent** with LangChain

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://langchain.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Overview

This system generates professional Jira tickets using a **ReAct Agent** (not a fixed workflow). The agent **reasons** about what to do, **chooses** tools dynamically, **observes** results, and **adapts** its strategy.

### 🤖 This is an AGENT, Not a Workflow

| Feature | Workflow | Agent (This Project) |
|---------|----------|---------------------|
| Decision-making | ❌ Fixed steps | ✅ Reasons about next action |
| Tool selection | ❌ Hardcoded | ✅ Chooses tools dynamically |
| Self-correction | ❌ None | ✅ Validates and retries |
| Observation loop | ❌ None | ✅ Observe → Think → Act |
| Adaptive behavior | ❌ Same path always | ✅ Varies based on context |

### Key Features

- **🧠 ReAct Agent**: Reasons, acts, observes, adapts (not fixed pipeline)
- **🔍 Semantic Search**: Hybrid retrieval (BM25 + dense vectors)
- **🎨 Style Adaptive**: Agent detects and matches ticket styles
- **✅ Self-Validation**: Agent checks quality and self-corrects
- **🚀 Jira Integration**: Automatically create tickets in your Jira board
- **🔧 LangChain Integration**: Modular architecture with swappable providers

## 🏗️ Architecture

### ReAct Agent Loop

```
┌─────────────────────────────────────────────────────────────┐
│                    USER REQUEST                              │
│            "Create ticket for data migration"                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                    🧠 AGENT REASONING                        │
│            "What should I do first?"                         │
│            "I should search for similar tickets"             │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │     TOOL SELECTION (Agent Decides) │
        └─────────────────┬─────────────────┘
                          │
    ┌─────────┬───────────┼───────────┬─────────┬─────────┐
    ▼         ▼           ▼           ▼         ▼         ▼
┌───────┐ ┌───────┐ ┌─────────┐ ┌─────────┐ ┌───────┐ ┌───────┐
│Search │ │Generate│ │ Refine │ │Validate │ │Create │ │ ...   │
│Tickets│ │ Draft │ │ Ticket │ │ Quality │ │ Jira  │ │       │
└───┬───┘ └───┬───┘ └────┬────┘ └────┬────┘ └───┬───┘ └───────┘
    │         │          │           │          │
    └─────────┴──────────┴───────────┴──────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────┐
│                   👁️ OBSERVE RESULT                         │
│            "Found 5 similar tickets about migrations"        │
│            "Score: 7/10 - needs more technical detail"       │
└─────────────────────────┬───────────────────────────────────┘
                          │
                    ┌─────▼─────┐
                    │  Done?    │──No──→ Back to Reasoning
                    └─────┬─────┘
                          │Yes
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                 FINAL TICKET OUTPUT                          │
│          Agent-generated, validated, context-aware           │
└─────────────────────────────────────────────────────────────┘
```

### Example Agent Trace

```
Step 1: 💭 "I should search for similar tickets first"
        🔧 Action: search_similar_tickets
        👁️ Observation: Found 5 tickets about data migration

Step 2: 💭 "Good context! I'll generate a verbose draft"
        🔧 Action: generate_draft_ticket  
        👁️ Observation: Draft created (1200 chars)

Step 3: 💭 "Let me validate the quality"
        🔧 Action: validate_ticket
        👁️ Observation: Score 7/10, needs technical detail

Step 4: 💭 "I should refine with technical focus"
        🔧 Action: refine_ticket (focus=technical)
        👁️ Observation: Added implementation details

Step 5: 💭 "Re-validate the refined ticket"
        🔧 Action: validate_ticket
        👁️ Observation: Score 9/10, ready!

Step 6: 💭 "User asked to create in Jira, using create tool"
        🔧 Action: create_jira_ticket
        👁️ Observation: ✅ Created PROJ-456

Step 7: 💭 "Ticket created successfully"
        ✅ Final Answer: Created PROJ-456: https://company.atlassian.net/browse/PROJ-456
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Google API Key (for Gemini)
- Node.js 18+ (for frontend)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/jira-ticket-rag.git
cd jira-ticket-rag

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env and add your GOOGLE_API_KEY

# (Optional) For Jira integration, also set:
# JIRA_BASE_URL=https://your-domain.atlassian.net
# JIRA_EMAIL=your-email@company.com
# JIRA_API_TOKEN=your-api-token  # Generate at: https://id.atlassian.com/manage-profile/security/api-tokens
# JIRA_PROJECT_KEY=PROJ
```

### Prepare Your Data

```bash
# 1. Place your Jira CSV export in data/raw/
# 2. Process the data
python scripts/prepare_dataset.py --input data/raw/YOUR_EXPORT.csv

# 3. Index the documents
python scripts/index_documents.py --input data/processed/jira_issues.jsonl
```

### Run the System

```bash
# Option 1: CLI Testing (generates ticket)
python scripts/test_generation.py --request "Create a ticket for user authentication"

# Option 2: CLI with Jira Creation (creates ticket in Jira!)
python scripts/test_generation.py --request "Create a ticket in Jira for user authentication"

# Option 3: Interactive Mode
python scripts/test_generation.py

# Option 4: API Server
python -m uvicorn backend.main:app --reload

# Option 5: Full Stack (Docker)
docker-compose up -d
```

## 📁 Project Structure

```
jira-ticket-rag-langchain/
├── src/
│   ├── agents/          # 🤖 ReAct Agent
│   │   ├── tools.py     # Agent tools (search, generate, refine, validate, CREATE)
│   │   └── jira_agent.py # ReAct agent orchestrator
│   ├── jira/            # 🚀 Jira Cloud API Integration (NEW!)
│   │   ├── client.py    # REST API client
│   │   └── models.py    # Request/response models, ADF conversion
│   ├── config/          # Settings and configuration
│   ├── core/            # Domain models and utilities
│   ├── data/            # LangChain loaders and splitters
│   ├── embeddings/      # Gemini embeddings wrapper
│   ├── vectorstore/     # ChromaDB wrapper
│   ├── retrieval/       # Retriever components
│   ├── llm/             # Gemini LLM and prompts
│   ├── chains/          # LCEL generation chains (used by agent)
│   └── pipeline/        # High-level pipeline (wraps agent)
├── scripts/             # CLI tools
├── backend/             # FastAPI API
├── frontend/            # React app
├── data/                # Data storage
├── requirements.txt     # Python dependencies
├── docker-compose.yml   # Container orchestration
└── README.md            # This file
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `GOOGLE_API_KEY` | Google AI API key | Required |
| `GEMINI_MODEL` | LLM model name | `gemini-2.5-flash-preview-05-20` |
| `EMBEDDING_MODEL` | Embedding model | `models/text-embedding-004` |
| `CHROMA_COLLECTION_NAME` | ChromaDB collection | `jira_issues` |
| `CHROMA_PERSIST_DIRECTORY` | ChromaDB storage path | `data/chroma` |
| `RAG_TOP_K` | Number of similar tickets | `5` |
| `RAG_SCORE_THRESHOLD` | Minimum similarity score | `0.3` |

## 📖 API Reference

### Generate Ticket

```bash
POST /api/generate
Content-Type: application/json

{
  "request": "Create a ticket for implementing OAuth2 authentication",
  "fast_mode": false
}
```

**Response:**
```json
{
  "ticket_text": "h2. Overview\n\nImplement OAuth2 authentication...",
  "draft_text": "...",
  "style_detected": "verbose",
  "refinement_applied": true,
  "retrieved_chunks": [...],
  "metadata": {
    "elapsed_seconds": 3.45,
    "model": "gemini-2.5-flash-preview-05-20"
  }
}
```

### Search Tickets

```bash
POST /api/search
Content-Type: application/json

{
  "query": "data migration",
  "top_k": 5
}
```

## 🧪 Development

### Running Tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# With coverage
pytest --cov=src --cov-report=html
```

### Code Quality

```bash
# Format code
black src/ scripts/ backend/

# Lint
ruff check src/ scripts/ backend/

# Type checking
mypy src/
```

## 📈 Interview Talking Points

This project demonstrates:

1. **ReAct Agent Architecture**: The system uses a reasoning agent, NOT a fixed workflow
   - Agent THINKS about what to do next
   - Agent CHOOSES tools dynamically based on observations
   - Agent SELF-CORRECTS when validation fails
   - Execution path varies per request

2. **RAG with Hybrid Retrieval**: BM25 + dense vectors improve accuracy for technical content

3. **LangChain Tool Integration**: Clean tool abstractions enable swapping providers

4. **Self-Validation Loop**: Agent validates its output and iterates until quality threshold met

5. **Full Automation Loop**: Agent can CREATE tickets directly in Jira
   - User says "create in Jira" → Agent generates, validates, then calls Jira API
   - Complete automation from idea to ticket in one command

6. **Production Patterns**: Dependency injection, configuration management, observability

7. **Explainability**: Reasoning trace shows WHY the agent made each decision

## 🤝 Contributing

Contributions are welcome! Please read our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the excellent RAG framework
- [Google Gemini](https://ai.google.dev/) for the LLM and embeddings
- [ChromaDB](https://www.trychroma.com/) for vector storage
