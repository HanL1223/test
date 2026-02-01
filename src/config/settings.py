"""
================================================================================
TUTORIAL: Centralized Configuration with Pydantic Settings
================================================================================

WHY PYDANTIC SETTINGS?
----------------------
Pydantic Settings provides:
  1. Type-safe environment variable loading
  2. Automatic validation (wrong types fail fast)
  3. Default values with override capability
  4. Documentation via type hints
  5. Nested configuration groups

LANGCHAIN RELEVANCE:
--------------------
LangChain components need configuration (API keys, model names, etc.).
Rather than passing these around everywhere, we centralize them here.
Components can then import and use these settings directly.

USAGE PATTERNS:
---------------
# Import the singleton
from src.config.settings import settings

# Use in code
llm = ChatGoogleGenerativeAI(
    model=settings.llm.model,
    temperature=settings.llm.temperature,
    google_api_key=settings.google_api_key,
)

================================================================================
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


# =============================================================================
# SECTION 1: NESTED CONFIGURATION CLASSES
# =============================================================================
# We break configuration into logical groups for better organization.
# Each group becomes a nested object in the main Settings class.


class LLMSettings(BaseSettings):
    """
    Configuration for the LLM (Language Model).
    
    INTERVIEW TALKING POINT:
    "We use the cheapest Gemini 2.5 model (gemini-2.5-flash) to balance
    cost and quality. For production, we could A/B test with gemini-2.5-pro
    for complex tickets."
    """
    model_config = SettingsConfigDict(env_prefix="GEMINI_")
    
    # Model name - using the cheapest Gemini 2.5 model
    model: str = Field(
        default="gemini-2.5-flash",
        description="Gemini model name for text generation"
    )
    
    # Temperature controls randomness (0=deterministic, 1=creative)
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Generation temperature"
    )
    
    # Maximum tokens in response
    max_tokens: int = Field(
        default=2048,
        alias="max_output_tokens",
        description="Maximum output tokens"
    )


class EmbeddingSettings(BaseSettings):
    """
    Configuration for the embedding model.
    
    WHY SEPARATE FROM LLM?
    Different models for different purposes:
      - LLM: Text generation (expensive, smart)
      - Embedding: Text → Vector (cheap, fast)
    """
    model_config = SettingsConfigDict(env_prefix="EMBEDDING_")
    
    model: str = Field(
        default="models/text-embedding-004",
        description="Embedding model name"
    )
    
    # Batch size for embedding multiple texts
    batch_size: int = Field(
        default=100,
        ge=1,
        le=500,
        description="Batch size for embedding requests"
    )


class ChromaSettings(BaseSettings):
    """
    Configuration for ChromaDB vector store.
    
    LANGCHAIN NOTE:
    LangChain's Chroma wrapper accepts these same parameters,
    making our config directly usable.
    """
    model_config = SettingsConfigDict(env_prefix="CHROMA_")
    
    persist_dir: Path = Field(
        default=Path("./data/chromadb"),
        description="Directory for ChromaDB persistence"
    )
    
    collection: str = Field(
        default="jira_issues",
        alias="collection_name",
        description="Collection name for Jira tickets"
    )


class RAGSettings(BaseSettings):
    """
    Configuration for Retrieval-Augmented Generation.
    
    TUNING GUIDANCE:
    - top_k: More = richer context but higher latency/cost
    - chunk_size: Larger = more context per chunk, fewer chunks
    - chunk_overlap: More = better continuity, more redundancy
    """
    model_config = SettingsConfigDict(env_prefix="RAG_")
    
    top_k: int = Field(
        default=6,
        ge=1,
        le=20,
        description="Number of documents to retrieve"
    )
    
    chunk_size: int = Field(
        default=1200,
        ge=100,
        le=4000,
        description="Text chunk size in characters"
    )
    
    chunk_overlap: int = Field(
        default=200,
        ge=0,
        le=1000,
        description="Overlap between chunks"
    )
    
    @field_validator("chunk_overlap")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        """Ensure overlap is less than chunk size."""
        # Note: In Pydantic v2, we can't easily access other fields here
        # Validation happens at Settings level
        return v


class ServerSettings(BaseSettings):
    """
    Configuration for the FastAPI server.
    """
    model_config = SettingsConfigDict(env_prefix="")
    
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Server port"
    )
    
    env: str = Field(
        default="development",
        description="Environment (development/staging/production)"
    )
    
    log_level: str = Field(
        default="INFO",
        description="Logging level"
    )
    
    cors_origins: str = Field(
        default="http://localhost:3000,http://localhost:80",
        description="Comma-separated CORS origins"
    )
    
    @property
    def cors_origins_list(self) -> list[str]:
        """Parse CORS origins into a list."""
        return [origin.strip() for origin in self.cors_origins.split(",")]
    
    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.env.lower() == "development"


class JiraSettings(BaseSettings):
    """
    Configuration for Jira Cloud API integration.
    
    WHY JIRA INTEGRATION?
    ---------------------
    The agent can not only GENERATE tickets but also CREATE them
    directly in Jira. This completes the automation loop:
    
    User Request → Agent Generates → Agent Creates in Jira
    
    SETUP:
    ------
    1. Get API token: https://id.atlassian.com/manage-profile/security/api-tokens
    2. Set environment variables:
       JIRA_BASE_URL=https://your-domain.atlassian.net
       JIRA_EMAIL=your-email@company.com
       JIRA_API_TOKEN=your-api-token
       JIRA_PROJECT_KEY=PROJ
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="JIRA_",
        extra="ignore",
    )
    
    # Jira instance URL (e.g., https://company.atlassian.net)
    base_url: str = Field(
        default="",
        description="Jira Cloud instance base URL"
    )
    
    # Atlassian account email
    email: str = Field(
        default="",
        description="Atlassian account email for authentication"
    )
    
    # API token (NOT password)
    api_token: str = Field(
        default="",
        description="Jira API token for authentication"
    )
    
    # Default project key
    project_key: str = Field(
        default="",
        description="Default Jira project key (e.g., PROJ)"
    )
    
    # Default assignee (optional)
    assignee_account_id: Optional[str] = Field(
        default=None,
        description="Default assignee Jira account ID"
    )
    
    # Custom field for start date (optional, project-specific)
    start_date_field: Optional[str] = Field(
        default=None,
        description="Custom field ID for start date (e.g., customfield_10045)"
    )
    
    @property
    def is_configured(self) -> bool:
        """Check if Jira integration is fully configured."""
        return bool(
            self.base_url and 
            self.email and 
            self.api_token and 
            self.project_key
        )


# =============================================================================
# SECTION 2: MAIN SETTINGS CLASS
# =============================================================================
# This aggregates all configuration groups and adds top-level settings.


class Settings(BaseSettings):
    """
    Main application settings.
    
    ARCHITECTURE PATTERN: Singleton via lru_cache
    We use @lru_cache on get_settings() to ensure only one Settings
    instance exists. This is important because:
      1. Environment is read once, not repeatedly
      2. Validation happens once at startup
      3. All code shares the same config
    
    USAGE:
        from src.config.settings import settings
        print(settings.google_api_key)
        print(settings.llm.model)
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore unknown env vars
        case_sensitive=False,
    )
    
    # -------------------------------------------------------------------------
    # API Keys
    # -------------------------------------------------------------------------
    google_api_key: str = Field(
        default="",
        description="Google API key for Gemini"
    )
    
    # -------------------------------------------------------------------------
    # Nested Configuration Groups
    # -------------------------------------------------------------------------
    # These are populated from environment variables with their respective prefixes
    llm: LLMSettings = Field(default_factory=LLMSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    chroma: ChromaSettings = Field(default_factory=ChromaSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    server: ServerSettings = Field(default_factory=ServerSettings)
    jira: JiraSettings = Field(default_factory=JiraSettings)  # NEW: Jira integration
    
    # -------------------------------------------------------------------------
    # Computed Properties
    # -------------------------------------------------------------------------
    @property
    def project_root(self) -> Path:
        """Get the project root directory."""
        return Path(__file__).resolve().parents[2]
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory."""
        return self.project_root / "data"
    
    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    @field_validator("google_api_key")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        """Warn if API key is missing (don't fail - might be set later)."""
        if not v or v == "your-google-api-key-here":
            import warnings
            warnings.warn(
                "GOOGLE_API_KEY is not set. API calls will fail. "
                "Set it in .env or as an environment variable."
            )
        return v


# =============================================================================
# SECTION 3: SINGLETON ACCESS
# =============================================================================
# We provide a cached function to get the settings instance.


@lru_cache
def get_settings() -> Settings:
    """
    Get the application settings (singleton).
    
    Uses lru_cache to ensure only one instance is created.
    This is the recommended pattern for Pydantic Settings.
    
    Returns:
        Settings: Application configuration
    """
    return Settings()


# Convenience: Direct access to settings
# This lets you do: from src.config.settings import settings
settings = get_settings()


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Pydantic Settings for type-safe config management
# 2. Nested configuration groups for organization
# 3. Environment variable prefixes (GEMINI_, CHROMA_, etc.)
# 4. Singleton pattern with lru_cache
# 5. Property methods for computed values
#
# INTERVIEW TALKING POINTS:
# - "We centralize all configuration to avoid scattered env var reads"
# - "Type validation catches config errors at startup, not runtime"
# - "Nested groups make it easy to pass config to LangChain components"
#
# =============================================================================
