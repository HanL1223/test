"""
Configuration package for the Jira Ticket RAG system.

Exports:
    settings: Singleton Settings instance
    get_settings: Function to get settings (for dependency injection)
    Settings: Settings class (for type hints)
"""

from src.config.settings import Settings, get_settings, settings

__all__ = ["Settings", "get_settings", "settings"]
