"""
================================================================================
Jira Integration Package
================================================================================

This package provides integration with Jira Cloud REST API for:
- Creating new issues
- Updating existing issues
- Querying issue information

CONFIGURATION:
--------------
Set these environment variables:
    JIRA_BASE_URL=https://your-domain.atlassian.net
    JIRA_EMAIL=your-email@company.com
    JIRA_API_TOKEN=your-api-token
    JIRA_PROJECT_KEY=PROJ

USAGE:
------
    from src.jira import JiraClient, create_jira_client
    
    # Create client
    client = create_jira_client()
    
    # Create issue
    result = client.create_issue(
        summary="Implement user auth",
        description="Add OAuth2 support...",
        issue_type="Task",
    )
    print(f"Created: {result.key}")

================================================================================
"""

from src.jira.client import (
    JiraClient,
    create_jira_client,
    get_jira_client,
)

from src.jira.models import (
    JiraIssueCreate,
    JiraIssueUpdate,
    JiraCreateResult,
    JiraUpdateResult,
)

__all__ = [
    # Client
    "JiraClient",
    "create_jira_client",
    "get_jira_client",
    # Models
    "JiraIssueCreate",
    "JiraIssueUpdate",
    "JiraCreateResult",
    "JiraUpdateResult",
]
