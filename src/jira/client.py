"""
================================================================================
TUTORIAL: Jira Cloud REST API Client
================================================================================

PURPOSE:
--------
This module provides a client for interacting with Jira Cloud REST API.
It handles:
- Authentication (Basic auth with API token)
- Issue creation
- Issue updates
- ADF conversion for descriptions

JIRA CLOUD API:
---------------
Jira Cloud uses REST API v3 which requires:
- Basic auth with email + API token
- Atlassian Document Format (ADF) for rich text fields
- JSON payloads with specific field structures

AUTHENTICATION:
---------------
Generate an API token at:
https://id.atlassian.com/manage-profile/security/api-tokens

Then set environment variables:
    JIRA_EMAIL=your-email@company.com
    JIRA_API_TOKEN=your-api-token

================================================================================
"""

import base64
import json
import logging
from typing import Optional
from functools import lru_cache

import requests
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import settings
from src.jira.models import (
    JiraIssueCreate,
    JiraIssueUpdate,
    JiraCreateResult,
    JiraUpdateResult,
    text_to_adf,
    jira_markup_to_adf,
)

logger = logging.getLogger(__name__)


# =============================================================================
# SECTION 1: JIRA CLIENT CLASS
# =============================================================================


class JiraClient:
    """
    Client for Jira Cloud REST API.
    
    FEATURES:
    ---------
    - Create issues with automatic ADF conversion
    - Update existing issues
    - Retry logic for transient failures
    - Configurable via environment variables
    
    USAGE:
        client = JiraClient(
            base_url="https://company.atlassian.net",
            email="user@company.com",
            api_token="your-token",
            project_key="PROJ",
        )
        
        result = client.create_issue(
            summary="New feature",
            description="Implement...",
        )
        print(f"Created: {result.url}")
    """
    
    def __init__(
        self,
        base_url: str,
        email: str,
        api_token: str,
        project_key: str,
        assignee_account_id: Optional[str] = None,
        start_date_field: Optional[str] = None,
    ):
        """
        Initialize the Jira client.
        
        Args:
            base_url: Jira instance URL (e.g., https://company.atlassian.net)
            email: Atlassian account email
            api_token: API token for authentication
            project_key: Default project key (e.g., "PROJ")
            assignee_account_id: Default assignee (optional)
            start_date_field: Custom field ID for start date (optional)
        """
        self.base_url = base_url.rstrip("/")
        self.email = email
        self.api_token = api_token
        self.project_key = project_key
        self.assignee_account_id = assignee_account_id
        self.start_date_field = start_date_field
        
        # Create session with auth
        self._session = requests.Session()
        self._session.headers.update(self._get_auth_header())
        self._session.headers.update({
            "Content-Type": "application/json",
            "Accept": "application/json",
        })
        
        logger.info(
            f"Initialized JiraClient: base_url={self.base_url}, "
            f"project={self.project_key}"
        )
    
    def _get_auth_header(self) -> dict:
        """Create Basic auth header."""
        auth_str = f"{self.email}:{self.api_token}"
        b64 = base64.b64encode(auth_str.encode("utf-8")).decode("utf-8")
        return {"Authorization": f"Basic {b64}"}
    
    # -------------------------------------------------------------------------
    # CREATE ISSUE
    # -------------------------------------------------------------------------
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def create_issue(
        self,
        summary: str,
        description: Optional[str] = None,
        issue_type: str = "Task",
        labels: Optional[list[str]] = None,
        assignee_account_id: Optional[str] = None,
        start_date: Optional[str] = None,
        project_key: Optional[str] = None,
        use_jira_markup: bool = False,
        custom_fields: Optional[dict] = None,
    ) -> JiraCreateResult:
        """
        Create a new Jira issue.
        
        Args:
            summary: Issue title (required)
            description: Issue description (plain text or Jira markup)
            issue_type: Task, Story, Bug, Epic, etc.
            labels: List of labels to apply
            assignee_account_id: Override default assignee
            start_date: Start date (YYYY-MM-DD), defaults to today
            project_key: Override default project
            use_jira_markup: If True, parse description as Jira markup
            custom_fields: Dict of custom field ID → value
            
        Returns:
            JiraCreateResult with success status and issue details
            
        Example:
            >>> result = client.create_issue(
            ...     summary="Implement OAuth2",
            ...     description="Add OAuth2 support for API auth",
            ...     issue_type="Story",
            ...     labels=["backend", "security"],
            ... )
            >>> print(result.url)
            https://company.atlassian.net/browse/PROJ-123
        """
        # Build issue data
        issue_data = JiraIssueCreate(
            summary=summary,
            description=description,
            issue_type=issue_type,
            labels=labels or [],
            assignee_account_id=assignee_account_id or self.assignee_account_id,
            start_date=start_date,
            custom_fields=custom_fields or {},
        )
        
        # Build fields payload
        fields = {
            "project": {"key": project_key or self.project_key},
            "summary": issue_data.summary,
            "issuetype": {"name": issue_data.issue_type},
        }
        
        # Add description in ADF format
        if issue_data.description:
            if use_jira_markup:
                adf = jira_markup_to_adf(issue_data.description)
            else:
                adf = text_to_adf(issue_data.description)
            if adf:
                fields["description"] = adf
        
        # Add assignee
        if issue_data.assignee_account_id:
            fields["assignee"] = {"accountId": issue_data.assignee_account_id}
        
        # Add labels
        if issue_data.labels:
            fields["labels"] = issue_data.labels
        
        # Add start date custom field
        if self.start_date_field and issue_data.start_date:
            fields[self.start_date_field] = issue_data.start_date
        
        # Add any custom fields
        for field_id, value in issue_data.custom_fields.items():
            fields[field_id] = value
        
        payload = {"fields": fields}
        
        logger.debug(f"Creating issue: {summary[:50]}...")
        
        # Make API request
        url = f"{self.base_url}/rest/api/3/issue"
        try:
            resp = self._session.post(url, data=json.dumps(payload))
            
            if resp.status_code not in (200, 201):
                error_msg = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.error(f"Failed to create issue: {error_msg}")
                return JiraCreateResult(
                    success=False,
                    error=error_msg,
                )
            
            data = resp.json()
            key = data.get("key")
            issue_id = data.get("id")
            browse_url = f"{self.base_url}/browse/{key}"
            
            logger.info(f"Created issue: {key} - {summary[:50]}")
            
            return JiraCreateResult(
                success=True,
                key=key,
                id=issue_id,
                url=browse_url,
            )
            
        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"Failed to create issue: {error_msg}")
            return JiraCreateResult(
                success=False,
                error=error_msg,
            )
    
    # -------------------------------------------------------------------------
    # UPDATE ISSUE
    # -------------------------------------------------------------------------
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def update_issue(
        self,
        issue_key: str,
        description: Optional[str] = None,
        summary: Optional[str] = None,
        labels: Optional[list[str]] = None,
        use_jira_markup: bool = False,
        custom_fields: Optional[dict] = None,
    ) -> JiraUpdateResult:
        """
        Update an existing Jira issue.
        
        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            description: New description (optional)
            summary: New summary (optional)
            labels: New labels (optional, replaces existing)
            use_jira_markup: If True, parse description as Jira markup
            custom_fields: Custom fields to update
            
        Returns:
            JiraUpdateResult with success status
            
        Example:
            >>> result = client.update_issue(
            ...     issue_key="PROJ-123",
            ...     description="Updated description with more details",
            ... )
            >>> print(result.success)
            True
        """
        issue_key = issue_key.strip().upper()
        
        # Build fields to update
        fields = {}
        
        if summary is not None:
            fields["summary"] = summary
        
        if description is not None:
            if use_jira_markup:
                adf = jira_markup_to_adf(description)
            else:
                adf = text_to_adf(description)
            if adf:
                fields["description"] = adf
        
        if labels is not None:
            fields["labels"] = labels
        
        # Add custom fields
        if custom_fields:
            for field_id, value in custom_fields.items():
                fields[field_id] = value
        
        if not fields:
            logger.warning(f"No fields to update for {issue_key}")
            return JiraUpdateResult(
                success=True,
                key=issue_key,
                url=f"{self.base_url}/browse/{issue_key}",
            )
        
        payload = {"fields": fields}
        
        logger.debug(f"Updating issue: {issue_key}")
        
        # Make API request
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        try:
            resp = self._session.put(url, data=json.dumps(payload))
            
            if resp.status_code not in (200, 204):
                error_msg = f"HTTP {resp.status_code}: {resp.text[:500]}"
                logger.error(f"Failed to update {issue_key}: {error_msg}")
                return JiraUpdateResult(
                    success=False,
                    key=issue_key,
                    error=error_msg,
                )
            
            browse_url = f"{self.base_url}/browse/{issue_key}"
            logger.info(f"Updated issue: {issue_key}")
            
            return JiraUpdateResult(
                success=True,
                key=issue_key,
                url=browse_url,
            )
            
        except requests.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            logger.error(f"Failed to update {issue_key}: {error_msg}")
            return JiraUpdateResult(
                success=False,
                key=issue_key,
                error=error_msg,
            )
    
    # -------------------------------------------------------------------------
    # GET ISSUE
    # -------------------------------------------------------------------------
    
    def get_issue(self, issue_key: str) -> Optional[dict]:
        """
        Get issue details.
        
        Args:
            issue_key: Issue key (e.g., "PROJ-123")
            
        Returns:
            Issue data dict, or None if not found
        """
        issue_key = issue_key.strip().upper()
        url = f"{self.base_url}/rest/api/3/issue/{issue_key}"
        
        try:
            resp = self._session.get(url)
            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 404:
                logger.warning(f"Issue not found: {issue_key}")
                return None
            else:
                logger.error(f"Failed to get {issue_key}: {resp.status_code}")
                return None
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return None
    
    # -------------------------------------------------------------------------
    # HEALTH CHECK
    # -------------------------------------------------------------------------
    
    def health_check(self) -> bool:
        """
        Check if Jira API is accessible.
        
        Returns:
            True if API is accessible, False otherwise
        """
        url = f"{self.base_url}/rest/api/3/myself"
        try:
            resp = self._session.get(url)
            return resp.status_code == 200
        except requests.RequestException:
            return False


# =============================================================================
# SECTION 2: FACTORY FUNCTIONS
# =============================================================================


def create_jira_client(
    base_url: Optional[str] = None,
    email: Optional[str] = None,
    api_token: Optional[str] = None,
    project_key: Optional[str] = None,
) -> JiraClient:
    """
    Create a Jira client from configuration.
    
    Uses settings from environment variables if not provided.
    
    Args:
        base_url: Override JIRA_BASE_URL
        email: Override JIRA_EMAIL
        api_token: Override JIRA_API_TOKEN
        project_key: Override JIRA_PROJECT_KEY
        
    Returns:
        Configured JiraClient
        
    Raises:
        ValueError: If required configuration is missing
    """
    # Get from settings or args
    base_url = base_url or settings.jira.base_url
    email = email or settings.jira.email
    api_token = api_token or settings.jira.api_token
    project_key = project_key or settings.jira.project_key
    
    # Validate required fields
    missing = []
    if not base_url:
        missing.append("JIRA_BASE_URL")
    if not email:
        missing.append("JIRA_EMAIL")
    if not api_token:
        missing.append("JIRA_API_TOKEN")
    if not project_key:
        missing.append("JIRA_PROJECT_KEY")
    
    if missing:
        raise ValueError(
            f"Missing required Jira configuration: {', '.join(missing)}. "
            "Set these environment variables or pass them as arguments."
        )
    
    return JiraClient(
        base_url=base_url,
        email=email,
        api_token=api_token,
        project_key=project_key,
        assignee_account_id=settings.jira.assignee_account_id,
        start_date_field=settings.jira.start_date_field,
    )


# Cached singleton
_client_instance: Optional[JiraClient] = None


def get_jira_client() -> JiraClient:
    """
    Get a cached singleton Jira client.
    
    Returns:
        Shared JiraClient instance
    """
    global _client_instance
    if _client_instance is None:
        _client_instance = create_jira_client()
    return _client_instance


def clear_jira_client_cache():
    """Clear the cached client instance."""
    global _client_instance
    _client_instance = None


# =============================================================================
# TUTORIAL REVIEW
# =============================================================================
#
# WHAT YOU LEARNED:
# 1. Jira Cloud REST API v3 authentication
# 2. Converting text to Atlassian Document Format (ADF)
# 3. Creating and updating issues programmatically
# 4. Retry logic for API resilience
#
# JIRA API NOTES:
# - API tokens: https://id.atlassian.com/manage-profile/security/api-tokens
# - API docs: https://developer.atlassian.com/cloud/jira/platform/rest/v3/
# - ADF spec: https://developer.atlassian.com/cloud/jira/platform/apis/document/structure/
#
# INTERVIEW TALKING POINTS:
# - "The client handles ADF conversion automatically"
# - "Retry logic with exponential backoff handles transient failures"
# - "Configuration comes from environment variables for security"
#
# =============================================================================
