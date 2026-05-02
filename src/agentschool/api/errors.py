"""API error types for AgentSchool."""

from __future__ import annotations


class AgentSchoolApiError(RuntimeError):
    """Base class for upstream API failures."""


class AuthenticationFailure(AgentSchoolApiError):
    """Raised when the upstream service rejects the provided credentials."""


class RateLimitFailure(AgentSchoolApiError):
    """Raised when the upstream service rejects the request due to rate limits."""


class RequestFailure(AgentSchoolApiError):
    """Raised for generic request or transport failures."""
