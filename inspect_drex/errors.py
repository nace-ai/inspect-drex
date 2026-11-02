"""Errors raised by inspect-drex.

Messages never include the request or response body, so state cannot leak into logs.
"""

from __future__ import annotations

import json
import re
from typing import Any

import nace_sdk

# Where to add funds when no base URL is configured. A configured base URL (the test console,
# a customer's own Drex deployment) has its own Billing page: see `billing_url`.
BILLING_URL = "https://console.nace.ai/dashboard/billing"


def billing_url(base_url: str | None) -> str:
    """The Billing page of the Drex at `base_url`: `<base_url>/dashboard/billing`."""
    if not base_url:
        return BILLING_URL
    root = base_url.strip().rstrip("/").removesuffix("/v1").rstrip("/")
    return f"{root}/dashboard/billing"


class DrexError(Exception):
    """Base class for every error this package raises."""


class DrexImageError(DrexError, ValueError):
    """An image can't be sent as given (a remote URL, an unknown type, ...)."""


class DrexImageModelError(DrexImageError):
    """Images were sent to a model that reads text only."""


class DrexAPIError(DrexError):
    """Drex (or the gateway in front of it) refused or failed a request.

    `status`, `request_id`, `type` and `issues` (`{"path", "message"}` per problem) come
    from the response. An execution error is never a failed criterion: it propagates.
    """

    def __init__(self, error: nace_sdk.NaceAPIError, message: str, issues: list[dict[str, str]]):
        super().__init__(message)
        self.status: int = error.status
        self.request_id: str | None = error.request_id
        self.type: str | None = error.type
        self.issues = issues


class DrexConnectionError(DrexError):
    """Drex could not be reached, or the request timed out."""


def _unwrap(message: str) -> tuple[str, list[dict[str, str]]]:
    """Make a gateway's raw error text readable (OpenRouter passes upstream text through)."""
    text = message.strip()
    wrapped = re.match(r"^HTTP \d{3}: (.*)$", text, re.DOTALL)
    if wrapped:
        text = wrapped.group(1).strip()
    try:
        data: Any = json.loads(text)
    except ValueError:
        return text, []
    if isinstance(data, dict):
        inner = data.get("detail") or data.get("message") or data.get("error")
        if isinstance(inner, str):
            return inner, []
        if isinstance(inner, dict) and isinstance(inner.get("message"), str):
            return inner["message"], []
        data = inner
    if isinstance(data, list):
        issues = []
        for item in data:
            if isinstance(item, dict) and item.get("message"):
                path = item.get("path", "")
                if isinstance(path, list):
                    path = ".".join(str(part) for part in path)
                issues.append({"path": str(path), "message": str(item["message"])})
        if issues:
            return "the request is invalid", issues
    return text, []


def convert_error(
    error: Exception, provider: str = "Drex", billing: str | None = BILLING_URL
) -> Exception:
    """Map an `nace_sdk` error to this package's errors; other errors pass through."""
    if isinstance(error, nace_sdk.NaceAPIError):
        message, found = _unwrap(error.message) if error.message else (None, [])
        issues = error.issues or found
        text = f"{provider} returned {error.status}"
        if message:
            text += f": {message}"
        lines = [
            f"{i.get('path') or '(request)'}: {i['message']}" for i in issues if i.get("message")
        ]
        if lines and lines != [message]:
            text += "\n" + "\n".join(f"  {line}" for line in lines)
        if error.status == 402:
            text += (
                f"\nAdd funds at {billing}"
                if billing
                else f"\nAdd funds to your {provider} account."
            )
        if error.request_id:
            text += f"\n(request_id={error.request_id})"
        return DrexAPIError(error, text, issues)
    if isinstance(error, nace_sdk.NaceAPIConnectionError):
        return DrexConnectionError(str(error))
    if isinstance(error, nace_sdk.NaceError):
        return DrexError(str(error))
    return error
