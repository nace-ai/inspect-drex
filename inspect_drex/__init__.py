"""inspect-drex: an Inspect scorer backed by Drex decision models from Nace."""

from inspect_drex._client import CLIENT_NAME, AsyncDrexClient, Ruling, Usage, is_text_only
from inspect_drex._version import __version__
from inspect_drex.criteria import Criterion, Verdict
from inspect_drex.errors import (
    BILLING_URL,
    DrexAPIError,
    DrexConnectionError,
    DrexError,
    DrexImageError,
    DrexImageModelError,
    billing_url,
)
from inspect_drex.scorer import default_state, drex_scorer

__all__ = [
    "BILLING_URL",
    "CLIENT_NAME",
    "AsyncDrexClient",
    "Criterion",
    "DrexAPIError",
    "DrexConnectionError",
    "DrexError",
    "DrexImageError",
    "DrexImageModelError",
    "Ruling",
    "Usage",
    "Verdict",
    "__version__",
    "billing_url",
    "default_state",
    "drex_scorer",
    "is_text_only",
]
