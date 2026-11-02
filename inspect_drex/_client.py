"""One thin Drex client shared by the metric and the module."""

from __future__ import annotations

import asyncio
import json
import os
import threading
import weakref
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import nace_sdk

from inspect_drex._images import as_data_url, to_drex_image
from inspect_drex._version import __version__
from inspect_drex.errors import DrexImageError, DrexImageModelError, billing_url, convert_error

# The default is Drex Max, which moves to the newest flagship with each Drex release.
# Pin with `model=` or NACE_DEFAULT_DECISION_MODEL.
DEFAULT_MODEL = "drex-max"
CLIENT_NAME = f"inspect-drex/{__version__}"
MIN_TIMEOUT = 60.0
DEFAULT_MAX_IN_FLIGHT = 8

# Facts about released, pinned versions never change; every other id is sent as given.
# `drex-latest` resolves to a text-only model today, so it is refused for images too.
TEXT_ONLY_MODELS = frozenset({"drex-v1.5", "drex-v2.0", "drex-latest"})


def is_text_only(model: str) -> bool:
    """Whether `model` is known to read text only."""
    return model.rsplit("/", 1)[-1].lstrip("~") in TEXT_ONLY_MODELS


OPENROUTER_PREFIX = "nace-ai/"


def openrouter_model(model: str) -> str:
    """The OpenRouter id of `model`: a bare Drex id gets the `nace-ai/` prefix.

    `drex-latest` has no OpenRouter id, so it is refused rather than guessed; ids that
    already name a provider (`nace-ai/drex-max`) are sent as given.
    """
    if "/" in model:
        return model
    if model == "drex-latest":
        raise ValueError(
            "drex-latest is not served through OpenRouter. "
            "Pass an id such as nace-ai/drex-max or nace-ai/drex-v2.0."
        )
    return OPENROUTER_PREFIX + model


def _is_openrouter(base_url: str | None) -> bool:
    host = (urlsplit(str(base_url)).hostname or "") if base_url else ""
    return host == "openrouter.ai" or host.endswith(".openrouter.ai")


def question_to_wire(question: Any) -> dict[str, Any]:
    """A question as Drex reads it: `instructions` is always a string (never null)."""
    if isinstance(question, nace_sdk.Noul | nace_sdk.Choice | nace_sdk.Score):
        wire = question.model_dump(mode="json")
    elif isinstance(question, Mapping):
        wire = dict(question)
    else:
        raise TypeError(f"Unsupported question type: {type(question).__name__}.")
    if wire.get("instructions") is None:
        wire["instructions"] = ""
    return wire


@dataclass
class Usage:
    """Running totals of what a client has spent, counted once per request."""

    requests: int = 0
    input_tokens: int = 0
    image_tokens: int = 0
    evaluation_ms: float = 0.0

    def add(self, input_tokens: int | None, image_tokens: int | None, ms: float | None) -> None:
        # Called from the event loop only, so a plain update is safe.
        self.requests += 1
        self.input_tokens += input_tokens or 0
        self.image_tokens += image_tokens or 0
        self.evaluation_ms += ms or 0.0


@dataclass
class Ruling:
    """One Drex response: the SDK's typed answers plus what to keep with a result."""

    response: nace_sdk.SystemOneResponse
    request_id: str | None

    @property
    def answers(self) -> dict[str, Any]:
        return self.response.answers

    @property
    def metadata(self) -> dict[str, Any]:
        usage = self.response.usage
        return {
            "request_id": self.request_id,
            "model": self.response.model,
            "input_tokens": usage.input_tokens,
            "image_tokens": getattr(usage, "image_tokens", None),
            "evaluation_time_ms": self.response.evaluation_time_ms,
        }


class AsyncDrexClient:
    """Async Drex over `nace-sdk`, with this package's attribution, limits and error mapping."""

    def __init__(
        self,
        *,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float = MIN_TIMEOUT,
        max_retries: int = 2,
        max_in_flight: int = DEFAULT_MAX_IN_FLIGHT,
        extra_headers: Mapping[str, str] | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = (model or os.environ.get("NACE_DEFAULT_DECISION_MODEL") or "").strip() or (
            DEFAULT_MODEL
        )
        base_url = (base_url if base_url is not None else os.environ.get("NACE_BASE_URL")) or None
        base_url = base_url.strip() or None if base_url else None
        self.openrouter = _is_openrouter(base_url)
        if self.openrouter:
            self.model = openrouter_model(self.model)
        # A gateway has no Drex Billing page; any other host (including a customer's own
        # deployment) has its own.
        self._billing = None if self.openrouter else billing_url(base_url)
        self.timeout = max(timeout, MIN_TIMEOUT)
        self.usage = Usage()
        self._max_in_flight = max(1, max_in_flight)
        self._slots: weakref.WeakKeyDictionary[Any, asyncio.Semaphore] = weakref.WeakKeyDictionary()
        self._clients: weakref.WeakKeyDictionary[Any, nace_sdk.AsyncNaceClient] = (
            weakref.WeakKeyDictionary()
        )
        # The key follows the destination: a Nace key is never sent to OpenRouter, nor an
        # OpenRouter key to Nace. (An explicit `api_key` is used as given.)
        if api_key is None and self.openrouter:
            api_key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
            if not api_key:
                raise ValueError(
                    "No OpenRouter API key was provided. Pass api_key or set OPENROUTER_API_KEY "
                    "(NACE_API_KEY is never sent to OpenRouter)."
                )
        headers = {"User-Agent": CLIENT_NAME, **(extra_headers or {})}
        if self.openrouter:
            headers.setdefault("HTTP-Referer", "https://nace.ai")
            headers.setdefault("X-OpenRouter-Title", "Drex for Inspect")
            if base_url:  # OpenRouter documents `.../api/v1`; the SDK appends `/v1/systemone`.
                base_url = base_url.rstrip("/").removesuffix("/v1")
        self._http_client = http_client
        if http_client is not None:
            self._add_hooks(http_client)
        self._sdk_options: dict[str, Any] = {
            "api_key": api_key,
            "base_url": base_url,
            # `timeout=None`: the SDK's default 30 s retry budget would cut short a 60 s
            # request and refuse to wait out a long `retry-after-ms`.
            "retry": nace_sdk.RetryPolicy(max_retries=max_retries, timeout=None),
            "timeout": self.timeout,
            "headers": headers,
            "client_info": CLIENT_NAME,
        }
        self._lock = threading.Lock()
        # Built now so a missing key fails here; the first event loop to ask adopts it.
        self._unbound: nace_sdk.AsyncNaceClient | None = self._new_sdk_client()

    def _add_hooks(self, http: httpx.AsyncClient) -> None:
        http.event_hooks["request"].append(_set_user_agent)  # the SDK sets its own agent
        if self.openrouter:  # nace-sdk adds the Vercel bypass secret from the environment
            http.event_hooks["request"].append(_drop_bypass_secret)

    def _new_sdk_client(self) -> nace_sdk.AsyncNaceClient:
        http = self._http_client
        if http is None:
            http = httpx.AsyncClient(timeout=self.timeout)
            self._add_hooks(http)
        try:
            return nace_sdk.AsyncNaceClient(**self._sdk_options, http_client=http)
        except nace_sdk.NaceError as error:
            raise ValueError(str(error)) from error

    def _sdk_client(self) -> nace_sdk.AsyncNaceClient:
        """The SDK client of the running event loop.

        Pooled connections can't cross loops, so each loop gets its own client (Inspect runs
        a new loop for every `eval()`). A caller-supplied `http_client` is shared by all of
        them, so it stays bound to a single loop.
        """
        loop = asyncio.get_running_loop()
        with self._lock:
            if loop not in self._clients:
                self._clients[loop] = self._unbound or self._new_sdk_client()
                self._unbound = None
            return self._clients[loop]

    def _slot(self) -> asyncio.Semaphore:
        """The in-flight limit of the running event loop (a semaphore can't cross loops)."""
        loop = asyncio.get_running_loop()
        if loop not in self._slots:
            self._slots[loop] = asyncio.Semaphore(self._max_in_flight)
        return self._slots[loop]

    @property
    def provider(self) -> str:
        return "OpenRouter" if self.openrouter else "Drex"

    async def ask(
        self,
        state: Any,
        questions: Mapping[str, Any],
        images: Sequence[Any] | None = None,
        *,
        model: str | None = None,
    ) -> Ruling:
        """Ask every question about one state in one request."""
        model = model or self.model
        if self.openrouter:
            model = openrouter_model(model)
        wire = {name: question_to_wire(q) for name, q in questions.items()}
        if not wire:
            raise ValueError("At least one question is required.")
        entries = [to_drex_image(image) for image in images or ()]
        if entries and is_text_only(model):
            raise DrexImageModelError(
                f"{model} reads text only, so it can't take images. "
                "Use an image model such as model='drex-max'."
            )
        if entries and self.openrouter:
            state, entries = _openrouter_state(state, entries), []
        try:
            async with self._slot():
                result = await self._sdk_client().system_one(
                    state=state, questions=wire, model=model, images=entries or None
                )
        except nace_sdk.NaceError as error:
            raise convert_error(error, self.provider, self._billing) from error
        try:  # the SDK resolves the id from the header, then the body (gateways drop headers)
            request_id: str | None = result.request_id
        except nace_sdk.NaceError:
            request_id = None
        ruling = Ruling(result, request_id)
        usage = result.usage
        self.usage.add(
            usage.input_tokens, getattr(usage, "image_tokens", None), result.evaluation_time_ms
        )
        return ruling


async def _drop_bypass_secret(request: httpx.Request) -> None:
    request.headers.pop("x-vercel-protection-bypass", None)


async def _set_user_agent(request: httpx.Request) -> None:
    request.headers["User-Agent"] = CLIENT_NAME


def _openrouter_state(state: Any, images: list[str | dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenRouter takes images as `image_url` parts of a `state` content array."""
    text = state if isinstance(state, str) else json.dumps(state)
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}]
    for image in images:
        if not isinstance(image, str):
            raise DrexImageError("OpenRouter accepts inline images only, not workspace files.")
        parts.append({"type": "image_url", "image_url": {"url": as_data_url(image)}})
    return parts
