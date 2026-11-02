from __future__ import annotations

import asyncio
import time
import types
from pathlib import Path

import httpx
import pytest
from nace_sdk._core import retry

from inspect_drex import AsyncDrexClient
from tests.fakes import FakeDrex

FIXTURES = Path(__file__).parent.parent / "fixtures"


@pytest.fixture(autouse=True)
def _no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NACE_API_KEY",
        "OPENROUTER_API_KEY",
        "NACE_BASE_URL",
        "NACE_DEFAULT_DECISION_MODEL",
        "DREX_API_KEY",
        "DREX_BASE_URL",
        "DREX_DEFAULT_MODEL",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Record the SDK's retry waits instead of sleeping."""
    recorded: list[float] = []

    async def fake_sleep(delay: float) -> None:
        recorded.append(delay)

    fake_time = types.SimpleNamespace(
        **{k: getattr(time, k) for k in dir(time) if not k.startswith("_")}
    )
    fake_time.sleep = recorded.append
    fake_asyncio = types.SimpleNamespace(
        **{k: getattr(asyncio, k) for k in dir(asyncio) if not k.startswith("_")}
    )
    fake_asyncio.sleep = fake_sleep
    monkeypatch.setattr(retry, "time", fake_time)
    monkeypatch.setattr(retry, "asyncio", fake_asyncio)
    return recorded


@pytest.fixture
def drex() -> FakeDrex:
    return FakeDrex()


@pytest.fixture
def make_client(drex: FakeDrex, sleeps: list[float]):
    def make(**kwargs: object) -> AsyncDrexClient:
        options: dict[str, object] = {
            "api_key": "nace_sk_test",
            "base_url": "https://drex.test",
            "http_client": httpx.AsyncClient(transport=drex.async_transport()),
        }
        options.update(kwargs)
        return AsyncDrexClient(**options)  # type: ignore[arg-type]

    return make


@pytest.fixture
def client(make_client) -> AsyncDrexClient:
    return make_client()


@pytest.fixture
def red_png() -> bytes:
    return (FIXTURES / "red.png").read_bytes()


@pytest.fixture
def blue_png() -> bytes:
    return (FIXTURES / "blue.png").read_bytes()
