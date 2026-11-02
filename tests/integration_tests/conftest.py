from __future__ import annotations

import os
from pathlib import Path

import pytest

from inspect_drex import AsyncDrexClient

FIXTURES = Path(__file__).parent.parent / "fixtures"


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    skip_direct = pytest.mark.skip(reason="live checks need NACE_API_KEY")
    skip_openrouter = pytest.mark.skip(reason="OpenRouter checks need OPENROUTER_API_KEY")
    for item in items:
        if "openrouter" in item.keywords:
            if not os.environ.get("OPENROUTER_API_KEY"):
                item.add_marker(skip_openrouter)
        elif "live" in item.keywords and not os.environ.get("NACE_API_KEY"):
            item.add_marker(skip_direct)


@pytest.fixture(params=["drex-max", "drex-v2.0"])
def model(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def live_client() -> AsyncDrexClient:
    return AsyncDrexClient()


@pytest.fixture
def red() -> bytes:
    return (FIXTURES / "red.png").read_bytes()


@pytest.fixture
def blue() -> bytes:
    return (FIXTURES / "blue.png").read_bytes()
