"""Live checks through OpenRouter. Run explicitly:

    OPENROUTER_API_KEY=sk-or-... uv run pytest tests/integration_tests/test_live_openrouter.py -s -v

They need only an OpenRouter key and synthetic inputs. `NACE_API_KEY` is neither needed nor
sent: the tests set it to a sentinel and check that no request carries it.
"""

from __future__ import annotations

import base64

import httpx
import pytest
from nace_sdk import Noul

from inspect_drex import AsyncDrexClient

pytestmark = pytest.mark.openrouter

BASE_URL = "https://openrouter.ai/api/v1"
SENTINEL = "nace_sk_must_never_reach_openrouter"
STATE = "The invoice for March was charged twice. Please refund one payment."
ASK = {"refund": Noul(instructions="Is the customer asking for a refund?")}


@pytest.fixture
def sent() -> list[httpx.Request]:
    return []


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch, sent: list[httpx.Request]) -> AsyncDrexClient:
    monkeypatch.setenv("NACE_API_KEY", SENTINEL)

    async def record(request: httpx.Request) -> None:
        sent.append(request)

    http = httpx.AsyncClient(event_hooks={"request": [record]})
    return AsyncDrexClient(base_url=BASE_URL, http_client=http)


def no_nace_key(sent: list[httpx.Request]) -> None:
    assert sent
    for request in sent:
        assert SENTINEL not in "".join(f"{k}={v}" for k, v in request.headers.items())
        assert SENTINEL not in request.content.decode(errors="ignore")


async def test_the_default_model_works_with_a_bare_id_prefixed_for_openrouter(client, sent):
    assert client.model == "nace-ai/drex-max"
    ruling = await client.ask(STATE, ASK)
    assert ruling.answers["refund"].noul > 0.5
    assert ruling.metadata["model"].startswith("nace-ai/drex-max")
    no_nace_key(sent)


@pytest.mark.parametrize("model", ["nace-ai/drex-v2.0", "nace-ai/drex-max"])
async def test_each_openrouter_model_id(client, sent, model):
    ruling = await client.ask(STATE, ASK, model=model)
    assert ruling.answers["refund"].noul > 0.5
    assert ruling.metadata["input_tokens"] > 0  # OpenRouter does not return a Drex request id
    no_nace_key(sent)


async def test_an_inline_image_on_drex_max(client, sent, red):
    url = "data:image/png;base64," + base64.b64encode(red).decode()
    ruling = await client.ask(
        "Image 1 is attached.",
        {"red": Noul(instructions="Is Image 1 predominantly red?")},
        [url],
        model="nace-ai/drex-max",
    )
    assert ruling.answers["red"].noul > 0.5
    no_nace_key(sent)


async def test_a_nace_key_is_not_needed(monkeypatch):
    monkeypatch.delenv("NACE_API_KEY", raising=False)
    ruling = await AsyncDrexClient(base_url=BASE_URL).ask(STATE, ASK)
    assert ruling.answers["refund"].noul > 0.5
