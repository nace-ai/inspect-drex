from __future__ import annotations

import asyncio
import base64

import httpx
import pytest
from inspect_ai.model import ContentImage
from nace_sdk import Noul

from inspect_drex import AsyncDrexClient, DrexAPIError, DrexConnectionError, DrexImageError
from inspect_drex._images import as_data_url, to_drex_image
from tests.fakes import error


def test_image_forms_become_images_entries(red_png, tmp_path):
    b64 = base64.b64encode(red_png).decode()
    path = tmp_path / "red.png"
    path.write_bytes(red_png)
    assert to_drex_image("data:image/png;base64,AAAA") == "data:image/png;base64,AAAA"
    assert to_drex_image(b64) == b64
    assert to_drex_image(ContentImage(image="data:image/png;base64," + b64)).startswith(
        "data:image/png;base64,"
    )
    assert isinstance(to_drex_image(red_png), str)
    assert isinstance(to_drex_image(str(path)), str)
    assert isinstance(to_drex_image(path), str)


@pytest.mark.parametrize("bad", ["not an image path", "", 42, {"type": "other"}])
def test_unreadable_images_are_refused(bad):
    with pytest.raises(DrexImageError):
        to_drex_image(bad)


def test_data_url_type_is_sniffed_from_the_bytes(red_png):
    assert as_data_url(base64.b64encode(red_png).decode()).startswith("data:image/png;")
    jpeg = base64.b64encode(b"\xff\xd8\xff\xe0" + b"0" * 20).decode()
    assert as_data_url(jpeg).startswith("data:image/jpeg;")
    webp = base64.b64encode(b"RIFF\x00\x00\x00\x00WEBPVP8 ").decode()
    assert as_data_url(webp).startswith("data:image/webp;")
    assert as_data_url("data:image/png;base64,AA") == "data:image/png;base64,AA"


async def test_openrouter_text_errors_are_made_readable(make_client, drex):
    client = make_client(base_url="https://openrouter.ai/api/v1")
    body = '[{"path": ["questions", "q"], "message": "bad question"}]'
    drex.replies.append((422, {}, {"error": {"message": f"HTTP 422: {body}"}}))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert "OpenRouter returned 422" in str(raised.value)
    assert "questions.q: bad question" in str(raised.value)


async def test_openrouter_detail_wrapper_is_unwrapped(make_client, drex):
    client = make_client(base_url="https://openrouter.ai/api/v1")
    drex.replies.append((400, {}, {"error": {"message": 'HTTP 400: {"detail": "nope"}'}}))
    with pytest.raises(DrexAPIError, match="nope"):
        await client.ask("x", {"q": Noul(instructions="?")})


async def test_connection_failures_are_reported_as_such(sleeps):
    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down", request=request)

    client = AsyncDrexClient(
        api_key="k",
        base_url="https://drex.test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(boom)),
    )
    with pytest.raises(DrexConnectionError):
        await client.ask("x", {"q": Noul(instructions="?")})


def test_missing_api_key_is_a_value_error():
    with pytest.raises(ValueError, match="API key"):
        AsyncDrexClient()


def test_blank_base_url_means_default(monkeypatch):
    monkeypatch.setenv("NACE_BASE_URL", "  ")
    assert AsyncDrexClient(api_key="k").provider == "Drex"


async def test_unsupported_question_type_is_a_type_error(client):
    with pytest.raises(TypeError):
        await client.ask("x", {"q": object()})


async def test_server_errors_are_api_errors(client, drex, sleeps):
    drex.replies.extend(error(500, "boom", kind="api_error") for _ in range(3))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert raised.value.status == 500


async def test_timeouts_are_connection_errors(sleeps):
    def slow(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=request)

    client = AsyncDrexClient(
        api_key="k",
        base_url="https://drex.test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(slow)),
    )
    with pytest.raises(DrexConnectionError):
        await client.ask("x", {"q": Noul(instructions="?")})


async def test_each_event_loop_gets_its_own_sdk_client():
    client = AsyncDrexClient(api_key="k", base_url="https://drex.test")
    first = client._sdk_client()
    assert client._sdk_client() is first

    def other_loop():
        async def get():
            return client._sdk_client()

        return asyncio.run(get())

    assert await asyncio.to_thread(other_loop) is not first
