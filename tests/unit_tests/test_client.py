"""The contract cases: models, questions, images, errors, retries, backends, attribution."""

from __future__ import annotations

import asyncio
import base64

import httpx
import pytest
from nace_sdk import Choice, Noul, Score

from inspect_drex import (
    BILLING_URL,
    CLIENT_NAME,
    AsyncDrexClient,
    DrexAPIError,
    DrexImageError,
    DrexImageModelError,
    billing_url,
    is_text_only,
)
from tests.fakes import error, ok_body

MODELS = ["drex-v2.0", "drex-max"]
QUESTIONS = {
    "refund": Noul(instructions="Is this a refund request?"),
    "team": Choice(instructions="Which team?", criteria={"billing": "Payments", "tech": None}),
    "tone": Score(instructions="How polite?", criteria=["rude", "neutral", "polite"]),
}


async def test_default_model_is_drex_max(client, drex):
    await client.ask("hello", {"q": Noul(instructions="Hi?")})
    assert drex.bodies[0]["model"] == "drex-max"


async def test_model_comes_from_the_environment(monkeypatch, make_client, drex):
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v2.0")
    await make_client().ask("hello", {"q": Noul(instructions="Hi?")})
    assert drex.bodies[0]["model"] == "drex-v2.0"


@pytest.mark.parametrize("model", [*MODELS, "drex-v9.9"])
async def test_any_model_id_is_sent_as_given(client, drex, model):
    await client.ask("hello", {"q": Noul(instructions="Hi?")}, model=model)
    assert drex.bodies[0]["model"] == model


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    ("name", "kind"), [("refund", "noul"), ("team", "choice"), ("tone", "score")]
)
async def test_single_question_of_each_type(client, drex, model, name, kind):
    ruling = await client.ask("I want my money back", {name: QUESTIONS[name]}, model=model)
    assert list(drex.bodies[0]["questions"]) == [name]
    assert ruling.answers[name].type == kind
    assert ruling.metadata["request_id"] == "req_test"


@pytest.mark.parametrize("model", MODELS)
async def test_a_mixed_batch_is_one_request_with_full_probabilities(client, drex, model):
    ruling = await client.ask("I want my money back", QUESTIONS, model=model)
    assert len(drex.bodies) == 1
    assert set(drex.bodies[0]["questions"]) == set(QUESTIONS)
    assert ruling.answers["team"].probabilities == pytest.approx({"billing": 0.8, "tech": 0.2})
    assert set(ruling.answers["tone"].probabilities) == {0, 1, 2}
    assert ruling.metadata["input_tokens"] == 120
    assert ruling.metadata["evaluation_time_ms"] == 81.5


async def test_usage_is_counted_once_per_request_not_per_question(client):
    await client.ask("x", QUESTIONS)
    assert client.usage.requests == 1
    assert client.usage.input_tokens == 120


async def test_criteria_only_noul_sends_empty_instructions(client, drex):
    await client.ask("x", {"q": Noul(criteria={"true": "mentions a refund"})})
    question = drex.bodies[0]["questions"]["q"]
    assert question["instructions"] == ""
    assert question["criteria"] == {"true": "mentions a refund"}


async def test_raw_question_with_null_instructions_is_sent_as_a_string(client, drex):
    await client.ask(
        "x", {"q": {"type": "noul", "instructions": None, "criteria": {"true": "yes"}}}
    )
    assert drex.bodies[0]["questions"]["q"]["instructions"] == ""


async def test_null_choice_description_is_kept(client, drex):
    await client.ask("x", {"q": QUESTIONS["team"]})
    assert drex.bodies[0]["questions"]["q"]["criteria"] == {"billing": "Payments", "tech": None}


async def test_json_state_is_sent_as_json(client, drex):
    await client.ask({"input": "a", "output": ["b"]}, {"q": Noul(instructions="ok?")})
    assert drex.bodies[0]["state"] == {"input": "a", "output": ["b"]}


async def test_one_image_on_max(client, drex, red_png):
    await client.ask(
        "Image 1 is?", {"q": Noul(instructions="Is Image 1 red?")}, [red_png], model="drex-max"
    )
    assert len(drex.bodies[0]["images"]) == 1


async def test_two_images_keep_their_order_and_reversing_changes_it(
    client, drex, red_png, blue_png
):
    questions = {"q": Noul(instructions="Is Image 1 red?")}
    await client.ask("x", questions, [red_png, blue_png], model="drex-max")
    await client.ask("x", questions, [blue_png, red_png], model="drex-max")
    forward, reverse = (
        [base64.b64decode(i.split(",", 1)[-1]) for i in body["images"]] for body in drex.bodies
    )
    assert forward == [red_png, blue_png]
    assert reverse == [blue_png, red_png]


@pytest.mark.parametrize("model", ["drex-v2.0", "drex-v1.5", "drex-latest"])
async def test_images_on_text_only_models_are_refused_before_sending(client, drex, red_png, model):
    with pytest.raises(DrexImageModelError, match="reads text only"):
        await client.ask("x", {"q": Noul(instructions="?")}, [red_png], model=model)
    assert drex.bodies == []


async def test_unknown_models_are_not_assumed_text_only():
    assert not is_text_only("drex-max")
    assert not is_text_only("drex-v3.0")
    assert is_text_only("nace-ai/drex-v2.0")


async def test_remote_image_urls_are_refused(client, drex):
    with pytest.raises(DrexImageError, match="doesn't fetch image URLs"):
        await client.ask("x", {"q": Noul(instructions="?")}, ["https://example.com/a.png"])
    assert drex.bodies == []


async def test_workspace_file_image_is_passed_through(client, drex):
    ref = {"type": "workspace_file", "workspace_id": "w", "file_id": "f"}
    await client.ask("x", {"q": Noul(instructions="?")}, [ref])
    assert drex.bodies[0]["images"] == [ref]


async def test_too_many_questions_surface_drexs_422_path_and_message(client, drex):
    questions = {f"q{i}": Noul(instructions="?") for i in range(3)}
    drex.replies.append(
        error(
            422,
            "too many questions",
            issues=[{"path": "questions", "message": "too many questions"}],
        )
    )
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", questions, model="drex-max")
    assert raised.value.status == 422
    assert raised.value.issues == [{"path": "questions", "message": "too many questions"}]
    assert "questions: too many questions" in str(raised.value)


async def test_422_lists_each_issue(client, drex):
    drex.replies.append(
        error(
            422,
            "invalid request",
            issues=[
                {"path": "questions.a.type", "message": "unknown type"},
                {"path": "state", "message": "required"},
            ],
        )
    )
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"a": Noul(instructions="?")})
    text = str(raised.value)
    assert "questions.a.type: unknown type" in text
    assert "state: required" in text


async def test_401_is_an_authentication_error(client, drex):
    drex.replies.append(error(401, "Invalid API key", kind="authentication_error"))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert raised.value.status == 401
    assert "401" in str(raised.value)


async def test_402_points_to_billing(client, drex):
    drex.replies.append(error(402, "no available funds", kind="insufficient_credit_error"))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert "https://drex.test/dashboard/billing" in str(raised.value)
    assert BILLING_URL not in str(raised.value)


def test_billing_url_follows_the_configured_host():
    assert billing_url(None) == BILLING_URL
    assert (
        billing_url("https://drex.customer.example")
        == "https://drex.customer.example/dashboard/billing"
    )
    assert (
        billing_url("https://drex.customer.example/")
        == "https://drex.customer.example/dashboard/billing"
    )
    assert (
        billing_url("https://drex.customer.example/v1")
        == "https://drex.customer.example/dashboard/billing"
    )


async def test_402_on_openrouter_has_no_drex_billing_link(make_client, drex):
    drex.replies.append(error(402, "insufficient credits", kind="insufficient_credit_error"))
    client = make_client(base_url="https://openrouter.ai/api/v1")
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")}, model="nace-ai/drex-max")
    assert "nace.ai/dashboard" not in str(raised.value)
    assert "OpenRouter account" in str(raised.value)


async def test_a_customers_own_deployment_works_with_the_nace_key(monkeypatch, drex):
    # Drex on a customer's infrastructure: any host is normal and takes the Nace key.
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_customer")
    monkeypatch.setenv("NACE_BASE_URL", "https://drex.customer.example")
    http = httpx.AsyncClient(transport=drex.async_transport())
    client = AsyncDrexClient(http_client=http)
    assert not client.openrouter
    await client.ask("x", {"q": Noul(instructions="?")})
    request, body = drex.requests[0], drex.bodies[0]
    assert str(request.url) == "https://drex.customer.example/v1/systemone"
    assert request.headers["authorization"] == "Bearer nace_sk_customer"
    assert body["model"] == "drex-max"
    assert "http-referer" not in request.headers
    assert isinstance(body["questions"]["q"]["instructions"], str)
    drex.replies.append(error(402, "no available funds", kind="insufficient_credit_error"))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert "https://drex.customer.example/dashboard/billing" in str(raised.value)
    assert "console.nace.ai" not in str(raised.value)


async def test_a_customer_host_takes_images_as_images_not_openrouter_parts(client, drex, red_png):
    custom = AsyncDrexClient(
        api_key="k",
        base_url="https://drex.customer.example",
        http_client=httpx.AsyncClient(transport=drex.async_transport()),
    )
    await custom.ask("x", {"q": Noul(instructions="?")}, [red_png])
    assert isinstance(drex.bodies[0]["state"], str)
    assert len(drex.bodies[0]["images"]) == 1


@pytest.mark.parametrize(("status", "kind"), [(429, "rate_limit_error"), (529, "overloaded_error")])
async def test_429_and_529_are_retried_after_retry_after_ms(client, drex, sleeps, status, kind):
    drex.replies.append(error(status, "slow down", kind=kind, headers={"retry-after-ms": "1500"}))
    ruling = await client.ask("x", {"q": Noul(instructions="?")})
    assert len(drex.bodies) == 2
    assert sleeps == [pytest.approx(1.5)]
    assert ruling.answers["q"].type == "noul"


async def test_long_retry_after_is_honored_beyond_the_sdk_default_budget(client, drex, sleeps):
    drex.replies.append(
        error(429, "slow", kind="rate_limit_error", headers={"retry-after-ms": "45000"})
    )
    ruling = await client.ask("x", {"q": Noul(instructions="?")})
    assert len(drex.bodies) == 2
    assert sleeps == [pytest.approx(45.0)]
    assert ruling.answers["q"].type == "noul"


async def test_the_request_goes_out_with_a_timeout_of_at_least_a_minute(make_client, drex):
    await make_client(timeout=5).ask("x", {"q": Noul(instructions="?")})
    assert drex.requests[0].extensions["timeout"]["read"] >= 60


async def test_image_tokens_are_in_the_metadata_and_the_usage_total(client, drex):
    def reply(body):
        payload = ok_body(body)
        payload["usage"]["image_tokens"] = 33
        return 200, {}, payload

    drex.replies.append(reply)
    ruling = await client.ask("x", {"q": Noul(instructions="?")})
    assert ruling.metadata["image_tokens"] == 33
    assert client.usage.image_tokens == 33


async def test_request_id_falls_back_to_the_body_when_the_header_is_missing(client, drex):
    drex.replies.append(
        lambda body: (200, {"x-request-id": ""}, ok_body(body) | {"request_id": "req_body"})
    )
    ruling = await client.ask("x", {"q": Noul(instructions="?")})
    assert ruling.metadata["request_id"] == "req_body"


async def test_exhausted_retries_raise_an_execution_error_not_a_score(make_client, drex, sleeps):
    client = make_client(max_retries=1)
    drex.replies.extend(error(429, "slow", kind="rate_limit_error") for _ in range(2))
    with pytest.raises(DrexAPIError) as raised:
        await client.ask("x", {"q": Noul(instructions="?")})
    assert raised.value.status == 429
    assert len(drex.bodies) == 2


async def test_attribution_headers_are_sent_on_every_request(client, drex):
    await client.ask("x", {"q": Noul(instructions="?")})
    await client.ask("y", {"q": Noul(instructions="?")})
    for request in drex.requests:
        assert request.headers["x-nace-client"] == CLIENT_NAME
        assert request.headers["user-agent"] == CLIENT_NAME
    assert CLIENT_NAME.startswith("inspect-drex/")


async def test_timeout_is_at_least_60_seconds(make_client):
    assert make_client(timeout=5).timeout == 60.0


async def test_openrouter_shapes_the_request(make_client, drex, red_png):
    client = make_client(base_url="https://openrouter.ai/api/v1")
    await client.ask(
        "describe",
        {"q": {"type": "noul", "criteria": {"true": "red"}}},
        [red_png],
        model="nace-ai/drex-max",
    )
    body, request = drex.bodies[0], drex.requests[0]
    assert str(request.url) == "https://openrouter.ai/api/v1/systemone"
    assert body["model"] == "nace-ai/drex-max"
    assert "images" not in body
    assert body["questions"]["q"]["instructions"] == ""
    assert [part["type"] for part in body["state"]] == ["text", "image_url"]
    assert body["state"][1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert request.headers["http-referer"] == "https://nace.ai"
    assert request.headers["x-openrouter-title"] == "Drex for Inspect"
    assert request.headers["x-nace-client"] == CLIENT_NAME


@pytest.mark.parametrize(
    "base_url",
    ["https://openrouter.ai/api/v1", "https://openrouter.ai/api/v1/", "https://openrouter.ai/api"],
)
async def test_openrouter_base_url_never_doubles_the_v1(make_client, drex, base_url):
    await make_client(base_url=base_url).ask(
        "x", {"q": Noul(instructions="?")}, model="nace-ai/drex-max"
    )
    assert str(drex.requests[0].url) == "https://openrouter.ai/api/v1/systemone"


def test_only_openrouter_hosts_get_openrouter_shaping():
    for url in ("https://evilopenrouter.ai/v1", "https://openrouter.ai.evil.test/v1"):
        assert not AsyncDrexClient(api_key="k", base_url=url).openrouter
    assert AsyncDrexClient(api_key="k", base_url="https://api.openrouter.ai/v1").openrouter


async def test_openrouter_prefixes_bare_ids_including_the_default(make_client, drex):
    client = make_client(base_url="https://openrouter.ai/api/v1")
    assert client.model == "nace-ai/drex-max"
    await client.ask("x", {"q": Noul(instructions="?")})
    await client.ask("x", {"q": Noul(instructions="?")}, model="drex-v2.0")
    await client.ask("x", {"q": Noul(instructions="?")}, model="nace-ai/drex-v2.0")
    await client.ask("x", {"q": Noul(instructions="?")}, model="other/model")
    assert [body["model"] for body in drex.bodies] == [
        "nace-ai/drex-max",
        "nace-ai/drex-v2.0",
        "nace-ai/drex-v2.0",
        "other/model",
    ]


async def test_openrouter_refuses_drex_latest_instead_of_guessing(make_client, drex):
    client = make_client(base_url="https://openrouter.ai/api/v1")
    with pytest.raises(ValueError, match="drex-latest"):
        await client.ask("x", {"q": Noul(instructions="?")}, model="drex-latest")
    with pytest.raises(ValueError, match="drex-latest"):
        make_client(base_url="https://openrouter.ai/api/v1", model="drex-latest")
    assert not drex.bodies


async def test_the_direct_backend_never_gets_the_prefix(client, drex):
    await client.ask("x", {"q": Noul(instructions="?")})
    assert drex.bodies[0]["model"] == "drex-max"


async def test_a_nace_key_is_never_sent_to_openrouter(monkeypatch):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_secret")
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        AsyncDrexClient(base_url="https://openrouter.ai/api/v1")


async def test_the_key_and_bypass_secret_follow_the_host(monkeypatch, drex):
    monkeypatch.setenv("NACE_API_KEY", "nace_sk_secret")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("NACE_VERCEL_PROTECTION_BYPASS", "bypass-secret")
    sent = {}
    for name, url in (
        ("openrouter", "https://openrouter.ai/api/v1"),
        ("nace", "https://drex.test"),
    ):
        http = httpx.AsyncClient(transport=drex.async_transport())
        await AsyncDrexClient(base_url=url, http_client=http).ask(
            "x", {"q": Noul(instructions="?")}, model="m"
        )
        sent[name] = drex.requests[-1].headers
    assert sent["openrouter"]["authorization"] == "Bearer sk-or-secret"
    assert "x-vercel-protection-bypass" not in sent["openrouter"]
    assert sent["nace"]["authorization"] == "Bearer nace_sk_secret"
    assert sent["nace"]["x-vercel-protection-bypass"] == "bypass-secret"


async def test_the_default_http_client_carries_attribution_and_a_long_timeout(monkeypatch, drex):
    real = httpx.AsyncClient
    monkeypatch.setattr(
        "inspect_drex._client.httpx.AsyncClient",
        lambda **kw: real(transport=drex.async_transport(), **kw),
    )
    client = AsyncDrexClient(api_key="k", base_url="https://drex.test", timeout=5)
    await client.ask("x", {"q": Noul(instructions="?")})
    request = drex.requests[0]
    assert request.headers["user-agent"] == CLIENT_NAME
    assert request.headers["x-nace-client"] == CLIENT_NAME
    assert request.extensions["timeout"]["read"] >= 60


async def test_direct_backend_sends_no_openrouter_headers(client, drex):
    await client.ask("x", {"q": Noul(instructions="?")})
    assert "http-referer" not in drex.requests[0].headers
    assert "x-openrouter-title" not in drex.requests[0].headers
    assert str(drex.requests[0].url) == "https://drex.test/v1/systemone"


async def test_at_most_8_requests_are_in_flight(make_client, drex):
    drex.delay = 0.02
    client = make_client()
    await asyncio.gather(*(client.ask("x", {"q": Noul(instructions="?")}) for _ in range(40)))
    assert drex.peak_in_flight == 8
