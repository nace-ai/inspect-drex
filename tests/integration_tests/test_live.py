"""Live contract checks against real Drex. Run explicitly:

    NACE_API_KEY=nace_sk_... uv run pytest tests/integration_tests -s -v

These tests send synthetic inputs and make a handful of billed requests. Set NACE_BASE_URL
to run them against a Drex deployment other than the default.
"""

from __future__ import annotations

import base64

import pytest
from inspect_ai.model import ChatMessageUser, ContentImage, ContentText, ModelName, ModelOutput
from inspect_ai.scorer import Target
from inspect_ai.solver import TaskState
from nace_sdk import Choice, Noul, Score

from inspect_drex import (
    AsyncDrexClient,
    Criterion,
    DrexAPIError,
    DrexError,
    DrexImageModelError,
    drex_scorer,
)

pytestmark = pytest.mark.live

STATE = "The invoice for March was charged twice. Please refund one payment."


def sample(content: str | list, output: str) -> TaskState:
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input="x",
        messages=[ChatMessageUser(content=content)],
        output=ModelOutput.from_content("mockllm/model", output),
    )


def png(data: bytes) -> ContentImage:
    return ContentImage(image="data:image/png;base64," + base64.b64encode(data).decode())


async def test_each_question_type_and_a_mixed_batch(live_client, model):
    ruling = await live_client.ask(
        STATE,
        {
            "refund": Noul(instructions="Is the customer asking for a refund?"),
            "team": Choice(
                instructions="Which team?", criteria={"billing": "Payments", "technical": None}
            ),
            "tone": Score(instructions="How polite?", criteria=["rude", "neutral", "polite"]),
        },
        model=model,
    )
    assert ruling.answers["refund"].noul > 0.5
    assert ruling.answers["team"].choice == "billing"
    assert abs(sum(ruling.answers["team"].probabilities.values()) - 1) < 1e-3
    assert 0 <= ruling.answers["tone"].score <= 2
    assert ruling.metadata["request_id"]
    assert ruling.metadata["input_tokens"] > 0


async def test_criteria_only_noul(live_client, model):
    ruling = await live_client.ask(
        STATE, {"q": Noul(criteria={"true": "asks for a refund"})}, model=model
    )
    assert ruling.answers["q"].noul > 0.5


async def test_the_scorer_prefers_a_grounded_answer(model):
    rubric = Criterion.noul("grounded", "Does the output state the reference fact?")
    scorer = drex_scorer(rubric, model=model)
    target = Target("5 business days")
    good = await scorer(sample("Refund timing?", "Refunds arrive within 5 business days."), target)
    bad = await scorer(sample("Refund timing?", "Please contact support."), target)
    assert good.value > bad.value
    assert good.metadata["drex"]["request_id"]
    assert set(good.metadata["drex"]["criteria"]["grounded"]["probabilities"]) == {"yes", "no"}


async def test_one_and_two_images_on_max_and_reversed_order(red, blue):
    scorer = drex_scorer(Criterion.noul("red", "Is Image 1 a solid red picture?"), model="drex-max")
    target = Target("")
    one = await scorer(sample([ContentText(text="Look."), png(red)], "ok"), target)
    forward = await scorer(sample([png(red), png(blue)], "ok"), target)
    reverse = await scorer(sample([png(blue), png(red)], "ok"), target)
    assert one.value > 0.5
    assert forward.value > 0.5
    assert reverse.value < 0.5


async def test_images_on_v2_are_refused_client_side(red):
    scorer = drex_scorer("ok", model="drex-v2.0")
    with pytest.raises(DrexImageModelError):
        await scorer(sample([png(red)], "ok"), Target(""))


async def test_too_many_questions_are_refused_with_the_questions_path(live_client):
    # nace-sdk may refuse an oversized request before sending it; either way the message
    # names the questions, and a 422 from Drex carries the `questions` issue path.
    questions = {f"q{i}": Noul(instructions="Is this text?") for i in range(600)}
    with pytest.raises(DrexError) as raised:
        await live_client.ask("hello", questions, model="drex-max")
    if isinstance(raised.value, DrexAPIError):
        assert raised.value.status == 422
        assert any(issue["path"] == "questions" for issue in raised.value.issues)
    else:
        assert "questions" in str(raised.value).lower()


async def test_a_bad_key_is_a_401():
    with pytest.raises(DrexAPIError) as raised:
        await AsyncDrexClient(api_key="nace_sk_invalid").ask("x", {"q": Noul(instructions="?")})
    assert raised.value.status == 401
