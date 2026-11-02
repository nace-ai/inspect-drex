"""The scorer against a fake Drex: what it sends, what it scores, how errors surface."""

from __future__ import annotations

import base64
import math

import pytest
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    ContentImage,
    ContentText,
    ModelName,
    ModelOutput,
)
from inspect_ai.scorer import CORRECT, INCORRECT, Target
from inspect_ai.solver import TaskState, generate
from nace_sdk import NoulAnswer, ScoreAnswer

from inspect_drex import (
    CLIENT_NAME,
    Criterion,
    DrexAPIError,
    DrexError,
    DrexImageModelError,
    drex_scorer,
)
from tests.fakes import error, ok_body

NOUL = {"type": "noul", "noul": 0.9}


def task_state(
    text: str = "How do I reset my password?",
    output: str = "Click 'Forgot password'.",
    *,
    content: list | None = None,
) -> TaskState:
    user = ChatMessageUser(content=content if content is not None else text)
    return TaskState(
        model=ModelName("mockllm/model"),
        sample_id=1,
        epoch=1,
        input=text,
        messages=[ChatMessageSystem(content="You are a support agent."), user],
        output=ModelOutput.from_content("mockllm/model", output),
    )


def png_url(data: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(data).decode()


async def judge(scorer, state=None, target="Reset via email."):
    return await scorer(state or task_state(), Target(target))


async def test_a_rubric_string_is_one_yes_no_criterion_judged_with_input_output_reference(
    client, drex
):
    score = await judge(drex_scorer("The answer is helpful.", client=client))
    body = drex.bodies[0]
    assert body["model"] == "drex-max"
    assert list(body["questions"]) == ["criterion"]
    assert body["questions"]["criterion"]["type"] == "noul"
    assert body["questions"]["criterion"]["instructions"] == "The answer is helpful."
    assert body["state"] == {
        "input": "How do I reset my password?",
        "output": "Click 'Forgot password'.",
        "reference": "Reset via email.",
    }
    assert score.value == pytest.approx(0.9)
    assert score.answer == "Click 'Forgot password'."
    meta = score.metadata["drex"]
    assert meta["request_id"] == "req_test"
    assert meta["model"] == "drex-max"
    assert meta["input_tokens"] == 120
    assert meta["evaluation_time_ms"] == 81.5
    assert meta["criteria"]["criterion"]["probabilities"] == pytest.approx({"yes": 0.9, "no": 0.1})


async def test_no_target_means_no_reference(client, drex):
    await judge(drex_scorer("ok", client=client), target="")
    assert "reference" not in drex.bodies[0]["state"]


async def test_include_target_false_hides_the_reference(client, drex):
    await judge(drex_scorer("ok", include_target=False, client=client))
    assert "reference" not in drex.bodies[0]["state"]


async def test_a_defect_question_scores_the_chance_it_does_not_hold(client):
    rubric = Criterion.noul("toxic", "The reply is rude.", want=False)
    score = await judge(drex_scorer(rubric, client=client))
    assert score.value == pytest.approx(0.1)
    detail = score.metadata["drex"]["criteria"]["toxic"]
    assert detail["passed"] is False
    assert detail["probabilities"]["yes"] == pytest.approx(0.9)  # raw answer stays visible


async def test_a_mixed_batch_is_one_request_and_keeps_every_probability(client, drex):
    rubric = [
        Criterion.noul("helpful", "The answer is helpful."),
        Criterion.choice(
            "team",
            "Which team owns this?",
            {"billing": "Payments", "tech": None},
            good="tech",
        ),
        Criterion.score("tone", "How polite?", ["rude", "neutral", "polite"]),
    ]
    score = await judge(drex_scorer(rubric, client=client))
    assert len(drex.bodies) == 1
    assert set(drex.bodies[0]["questions"]) == {"helpful", "team", "tone"}
    criteria = score.metadata["drex"]["criteria"]
    assert criteria["team"]["pass_score"] == pytest.approx(0.2)
    assert criteria["team"]["probabilities"] == pytest.approx({"billing": 0.8, "tech": 0.2})
    assert criteria["tone"]["probabilities"] == pytest.approx({"0": 1.0, "1": 0.0, "2": 0.0})
    assert score.value == pytest.approx((0.9 + 0.2 + 0.0) / 3)
    assert score.metadata["drex"]["input_tokens"] == 120  # counted once, not per question


async def test_min_aggregate_takes_the_weakest_criterion(client):
    rubric = [Criterion.noul("a", "?"), Criterion.score("b", "?", ["bad", "good"])]
    score = await judge(drex_scorer(rubric, aggregate="min", client=client))
    assert score.value == pytest.approx(0.0)


async def test_weights_change_the_mean(client):
    rubric = [Criterion.noul("a", "?", weight=3), Criterion.score("b", "?", ["bad", "good"])]
    score = await judge(drex_scorer(rubric, client=client))
    assert score.value == pytest.approx((0.9 * 3 + 0.0) / 4)


async def test_grade_output_uses_each_criterions_own_threshold(client):
    passing = Criterion.noul("a", "?", threshold=0.8)
    failing = Criterion.noul("a", "?", threshold=0.95)
    assert (await judge(drex_scorer(passing, output="grade", client=client))).value == CORRECT
    graded = await judge(drex_scorer(failing, output="grade", client=client))
    assert graded.value == INCORRECT
    assert graded.metadata["drex"]["pass_score"] == pytest.approx(0.9)  # probability still kept
    assert "failed" in graded.explanation


async def test_model_is_passed_through_and_can_come_from_the_environment(
    monkeypatch, make_client, drex
):
    await judge(drex_scorer("ok", model="drex-v2.0", client=make_client()))
    monkeypatch.setenv("NACE_DEFAULT_DECISION_MODEL", "drex-v9.9")
    await judge(drex_scorer("ok", client=make_client()))
    assert [b["model"] for b in drex.bodies] == ["drex-v2.0", "drex-v9.9"]


async def test_one_image_in_the_user_message_goes_to_drex_max(client, drex, red_png):
    content = [ContentText(text="Describe this."), ContentImage(image=png_url(red_png))]
    await judge(drex_scorer("Image 1 is red.", client=client), task_state(content=content))
    body = drex.bodies[0]
    assert len(body["images"]) == 1
    assert body["state"]["input"] == "Describe this.\n[Image 1]"


async def test_two_images_keep_their_order_and_the_reverse_changes_it(
    client, drex, red_png, blue_png
):
    def content(first: bytes, second: bytes) -> list:
        return [
            ContentImage(image=png_url(first)),
            ContentText(text="then"),
            ContentImage(image=png_url(second)),
        ]

    scorer = drex_scorer("Image 1 is red.", client=client)
    await judge(scorer, task_state(content=content(red_png, blue_png)))
    await judge(scorer, task_state(content=content(blue_png, red_png)))
    forward, reverse = (
        [base64.b64decode(i.split(",", 1)[-1]) for i in body["images"]] for body in drex.bodies
    )
    assert forward == [red_png, blue_png]
    assert reverse == [blue_png, red_png]
    assert drex.bodies[0]["state"]["input"] == "[Image 1]\nthen\n[Image 2]"


async def test_images_with_a_text_only_model_are_refused_before_sending(client, drex, red_png):
    content = [ContentImage(image=png_url(red_png))]
    scorer = drex_scorer("ok", model="drex-v2.0", client=client)
    with pytest.raises(DrexImageModelError, match="reads text only"):
        await judge(scorer, task_state(content=content))
    assert drex.bodies == []


async def test_a_custom_state_and_images_replace_the_default(client, drex, blue_png):
    scorer = drex_scorer(
        "ok",
        state=lambda s, t: {"answer": s.output.completion},
        images=lambda s, t: [blue_png],
        client=client,
    )
    await judge(scorer)
    assert drex.bodies[0]["state"] == {"answer": "Click 'Forgot password'."}
    assert len(drex.bodies[0]["images"]) == 1


def test_images_need_a_custom_state(client):
    with pytest.raises(ValueError, match="custom `state`"):
        drex_scorer("ok", images=lambda s, t: [], client=client)


def test_criteria_must_be_named_uniquely_and_not_empty(client):
    with pytest.raises(ValueError, match="unique"):
        drex_scorer([Criterion.noul("a", "?"), Criterion.noul("a", "?")], client=client)
    with pytest.raises(ValueError, match="At least one"):
        drex_scorer([], client=client)


@pytest.mark.parametrize("weights", [(0, 0), (2, -1), (1, math.nan), (1, math.inf)])
def test_weights_must_be_finite_non_negative_with_a_positive_total(client, weights):
    rubric = [Criterion.noul(f"c{i}", "?", weight=w) for i, w in enumerate(weights)]
    with pytest.raises(ValueError, match="weights"):
        drex_scorer(rubric, client=client)


async def test_instructions_are_always_strings_for_a_criteria_only_noul(client, drex):
    rubric = Criterion.noul("refund", "", criteria={"true": "asks for a refund"})
    await judge(drex_scorer(rubric, client=client))
    question = drex.bodies[0]["questions"]["refund"]
    assert question["instructions"] == ""
    assert question["criteria"] == {"true": "asks for a refund"}


async def test_a_null_choice_description_is_kept(client, drex):
    rubric = Criterion.choice("team", "Which?", {"billing": "Payments", "tech": None}, good="tech")
    await judge(drex_scorer(rubric, client=client))
    assert drex.bodies[0]["questions"]["team"]["criteria"] == {"billing": "Payments", "tech": None}


async def test_attribution_headers_are_sent_on_every_request(client, drex):
    scorer = drex_scorer("ok", client=client)
    await judge(scorer)
    await judge(scorer)
    for request in drex.requests:
        assert request.headers["x-nace-client"] == CLIENT_NAME
        assert request.headers["user-agent"] == CLIENT_NAME
    assert CLIENT_NAME.startswith("inspect-drex/")


async def test_422_shows_each_issue_and_is_not_a_failed_criterion(client, drex):
    drex.replies.append(
        error(
            422,
            "invalid request",
            issues=[
                {"path": "questions", "message": "too many questions"},
                {"path": "state", "message": "required"},
            ],
        )
    )
    with pytest.raises(DrexAPIError) as raised:
        await judge(drex_scorer([Criterion.noul(f"q{i}", "?") for i in range(3)], client=client))
    assert "questions: too many questions" in str(raised.value)
    assert "state: required" in str(raised.value)


async def test_402_points_to_billing(client, drex):
    drex.replies.append(error(402, "no available funds", kind="insufficient_credit_error"))
    with pytest.raises(DrexAPIError, match="billing") as raised:
        await judge(drex_scorer("ok", client=client))
    assert "https://drex.test/dashboard/billing" in str(raised.value)


async def test_401_is_an_authentication_error(client, drex):
    drex.replies.append(error(401, "Invalid API key", kind="authentication_error"))
    with pytest.raises(DrexAPIError) as raised:
        await judge(drex_scorer("ok", client=client))
    assert raised.value.status == 401


@pytest.mark.parametrize(("status", "kind"), [(429, "rate_limit_error"), (529, "overloaded_error")])
async def test_429_and_529_are_retried_after_retry_after_ms(client, drex, sleeps, status, kind):
    drex.replies.append(error(status, "slow", kind=kind, headers={"retry-after-ms": "1500"}))
    score = await judge(drex_scorer("ok", client=client))
    assert len(drex.bodies) == 2
    assert sleeps == [pytest.approx(1.5)]
    assert score.value == pytest.approx(0.9)


async def test_exhausted_retries_are_an_execution_error(make_client, drex, sleeps):
    client = make_client(max_retries=1)
    drex.replies.extend(error(429, "slow", kind="rate_limit_error") for _ in range(2))
    with pytest.raises(DrexAPIError) as raised:
        await judge(drex_scorer("ok", client=client))
    assert raised.value.status == 429


async def test_on_error_unscored_keeps_the_sample_but_not_its_value(client, drex):
    drex.replies.append(error(402, "no available funds", kind="insufficient_credit_error"))
    score = await judge(drex_scorer("ok", on_error="unscored", client=client))
    assert math.isnan(score.value)
    assert score.reason == "scoring_failed"
    assert "402" in score.explanation
    assert score.answer == "Click 'Forgot password'."


def run_eval(scorer, *, solver=None, samples=2, tmp_path=None):
    task = Task(
        dataset=[Sample(input=f"question {i}", target="ref") for i in range(samples)],
        solver=solver or generate(),
        scorer=scorer,
    )
    (log,) = eval(
        task,
        model="mockllm/model",
        display="none",
        log_dir=str(tmp_path),
        fail_on_error=False,
        max_samples=1,
    )
    return log


def test_in_an_eval_the_metrics_report_the_mean_and_stderr(client, drex, tmp_path):
    drex.replies.extend([_ok(1.0, "req_a"), _ok(0.5, "req_b")])
    log = run_eval(drex_scorer("ok", client=client), tmp_path=tmp_path)
    assert log.status == "success"
    metrics = log.results.scores[0].metrics
    assert metrics["mean"].value == pytest.approx(0.75)
    assert metrics["stderr"].value == pytest.approx(0.25)
    assert [s.score.metadata["drex"]["request_id"] for s in log.samples] == ["req_a", "req_b"]


def test_in_an_eval_an_execution_error_errors_the_sample_not_the_score(client, drex, tmp_path):
    drex.replies.extend(
        [
            error(402, "no available funds", kind="insufficient_credit_error"),
            _ok(1.0, "req_a"),
        ]
    )
    log = run_eval(drex_scorer("ok", client=client), tmp_path=tmp_path)
    failed, scored = log.samples
    assert failed.error is not None
    assert "402" in failed.error.message
    assert not failed.scores  # an errored sample carries no score at all
    assert scored.error is None
    assert log.results.scores[0].metrics["mean"].value == pytest.approx(1.0)


def test_unscored_samples_stay_in_the_log_and_out_of_the_metrics(client, drex, tmp_path):
    drex.replies.extend(
        [
            error(402, "no available funds", kind="insufficient_credit_error"),
            _ok(1.0, "req_a"),
        ]
    )
    log = run_eval(drex_scorer("ok", on_error="unscored", client=client), tmp_path=tmp_path)
    unscored, scored = log.samples
    assert unscored.scores["drex_scorer"].reason == "scoring_failed"
    assert log.results.scores[0].metrics["mean"].value == pytest.approx(1.0)
    assert scored.scores["drex_scorer"].value == pytest.approx(1.0)


def _ok(noul: float, request_id: str) -> tuple[int, dict[str, str], dict]:
    body = {
        "model": "drex-max",
        "answers": {"criterion": {"type": "noul", "noul": noul}},
        "usage": {"input_tokens": 10, "image_tokens": 0, "output_tokens": 1},
        "evaluation_time_ms": 5.0,
    }
    return 200, {"x-request-id": request_id}, body


async def test_the_scorer_is_registered_for_inspect_score(client, drex):
    """`inspect score <log> --scorer inspect_drex/drex_scorer -S criteria=...` resolves it."""
    from inspect_ai._util.registry import registry_info
    from inspect_ai.scorer._scorer import scorer_create

    created = scorer_create(
        "inspect_drex/drex_scorer", criteria="The answer is helpful.", client=client
    )
    assert registry_info(created).name == "inspect_drex/drex_scorer"
    assert (await created(task_state(), Target("ref"))).value == pytest.approx(0.9)


def _level(score: float) -> dict:
    return {
        "type": "score",
        "score": score,
        "legend": ["rude", "neutral", "polite"],
        "probabilities": {"0": 0.0, "1": 1.0, "2": 0.0},
        "confidence": 0.9,
    }


@pytest.mark.parametrize(
    ("level", "invert", "expected"),
    [(0.0, False, 0.0), (1.0, False, 0.5), (2.0, False, 1.0), (0.0, True, 1.0), (2.0, True, 0.0)],
)
async def test_score_rubrics_have_an_explicit_direction(client, drex, level, invert, expected):
    drex.overrides["tone"] = _level(level)
    rubric = Criterion.score("tone", "How polite?", ["rude", "neutral", "polite"], invert=invert)
    score = await judge(drex_scorer(rubric, client=client))
    assert score.value == pytest.approx(expected)
    assert ("lower is better" in score.metadata["drex"]["criteria"]["tone"]["detail"]) == invert


async def test_an_answer_of_the_wrong_type_is_an_execution_error_not_a_zero(client, drex):
    drex.overrides["criterion"] = {
        "type": "choice",
        "choice": "a",
        "confidence": 0.8,
        "probabilities": {"a": 1.0},
    }
    with pytest.raises(DrexError, match="unexpected answer type"):
        await judge(drex_scorer("ok", client=client))
    score = await judge(drex_scorer("ok", on_error="unscored", client=client))
    assert math.isnan(score.value)
    assert score.reason == "scoring_failed"


def test_credentials_are_refused_because_inspect_logs_scorer_arguments():
    with pytest.raises(ValueError, match="plain text"):
        drex_scorer("ok", api_key="nace_sk_secret")
    with pytest.raises(ValueError, match="plain text"):
        drex_scorer("ok", extra_headers={"x": "y"})


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_a_non_finite_probability_is_an_execution_error_not_a_pass(bad):
    with pytest.raises(DrexError, match="non-finite"):
        Criterion.noul("c", "ok").passing(NoulAnswer(type="noul", noul=bad))


@pytest.mark.parametrize("invert", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_a_non_finite_score_is_an_execution_error_not_a_clamped_result(bad, invert):
    criterion = Criterion.score("c", "ok", ["bad", "mid", "good"], invert=invert)
    answer = ScoreAnswer(
        score=bad,
        confidence=1.0,
        legend=("bad", "mid", "good"),
        probabilities={0: 0.2, 1: 0.3, 2: 0.5},
    )
    with pytest.raises(DrexError, match="non-finite"):
        criterion.passing(answer)


@pytest.mark.parametrize("invert", [False, True])
def test_a_score_answer_without_levels_is_an_execution_error(invert):
    criterion = Criterion.score("c", "ok", ["bad", "good"], invert=invert)
    answer = ScoreAnswer(score=0.0, confidence=1.0, legend=(), probabilities={})
    with pytest.raises(DrexError, match="no score levels"):
        criterion.passing(answer)


async def test_a_missing_answer_is_an_execution_error_not_a_zero(client, drex):
    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    with pytest.raises(DrexError, match="no answer"):
        await judge(drex_scorer("ok", client=client))
    drex.replies.append(lambda body: (200, {}, {**ok_body(body), "answers": {}}))
    score = await judge(drex_scorer("ok", on_error="unscored", client=client))
    assert math.isnan(score.value)
    assert score.reason == "scoring_failed"
