"""`drex_scorer`: judge an Inspect sample's output with a Drex decision model."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any, Literal

from inspect_ai.model import ChatMessage, ContentImage, ContentText
from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    Target,
    mean,
    scorer,
    stderr,
)
from inspect_ai.solver import TaskState

from inspect_drex._client import AsyncDrexClient
from inspect_drex.criteria import Criterion, Verdict
from inspect_drex.errors import DrexError

StateFn = Callable[[TaskState, Target], Any]
ImagesFn = Callable[[TaskState, Target], Sequence[Any]]

# Inspect's own reason strings for "the scorer could not run" (see `Score.unscored`).
SCORING_FAILED = "scoring_failed"


def _message_text(message: ChatMessage, images: list[ContentImage]) -> str:
    """The text of one message, with "[Image N]" where an image sits (N counts across calls)."""
    if isinstance(message.content, str):
        return message.content
    parts: list[str] = []
    for part in message.content:
        if isinstance(part, ContentText):
            parts.append(part.text)
        elif isinstance(part, ContentImage):
            images.append(part)
            parts.append(f"[Image {len(images)}]")
        else:  # audio, video, reasoning, ...: never dropped silently
            parts.append(f"[{part.type} omitted]")
    return "\n".join(parts)


def default_state(
    state: TaskState, target: Target, *, include_target: bool = True
) -> tuple[dict[str, Any], list[ContentImage]]:
    """`{"input", "output", "reference"}` and the images found in the user's messages.

    Images keep their order, and the input text names each one "[Image N]" so a criterion
    can refer to "Image 1". `reference` is the sample target, when there is one.
    """
    images: list[ContentImage] = []
    turns = [_message_text(m, images) for m in state.messages if m.role == "user"]
    drex_state: dict[str, Any] = {
        "input": "\n\n".join(turns) if turns else state.input_text,
        "output": state.output.completion,
    }
    reference = target.text if include_target else ""
    if reference:
        drex_state["reference"] = reference
    return drex_state, images


def _criteria(rubric: str | Criterion | Sequence[Criterion]) -> list[Criterion]:
    if isinstance(rubric, str):
        return [Criterion.noul("criterion", rubric)]
    criteria = [rubric] if isinstance(rubric, Criterion) else list(rubric)
    if not criteria:
        raise ValueError("At least one criterion is required.")
    names = [c.name for c in criteria]
    if len(set(names)) != len(names):
        raise ValueError("Criterion names must be unique.")
    if any(not math.isfinite(c.weight) or c.weight < 0 for c in criteria):
        raise ValueError("Criterion weights must be finite and non-negative.")
    if sum(c.weight for c in criteria) <= 0:
        raise ValueError("Criterion weights must sum to more than zero.")
    return criteria


def _grade(
    criteria: Sequence[Criterion], answers: dict[str, Any]
) -> tuple[dict[str, float], dict[str, str], list[str]]:
    scores: dict[str, float] = {}
    details: dict[str, str] = {}
    for criterion in criteria:
        answer = answers.get(criterion.name)
        if answer is None:
            raise DrexError(f"Drex returned no answer for criterion {criterion.name!r}.")
        scores[criterion.name], details[criterion.name] = criterion.passing(answer)
    failed = [c.name for c in criteria if scores[c.name] < c.threshold]
    return scores, details, failed


@scorer(metrics=[mean(), stderr()])
def drex_scorer(
    criteria: str | Criterion | Sequence[Criterion],
    *,
    model: str | None = None,
    state: StateFn | None = None,
    images: ImagesFn | None = None,
    include_target: bool = True,
    aggregate: Literal["mean", "min"] = "mean",
    output: Literal["score", "grade"] = "score",
    on_error: Literal["raise", "unscored"] = "raise",
    client: AsyncDrexClient | None = None,
    **client_options: Any,
) -> Scorer:
    """Score each sample by asking Drex whether its output meets your criteria.

    One request per sample, however many criteria (within the model's per-request question limit).

    - **`output="score"`** (default): `Score.value` is the 0-1 pass probability, the weighted
      mean of the criteria (or their `min` with `aggregate="min"`). `mean()` and `stderr()`
      report it directly. The probabilities Drex returned stay in `Score.metadata["drex"]`,
      so you can pick thresholds after the run.
    - **`output="grade"`**: `Score.value` is `CORRECT` when every criterion reaches its own
      `threshold`, else `INCORRECT`. `mean()` is then the pass rate.

    A criterion's pass score always means "how well it holds": a defect question needs
    `want=False` (`Criterion.noul("toxic", "The reply is rude.", want=False)`).

    An execution error (401, 402, 422, an outage) is never a failed criterion. By default it
    is raised and Inspect reports the sample as errored. With `on_error="unscored"` it
    becomes `Score.unscored(reason="scoring_failed")`, kept in the log but excluded from the
    metrics.

    Args:
        criteria: A rubric string (one yes/no criterion), a `Criterion`, or a sequence.
        model: A Drex model id. Default `drex-max`, or `NACE_DEFAULT_DECISION_MODEL`.
        state: `(task_state, target) -> state` for Drex. Default `default_state`: the user
            input, the output and the reference. Give Drex only what the criteria need.
        images: `(task_state, target) -> images` for a custom `state` (Drex Max only).
            Without `state`, images in the user's messages are sent in order.
        include_target: Send the sample target to Drex as `reference` (default state only).
        aggregate: `"mean"` (weighted) or `"min"` of the criteria pass scores.
        output: `"score"` or `"grade"`; see above.
        on_error: `"raise"` or `"unscored"`; see above.
        client: A prepared `AsyncDrexClient`.
        **client_options: Passed to `AsyncDrexClient` (`base_url`, `timeout`,
            `max_in_flight`, ...); `NACE_API_KEY` and `NACE_BASE_URL` are read by default.
            `api_key` and `extra_headers` are refused: Inspect writes scorer arguments to the
            eval log in plain text. Set `NACE_API_KEY` or pass a prepared `client=`.

    """
    if secrets := sorted({"api_key", "extra_headers"} & client_options.keys()):
        raise ValueError(
            f"{', '.join(secrets)} would be written to the eval log in plain text. "
            "Set NACE_API_KEY or pass client=AsyncDrexClient(...)."
        )
    rubric = _criteria(criteria)
    if images is not None and state is None:
        raise ValueError("`images` goes with a custom `state`; the default state finds them.")
    drex = client or AsyncDrexClient(model=model, **client_options)
    judge = model or drex.model

    async def score(task_state: TaskState, target: Target) -> Score:
        try:
            if state is None:
                drex_state, found = default_state(task_state, target, include_target=include_target)
            else:
                drex_state, found = (
                    state(task_state, target),
                    list(images(task_state, target) if images else []),
                )
            ruling = await drex.ask(
                drex_state, {c.name: c.question for c in rubric}, found, model=judge
            )
            scores, details, failed = _grade(rubric, ruling.answers)
        except DrexError as error:
            if on_error == "raise":
                raise
            return Score.unscored(
                reason=SCORING_FAILED,
                answer=task_state.output.completion,
                explanation=str(error),
                metadata={"drex": {"error": type(error).__name__}},
            )
        total = sum(c.weight for c in rubric)
        if aggregate == "min":
            value = min(scores.values())
        else:
            value = sum(scores[c.name] * c.weight for c in rubric) / total
        metadata = ruling.metadata
        verdict = Verdict(value, not failed, scores, failed, details, metadata)
        return Score(
            value=(CORRECT if verdict.passed else INCORRECT) if output == "grade" else value,
            answer=task_state.output.completion,
            explanation=verdict.feedback(rubric),
            metadata={
                "drex": {
                    **metadata,
                    "pass_score": value,
                    "passed": verdict.passed,
                    "criteria": {
                        c.name: {
                            "kind": c.kind,
                            "pass_score": scores[c.name],
                            "threshold": c.threshold,
                            "passed": c.name not in failed,
                            "detail": details[c.name],
                            "probabilities": _probabilities(ruling.answers[c.name]),
                        }
                        for c in rubric
                    },
                }
            },
        )

    return score


def _probabilities(answer: Any) -> dict[str, float]:
    """The answer's full probabilities, keyed by label (noul: yes / no)."""
    if hasattr(answer, "probabilities"):
        return {str(label): float(p) for label, p in answer.probabilities.items()}
    yes = float(answer.noul)
    return {"yes": yes, "no": 1.0 - yes if math.isfinite(yes) else yes}


__all__ = ["default_state", "drex_scorer"]
