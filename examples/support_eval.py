"""Judge a support assistant's answers with Drex inside an Inspect eval.

    uv run python examples/support_eval.py                    # offline: mockllm + a fake judge
    uv run python examples/support_eval.py --live             # real Drex (NACE_API_KEY)
    uv run python examples/support_eval.py --live --model drex-v2.0

The task replays canned assistant answers (`mockllm`), so only the judge is real with
`--live`. One Drex request per answer asks two questions at once: is the answer grounded in
the reference fact, and is it free of filler. Inspect reports the mean pass score and its
standard error; every sample keeps Drex's probabilities in `score.metadata["drex"]`, so you
can choose a threshold after the run.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from typing import Any

import httpx
from inspect_ai import Task, eval
from inspect_ai.dataset import Sample
from inspect_ai.model import ModelOutput, get_model
from inspect_ai.solver import generate

from inspect_drex import AsyncDrexClient, Criterion, drex_scorer

FACTS = {
    "How long do refunds take?": "5 business days",
    "Can I change my plan mid-month?": "prorated",
    "Where do I find my invoices?": "billing page",
    "Is there a free tier?": "120 requests per minute",
}
ANSWERS = [
    "Refunds take 5 business days.",
    "We're so sorry for the trouble! Please contact support for details.",
    "Invoices are on the billing page.",
    "Sorry, I can't say.",
]
CRITERIA = [
    Criterion.noul("grounded", "Does the answer state the reference fact?", threshold=0.7),
    Criterion.noul("filler", "Is the answer padded with filler or apologies?", want=False),
]


def fake_judge(request: httpx.Request) -> httpx.Response:
    """A stand-in for Drex that judges by keyword, so the example runs with no key."""
    body = json.loads(request.content)
    state = body["state"]
    answers: dict[str, Any] = {}
    for name in body["questions"]:
        if name == "grounded":
            yes = 0.95 if state["reference"].lower() in state["output"].lower() else 0.05
        else:
            yes = 0.9 if "sorry" in state["output"].lower() else 0.05
        answers[name] = {"type": "noul", "noul": yes}
    usage = {"input_tokens": 60, "image_tokens": 0, "output_tokens": 2}
    payload = {"model": body["model"], "answers": answers, "usage": usage, "evaluation_time_ms": 4}
    return httpx.Response(200, json=payload, headers={"x-request-id": "req_fake"})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--live", action="store_true", help="call real Drex (NACE_API_KEY)")
    parser.add_argument("--model", default=None, help="a Drex model id (default drex-max)")
    args = parser.parse_args()

    client = (
        AsyncDrexClient(model=args.model)
        if args.live
        else AsyncDrexClient(
            model=args.model,
            api_key="nace_sk_offline",
            base_url="https://drex.invalid",
            http_client=httpx.AsyncClient(transport=httpx.MockTransport(fake_judge)),
        )
    )
    assistant = get_model(
        "mockllm/model",
        custom_outputs=[ModelOutput.from_content("mockllm/model", text) for text in ANSWERS],
    )
    task = Task(
        dataset=[Sample(input=q, target=fact) for q, fact in FACTS.items()],
        solver=generate(),
        scorer=drex_scorer(CRITERIA, client=client),
    )
    with tempfile.TemporaryDirectory() as logs:
        (log,) = eval(task, model=assistant, log_dir=logs, display="none", max_samples=4)
        assert log.results is not None
        metrics = log.results.scores[0].metrics
        print(f"judge: {client.model}")
        print(f"mean pass score {metrics['mean'].value:.2f} (stderr {metrics['stderr'].value:.2f})")
        for sample in log.samples or []:
            assert sample.scores is not None
            print(f"- {sample.input!r}: {sample.scores['drex_scorer'].value:.2f}")
    usage = client.usage
    print(f"judge requests {usage.requests}, input tokens {usage.input_tokens}")


if __name__ == "__main__":
    main()
