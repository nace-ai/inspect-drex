# inspect-drex

A scorer for [Inspect](https://inspect.aisi.org.uk) (the UK AI Security Institute's evaluation
framework) that asks [Drex](https://console.nace.ai/docs) decision models from Nace whether a
sample's output meets your criteria. Drex answers typed questions and returns
**probabilities**, so you can choose the pass threshold after the run instead of before it.

**Also from Drex:** [Drex Browser Use](https://github.com/nace-ai/drex-browser-harness-pub), an AI browser agent and MCP server. One sentence fills a DoorDash cart in 16 moves.

- Drex Max reads text, JSON and images. `drex-v2.0` reads text and JSON only.
- Any other model id that Drex serves is passed through unchanged; there is no closed list.
- The default model is `drex-max`, which moves to the newest flagship with each Drex release.
  Pin `model=` (or set `NACE_DEFAULT_DECISION_MODEL`) if you need results that never change.
  See the Drex docs for each model's request limits.

## Install

```bash
pip install inspect-drex
export NACE_API_KEY=...   # an API key from the Drex console
```

Python 3.11 or newer (the floor of `inspect-ai`).

`NACE_BASE_URL` is optional. Set it, or pass `base_url=`, for a self-hosted Drex deployment on
your own domain. Any host that is not OpenRouter is treated as Drex itself: requests go to
`<base_url>/v1/systemone` and use your `NACE_API_KEY`.

## Use

```python
from inspect_ai import Task, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate
from inspect_drex import Criterion, drex_scorer


@task
def support():
    return Task(
        dataset=[Sample(input="How long do refunds take?", target="5 business days")],
        solver=generate(),
        scorer=drex_scorer(
            [
                Criterion.noul(
                    "grounded", "Does the answer state the reference fact?", threshold=0.7
                ),
                Criterion.noul("filler", "Is the answer padded with filler?", want=False),
            ],
            model="drex-max",
        ),
    )
```

Save it as `support.py` and run it with any model Inspect supports for the solver:

```bash
inspect eval support.py --model openai/gpt-4o-mini
```

A rubric string is a single yes/no criterion: `drex_scorer("The answer is helpful.")`.
The package registers itself with Inspect through an entry point, so a finished log can also be
scored from the CLI:

```bash
inspect score logs/run.eval --scorer inspect_drex/drex_scorer -S criteria="The answer is helpful."
```

`examples/support_eval.py` is an offline example (a mock model and a fake Drex server).

### What you get back

- `Score.value` is the 0-1 **pass probability** (weighted mean of the criteria, or `min` with
  `aggregate="min"`). `mean()` and `stderr()` report it. With `output="grade"` it is `CORRECT`
  or `INCORRECT` by each criterion's own `threshold`, and `mean()` is the pass rate.
- `Score.metadata["drex"]` keeps what Drex returned: `request_id`, `model`, `input_tokens`,
  `image_tokens`, `evaluation_time_ms`, and per criterion its kind, pass score, threshold,
  verdict and full probabilities (noul: `yes` / `no`; choice and score: per label).
  Input tokens are counted once per request, not once per question.
- For an unscored sample, `Score.metadata["drex"]` holds only `error` (the exception class
  name); `Score.explanation` has the message, including the status and `request_id`.
- `Score.explanation` lists the criteria that failed, with their probabilities.

### Criteria and score direction

`Criterion.noul` asks a yes/no question, `Criterion.choice` picks among labelled options and
`Criterion.score` places the output on an ordered scale. A criterion's pass score always means
"how well it holds": higher is better. A *defect* question ("is the reply rude?") needs
`want=False`, which scores `1 - P(yes)`. Choice criteria sum the probability of the labels you
mark `good`. Score criteria use the expected level over `levels - 1` (or its complement with
`invert=True`); an expected level is a fraction that can fall between two levels, and the raw
level is kept in the detail text. See Drex's guides on
[questions](https://console.nace.ai/docs/drex/guides/questions) and
[reading answers](https://console.nace.ai/docs/drex/guides/reading-answers).

### What Drex sees

By default `{"input", "output", "reference"}`: the user messages, the model output and the
sample target. Give Drex only what the criteria need, or pass your own
`state=lambda state, target: ...` (and `images=` for Drex Max).

Images in the user's messages are sent in order, and the input text names each one
`[Image 1]`, `[Image 2]`, so a criterion can say "Image 1". Images are never dropped: with a
text-only model (`drex-v2.0`, `drex-v1.5`, `drex-latest`) the scorer raises before sending
(with `on_error="unscored"` the sample is left unscored instead). Drex never fetches URLs, so
remote image URLs are refused; use data URLs or local files. Keep each request within Drex's
size limit (inline images count, base64 included). For larger images, upload them with
`nace-sdk` and pass the uploaded file through `images=`.

### Errors are not failed criteria

A 401 (bad key), 402 (no funds; the message links the billing page), 422 (each issue's
`path: message` is shown) or an outage raises, and Inspect marks the sample as errored. With
`on_error="unscored"` the sample stays in the log as `Score.unscored(reason="scoring_failed")`
and is left out of the metrics. 429 and 529 are retried by `nace-sdk`, honouring
`retry-after-ms`. The client timeout is at least 60 s, and the client caps concurrent requests
(`max_in_flight=`, default 8).

### OpenRouter

Drex is also served through OpenRouter. Pass `base_url="https://openrouter.ai/api/v1"` and use
model ids such as `nace-ai/<model>`; a bare id (`drex-max`) is sent as `nace-ai/drex-max`, and ids
with a provider prefix are sent as given. `drex-latest` is refused because OpenRouter has no id
for it.

```bash
export OPENROUTER_API_KEY=...
```

```python
drex_scorer(
    "The answer is helpful.",
    base_url="https://openrouter.ai/api/v1",
    model="nace-ai/drex-max",
)
```

The API key follows the host: for OpenRouter it is read from `OPENROUTER_API_KEY`, never from
`NACE_API_KEY`, so a Nace key is not sent to OpenRouter and an OpenRouter key is not sent to
Nace. Through OpenRouter, images work only inline (in `state` content parts); uploaded and
workspace-file images are refused.

### Secrets and the eval log

Inspect writes scorer arguments to the eval log in plain text, so `drex_scorer` refuses
`api_key=` and `extra_headers=`. Put the key in the environment, or build the client yourself
and pass `client=AsyncDrexClient(base_url=..., api_key=..., model=...)`.

### Attribution

Every request sends `User-Agent` and `X-Nace-Client` as `inspect-drex/<version>`, so Nace can
count usage per integration.

## What is verified

- The offline tests (`tests/unit_tests`) run against a fake Drex server and are the CI gate
  on Python 3.11, 3.12 and 3.13.
- The live tests passed against production Drex and against OpenRouter on 2026-10-29. They are
  not run in CI, because they need keys and make billed requests.

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run pytest tests/unit_tests --cov                          # offline
NACE_API_KEY=... uv run pytest tests/integration_tests -m live -s -rs
OPENROUTER_API_KEY=... uv run pytest tests/integration_tests -m openrouter -s -rs
```

## Limits

- Images need a model that reads them (Drex Max); text-only models raise before sending.
- Each request must fit Drex's size and question limits; one request is made per sample, however
  many criteria it holds.
- The scorer judges what you give it. It does not fetch URLs or files for the judge.

## License

MIT. See [LICENSE](LICENSE). Issues: <https://github.com/nace-ai/inspect-drex/issues>.
