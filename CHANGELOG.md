# Changelog

## 0.1.0

First release.

- `drex_scorer`: an Inspect scorer that asks Drex yes/no (`noul`), `choice` and `score`
  questions about a sample's output and returns the pass probability as `Score.value`, with
  Drex's full probabilities kept in `Score.metadata["drex"]`.
- Registered with Inspect through the `inspect_ai` entry point, so
  `inspect score --scorer inspect_drex/drex_scorer` works.
- Image input for Drex Max, refused before sending for text-only models.
- Drex through OpenRouter (`nace-ai/<model>` ids) with the key following the host.
- Self-hosted Drex deployments through `NACE_BASE_URL` or `base_url=`.
- Errors raise by default; `on_error="unscored"` leaves the sample out of the metrics.
