"""Criteria: what to ask Drex, and how an answer becomes a 0-1 pass score."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import nace_sdk

from inspect_drex.errors import DrexError

Kind = Literal["noul", "choice", "score"]


@dataclass(frozen=True)
class Criterion:
    """One thing to check, as a Drex question plus how an answer becomes a 0-1 *pass* score.

    The pass score always means "how well this criterion holds", higher is better, so a
    "defect" question needs `want=False`. Build one with `Criterion.noul`, `.choice` or
    `.score`; thresholds belong to you (`threshold` is the lowest passing score).
    """

    name: str
    kind: Kind
    question: Any
    threshold: float = 0.5
    weight: float = 1.0
    want: bool = True  # noul: want yes (True) or no (False)
    good: frozenset[str] = field(default_factory=frozenset)  # choice: acceptable labels
    invert: bool = False  # score: True when level 0 is the best

    @classmethod
    def noul(
        cls,
        name: str,
        instructions: str,
        *,
        want: bool = True,
        threshold: float = 0.5,
        weight: float = 1.0,
        criteria: Mapping[str, str] | None = None,
    ) -> Criterion:
        """A yes/no question. Pass score: P(yes) if `want` else P(no)."""
        question = nace_sdk.Noul(
            instructions=instructions, criteria=dict(criteria) if criteria else None
        )
        return cls(name, "noul", question, threshold, weight, want=want)

    @classmethod
    def choice(
        cls,
        name: str,
        instructions: str,
        options: Mapping[str, str | None],
        *,
        good: str | Sequence[str],
        threshold: float = 0.5,
        weight: float = 1.0,
    ) -> Criterion:
        """Pick one label. Pass score: total probability of the `good` labels."""
        labels = frozenset([good] if isinstance(good, str) else good)
        unknown = labels - set(options)
        if unknown or not labels:
            raise ValueError(f"`good` must name labels in `options`; got {sorted(labels)}.")
        question = nace_sdk.Choice(instructions=instructions, criteria=dict(options))
        return cls(name, "choice", question, threshold, weight, good=labels)

    @classmethod
    def score(
        cls,
        name: str,
        instructions: str,
        levels: Sequence[str],
        *,
        threshold: float = 0.5,
        weight: float = 1.0,
        invert: bool = False,
    ) -> Criterion:
        """An ordered rubric, lowest first.

        Pass score: expected level / (levels - 1), or its complement with `invert=True`.
        The raw expected level is kept in the feedback.
        """
        if len(levels) < 2:
            raise ValueError("A score rubric needs at least two levels.")
        question = nace_sdk.Score(instructions=instructions, criteria=list(levels))
        return cls(name, "score", question, threshold, weight, invert=invert)

    def passing(self, answer: Any) -> tuple[float, str]:
        """The 0-1 pass score for `answer`, and a factual description of the answer."""
        value, detail = self._passing(answer)
        if not math.isfinite(value):
            raise DrexError(f"Drex returned a non-finite value for criterion {self.name!r}.")
        return value, detail

    def _passing(self, answer: Any) -> tuple[float, str]:
        if self.kind == "noul" and isinstance(answer, nace_sdk.NoulAnswer):
            yes = float(answer.noul)
            return (
                yes if self.want else 1.0 - yes
            ), f"P(yes)={yes:.2f}, wanted {'yes' if self.want else 'no'}"
        if self.kind == "choice" and isinstance(answer, nace_sdk.ChoiceAnswer):
            probabilities = {str(k): float(v) for k, v in answer.probabilities.items()}
            good = sum(p for label, p in probabilities.items() if label in self.good)
            shown = ", ".join(f"{k}={v:.2f}" for k, v in probabilities.items())
            return good, f"picked {answer.choice!r} ({shown}); acceptable: {sorted(self.good)}"
        if self.kind == "score" and isinstance(answer, nace_sdk.ScoreAnswer):
            top = len(answer.probabilities) - 1
            if top < 1:
                raise DrexError(f"Drex returned no score levels for criterion {self.name!r}.")
            raw = float(answer.score)
            if not math.isfinite(raw):
                raise DrexError(f"Drex returned a non-finite value for criterion {self.name!r}.")
            normalized = min(max(raw / top, 0.0), 1.0)
            value = 1.0 - normalized if self.invert else normalized
            return value, f"expected level {float(answer.score):.2f} of 0-{top}" + (
                " (lower is better)" if self.invert else ""
            )
        raise DrexError(f"Drex answered criterion {self.name!r} with an unexpected answer type.")


@dataclass
class Verdict:
    """What one judgment found, per criterion, with the request's metadata."""

    score: float
    passed: bool
    scores: dict[str, float]
    failed: list[str]
    details: dict[str, str]
    metadata: dict[str, Any]

    def feedback(self, criteria: Sequence[Criterion]) -> str:
        """Failed criteria with their probabilities; nothing is inferred beyond the answers."""
        by_name = {c.name: c for c in criteria}
        if not self.failed:
            return f"All {len(criteria)} criteria passed (score {self.score:.2f})."
        lines = [f"{len(self.failed)} of {len(criteria)} criteria failed (score {self.score:.2f}):"]
        lines += [
            f"- {name}: pass score {self.scores[name]:.2f} < "
            f"threshold {by_name[name].threshold:.2f}; {self.details[name]}"
            for name in self.failed
        ]
        passed = [c.name for c in criteria if c.name not in self.failed]
        if passed:
            lines.append("Passed: " + ", ".join(passed) + ".")
        return "\n".join(lines)


__all__ = ["Criterion", "Kind", "Verdict"]
