"""
Author: Tim Frenzel
Version: 1.00
Usage:  Imported by other scripts (e.g., evaluate.py, quick_demo.py).
        `agg = ConsensusAggregator(method=...); best_dx, scores = agg(answers)`
        Run demo with `python src/consensus.py --demo`

Objective of the Code:
------------
Provide a lightweight, self‑contained consensus layer for the Mixture‑of‑Agents
(MoA) healthcare‑diagnosis pipeline.  The module ingests the individual
predictions (diagnosis, confidence, agent id) produced by the specialist agents
(Layer‑1) and the GPT‑3.5 refinement agent (Layer‑2), then returns a single
best‑estimate diagnosis plus provenance metadata.  Two aggregation strategies
are implemented:  
1. **Softmax‑weighted voting** based on agent confidence scores.  
2. **Simple majority vote** with confidence tie‑breaks.  
The design is dependency‑free, GPU‑free, and includes a minimal self‑test that
can be executed with `python consensus.py --demo`.
"""
from __future__ import annotations

import argparse
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, List, Dict, Literal, Tuple

__all__ = [
    "AgentAnswer",
    "ConsensusAggregator",
]

AggregationMethod = Literal["softmax", "majority"]


@dataclass(slots=True)
class AgentAnswer:
    """Container for a single agent's output.

    Attributes
    ----------
    agent_id: str
        Human‑readable identifier of the agent (e.g. "Cardiology‑Phi2").
    diagnosis: str
        The primary ICD‑10 / SNOMED code or free‑text diagnosis string proposed
        by the agent.
    confidence: float
        Self‑reported probability *p( diagnosis | data )* in **[0, 1]**.  Values
        outside the range are automatically clipped.
    reasoning: str | None, optional
        Optional free‑text rationale – stored for provenance but not used by the
        aggregator.
    extra: dict, optional
        Arbitrary metadata (latency, token‑cost, etc.) preserved for logging.
    """

    agent_id: str
    diagnosis: str
    confidence: float
    reasoning: str | None = None
    extra: Dict[str, float] = field(default_factory=dict)

    # --- built‑ins ---------------------------------------------------------
    def __post_init__(self) -> None:  # noqa: D401 – short
        # Clamp the confidence to [0,1] to avoid math errors.
        object.__setattr__(self, "confidence", max(0.0, min(self.confidence, 1.0)))

    def __repr__(self) -> str:  # noqa: D401 – short
        return f"<{self.agent_id}: {self.diagnosis} ({self.confidence:.2f})>"


class ConsensusAggregator:
    """Compute a final diagnosis from multiple *AgentAnswer* objects."""

    def __init__(self, method: AggregationMethod = "softmax", *, temp: float = 1.0) -> None:
        self.method = method
        if temp <= 0:
            raise ValueError("Temperature must be > 0 for softmax weighting.")
        self.temp = temp  # used only in softmax mode

    # ---------------------------------------------------------------------
    # public API
    # ---------------------------------------------------------------------
    def __call__(self, answers: Iterable[AgentAnswer]) -> Tuple[str, Dict[str, float]]:
        """Return *best_diagnosis* and per‑diagnosis aggregated scores."""
        answers_list: List[AgentAnswer] = list(answers)
        if not answers_list:
            raise ValueError("ConsensusAggregator received no answers.")

        if self.method == "softmax":
            scores = self._softmax_pool(answers_list)
        elif self.method == "majority":
            scores = self._majority_pool(answers_list)
        else:  # pragma: no cover – mypy safeguard
            raise ValueError(f"Unsupported aggregation method: {self.method}")

        # Select diagnosis with highest aggregate score, break ties by median
        # individual confidence (more conservative than max).
        best_diag = max(
            scores.items(),
            key=lambda kv: (kv[1]["score"], kv[1]["median_confidence"]),
        )[0]
        # Return simple dict mapping diagnosis -> score for downstream analysis
        flat_scores = {d: m["score"] for d, m in scores.items()}
        return best_diag, flat_scores

    # ------------------------------------------------------------------
    # internal helpers
    # ------------------------------------------------------------------
    def _softmax_pool(self, answers: List[AgentAnswer]) -> Dict[str, Dict[str, float]]:
        """Aggregate by temperature‑scaled softmax over confidences."""
        # Collect confidences per diagnosis
        diag2conf: Dict[str, List[float]] = defaultdict(list)
        for a in answers:
            diag2conf[a.diagnosis].append(a.confidence)

        diag2score: Dict[str, Dict[str, float]] = {}
        for diag, confs in diag2conf.items():
            # average confidence for that diagnosis
            avg_conf = statistics.mean(confs)
            diag2score[diag] = {
                "raw_confidences": confs,
                "avg_confidence": avg_conf,
            }
        # Softmax over the *avg_confidence* values
        logits = {
            diag: avg / self.temp for diag, avg in (d["avg_confidence"] for d in diag2score.items())  # type: ignore[arg-type]
        }
        # To keep ordering stable we recompute dict
        logits = {diag: d["avg_confidence"] / self.temp for diag, d in diag2score.items()}
        max_logit = max(logits.values())
        exp_sum = 0.0
        for diag, logit in logits.items():
            val = math.exp(logit - max_logit)
            diag2score[diag]["score"] = val
            exp_sum += val
        for diag in diag2score:
            diag2score[diag]["score"] /= exp_sum  # normalise to 1
            diag2score[diag]["median_confidence"] = statistics.median(diag2score[diag]["raw_confidences"])
        return diag2score

    def _majority_pool(self, answers: List[AgentAnswer]) -> Dict[str, Dict[str, float]]:
        """Aggregate by simple vote count; break ties via mean confidence."""
        votes = Counter(a.diagnosis for a in answers)
        # collate confidences for tie‑breaking and optional downstream debug
        diag2confs: Dict[str, List[float]] = defaultdict(list)
        for a in answers:
            diag2confs[a.diagnosis].append(a.confidence)

        diag2score: Dict[str, Dict[str, float]] = {}
        total_votes = sum(votes.values())
        for diag, n_votes in votes.items():
            confs = diag2confs[diag]
            diag2score[diag] = {
                "score": n_votes / total_votes,  # normalised vote share
                "median_confidence": statistics.median(confs),
                "raw_confidences": confs,
            }
        return diag2score


# ---------------------------------------------------------------------------
# Minimal built‑in demo / unit test
# ---------------------------------------------------------------------------

def _demo() -> None:  # pragma: no cover
    print("Running consensus.py demo…\n")
    sample_answers = [
        AgentAnswer("Cardiology‑Phi2", "STEMI", 0.85),
        AgentAnswer("Metabolic‑Phi2", "Non‑STEMI", 0.30),
        AgentAnswer("General‑Phi2", "STEMI", 0.60),
        AgentAnswer("GPT‑3.5‑Refine", "STEMI", 0.78),
    ]

    for method in ("softmax", "majority"):
        agg = ConsensusAggregator(method)
        best, scores = agg(sample_answers)
        print(f"Method = {method:8s}  → Best diagnosis: {best:>10s}  (scores: {scores})")

    print("\nAll good – demo finished.")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(description="Run built‑in demo for ConsensusAggregator.")
    parser.add_argument("--demo", action="store_true", help="Execute demo mode (default).")
    parser.add_argument("--method", choices=["softmax", "majority"], default="softmax", help="Aggregation method.")
    args = parser.parse_args()

    if args.demo:
        _demo()
    else:
        # For quick CLI use; read simple answers from stdin
        print("Enter answers as '<agent_id>::<diagnosis>::<confidence>' per line. End with EOF (Ctrl‑D).")
        lines = [ln.strip() for ln in iter(input, "")]  # type: ignore[arg-type]
        parsed: List[AgentAnswer] = []
        for ln in lines:
            try:
                aid, diag, conf = ln.split("::")
                parsed.append(AgentAnswer(aid, diag, float(conf)))
            except ValueError:
                raise SystemExit(f"Invalid input line: {ln!r}")
        best, _ = ConsensusAggregator(args.method)(parsed)
        print(f"\n>>> Final consensus diagnosis: {best}")
