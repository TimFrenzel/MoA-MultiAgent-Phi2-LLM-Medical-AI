"""
Author: Tim Frenzel
Version: 1.20
Usage:  Imported by other scripts (e.g., evaluate.py, quick_demo.py).
        `refiner = Refiner(api_key=key, system_prompt_path=...)`
        `diagnosis, rationale = refiner.refine_diagnosis(...)`
        Run demo with `python src/refine.py --demo [--offline]`

Objective of the Code
---------------------
Provide the **refinement layer helper** for the Mixture‑of‑Agents pipeline.  The
module builds a structured "Aggregate‑and‑Synthesize" prompt that feeds the
GPT‑3.5‑Turbo API with (i) the user's original clinical case and (ii) enumerated
candidate answers from specialist agents, then returns the single refined
response.

Context Within MoA
------------------
`refine.py` sits between Layer‑1 (Phi‑2 specialists) and Layer‑3 (consensus).
It transforms multiple draft diagnoses into one improved answer by prompting the
refinement LLM to reason over disagreements, resolve contradictions, and
produce a concise, evidence‑backed diagnosis.  The output plus a self‑reported
confidence score are forwarded to the consensus aggregator.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import textwrap
import time
from dataclasses import dataclass, field
from typing import List, Sequence

# Optional: import openai only if API key present — keeps unit tests lightweight
try:
    import openai  # type: ignore
except ImportError:  # pragma: no cover
    openai = None  # fallback for test environments

###############################################################################
# Prompt template
###############################################################################

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an expert clinical‑reasoning assistant.  Your task is to read several\
    candidate diagnostic assessments provided by domain‑specialist AI systems\
    and synthesise a single, accurate, well‑justified final answer.

    When writing the final answer:
    1. Identify the most probable diagnosis (or differential list in descending\
       likelihood) based on the combined evidence.
    2. Cite which candidate assessments support the conclusion (use their\
       [SOURCE_TAG] identifiers).  If they conflict, explain your resolution in\
       ≤2 sentences.
    3. If the information is insufficient for a confident diagnosis, say\
       "Insufficient evidence" instead of guessing.
    4. Output MUST follow **exactly** this JSON schema:
         {{
           "diagnosis": <string>,
           "reasoning": <string>,
           "confidence": <float 0‑1>
         }}
    """
)

###############################################################################
# Data structures
###############################################################################

@dataclass
class CandidateAnswer:
    """Container for a specialist agent's output."""

    agent_id: str
    answer: str
    confidence: float | None = None  # optional self‑reported confidence
    meta: dict = field(default_factory=dict)

    def to_prompt_block(self) -> str:  # noqa: D401
        """Format block for inclusion in the composite prompt."""
        conf_str = (
            f" (confidence={self.confidence:.2f})" if self.confidence is not None else ""
        )
        return f"[{self.agent_id}]{conf_str}:\n{self.answer.strip()}\n"


###############################################################################
# Prompt builder
###############################################################################

def build_refinement_prompt(  # noqa: D401
    user_case: str, candidates: Sequence[CandidateAnswer]
) -> str:
    """Create the composite user prompt for GPT‑3.5.

    Parameters
    ----------
    user_case:
        The original clinical vignette / patient record.
    candidates:
        Sequence of candidate answers emitted by previous agents.
    """
    candidate_blocks = "\n".join(c.to_prompt_block() for c in candidates)
    user_prompt = textwrap.dedent(
        f"""\
        USER CASE:\n{user_case.strip()}\n\nCANDIDATE ANSWERS:\n{candidate_blocks}\n"""
    )
    return user_prompt


###############################################################################
# GPT‑3.5 caller helper
###############################################################################

_OPENAI_TIMEOUT = 40  # seconds
_MODEL_NAME = os.getenv("REFINE_MODEL", "gpt-3.5-turbo")


def _call_openai(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    if openai is None:  # pragma: no cover
        raise RuntimeError("openai package missing — install openai>=1.0.0 or mock")
    openai.api_key = os.getenv("OPENAI_API_KEY", "")
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY not set in environment")

    response = openai.ChatCompletion.create(
        model=_MODEL_NAME,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
        timeout=_OPENAI_TIMEOUT,
    )
    return response.choices[0].message["content"]  # type: ignore[index]


###############################################################################
# Public API
###############################################################################


def refine(  # noqa: D401
    user_case: str,
    candidates: Sequence[CandidateAnswer],
    temperature: float = 0.3,
    call_llm: bool = True,
) -> dict[str, str | float]:
    """Run the refinement step and return parsed JSON dict."""

    user_prompt = build_refinement_prompt(user_case, candidates)

    if not call_llm:  # for offline unit tests
        mock = {
            "diagnosis": "<mock>",
            "reasoning": "<mock reasoning>",
            "confidence": 0.5,
        }
        return mock

    raw_out = _call_openai(SYSTEM_PROMPT, user_prompt, temperature=temperature)
    try:
        result = json.loads(raw_out)
        assert set(result) == {"diagnosis", "reasoning", "confidence"}
        result["confidence"] = float(result["confidence"])  # ensure numeric
        # Clamp confidence to [0,1]
        result["confidence"] = max(0.0, min(1.0, result["confidence"]))
        return result  # type: ignore[return-value]
    except Exception as exc:  # pragma: no cover — gate unexpected formats
        raise ValueError(f"Bad GPT output: {raw_out}") from exc


###############################################################################
# CLI demo & rudimentary self‑test (no external deps)
###############################################################################

_DEMO_CASE = """72‑year‑old male with crushing substernal chest pain,.."""
_DEMO_CANDIDATES = [
    CandidateAnswer("Cardiology‑Phi2", "Likely STEMI. Recommend immediate PCI.", 0.82),
    CandidateAnswer(
        "Metabolic‑Phi2",
        "Chest pain could be metabolic (DKA) but STEMI more probable. Suggest troponin & PCI.",
        0.55,
    ),
    CandidateAnswer("General‑GPT35", "Consider acute MI; initiate MONA protocol.", 0.70),
]


def _demo() -> None:  # pragma: no cover
    print("# Composite user prompt\n" + build_refinement_prompt(_DEMO_CASE, _DEMO_CANDIDATES)[:800])
    if "--offline" in os.sys.argv:
        res = refine(_DEMO_CASE, _DEMO_CANDIDATES, call_llm=False)
    else:
        res = refine(_DEMO_CASE, _DEMO_CANDIDATES)
    print("\n# GPT‑3.5 refined output\n", json.dumps(res, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a quick refinement demo.")
    parser.add_argument("--demo", action="store_true", help="Run with demo inputs")
    parser.add_argument("--offline", action="store_true", help="Skip real OpenAI call")
    args = parser.parse_args()

    if args.demo:
        if args.offline:
            os.sys.argv.append("--offline")
        _demo()
    else:
        parser.print_help()
