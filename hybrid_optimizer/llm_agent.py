"""llm_agent.py — LLM-driven semantic mutation, repair, and critique (pluggable backend)."""

from __future__ import annotations

import abc
import json
import re
from dataclasses import dataclass
from typing import Optional, Tuple

from .fitness import FitnessReport
from .genome import CodeGenome

SYSTEM_PROMPT = """You are a semantic code-mutation engine operating inside an \
autonomous software-optimization loop. You receive a candidate function, its \
multi-objective fitness report, and optional guidance. Return STRICT JSON with \
keys "source" (the full revised function source, syntactically valid Python) \
and "rationale" (<= 2 sentences explaining the change). Do not include markdown \
fences. Preserve the function's public signature unless explicitly asked to \
change it. Never introduce eval/exec/os.system/pickle or other unsafe calls."""


class LLMBackend(abc.ABC):
    @abc.abstractmethod
    def complete(self, system: str, user: str) -> str:
        ...


class AnthropicBackend(LLMBackend):
    """Thin adapter over the Anthropic Messages API."""

    def __init__(self, client, model: str = "claude-sonnet-4-6", max_tokens: int = 1200):
        self.client = client
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, system: str, user: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if getattr(block, "type", "") == "text")


class OfflineHeuristicBackend(LLMBackend):
    """Deterministic, network-free stand-in used for unit tests / demo.py."""

    def complete(self, system: str, user: str) -> str:
        match = re.search(r"CURRENT_SOURCE:\n(.*?)\nFITNESS_REPORT:", user, re.S)
        source = match.group(1).strip() if match else ""
        rationale = "Applied a deterministic offline heuristic refactor (memoization insertion)."
        new_source = self._maybe_memoize(source)
        return json.dumps({"source": new_source, "rationale": rationale})

    @staticmethod
    def _maybe_memoize(source: str) -> str:
        if "functools" in source or "def " not in source:
            return source
        first_def = re.search(r"def\s+(\w+)\s*\(", source)
        if not first_def:
            return source
        return "import functools\n\n@functools.lru_cache(maxsize=None)\n" + source


@dataclass
class LLMMutationResult:
    source: str
    rationale: str
    valid_json: bool = True


class LLMAgent:
    def __init__(self, backend: LLMBackend):
        self.backend = backend

    def _call(self, genome: CodeGenome, fitness: Optional[FitnessReport],
               instruction: str, extra: str = "") -> LLMMutationResult:
        fitness_block = json.dumps(fitness.vector()) if fitness else "{}"
        user = (
            f"INSTRUCTION:\n{instruction}\n\n"
            f"CURRENT_SOURCE:\n{genome.source}\n"
            f"FITNESS_REPORT:\n{fitness_block}\n\n"
            f"{extra}"
        )
        raw = self.backend.complete(SYSTEM_PROMPT, user)
        try:
            payload = json.loads(raw)
            return LLMMutationResult(source=payload["source"], rationale=payload.get("rationale", ""))
        except (json.JSONDecodeError, KeyError):
            return LLMMutationResult(source=raw, rationale="(unparsed LLM response)", valid_json=False)

    def propose_mutation(self, genome: CodeGenome, fitness: Optional[FitnessReport],
                          guidance: str = "") -> Tuple[str, str]:
        instruction = (
            "Propose ONE targeted improvement to raise the scalar fitness "
            "(prioritize correctness, then performance, then maintainability "
            "and security). " + guidance
        )
        result = self._call(genome, fitness, instruction)
        return result.source, result.rationale

    def repair(self, genome: CodeGenome, error_trace: str) -> Tuple[str, str]:
        instruction = "The candidate fails with the error below. Provide a minimal fix."
        result = self._call(genome, None, instruction, extra=f"ERROR_TRACE:\n{error_trace}\n")
        return result.source, result.rationale

    def critique(self, genome: CodeGenome, fitness: FitnessReport) -> str:
        instruction = ("In <= 3 sentences, explain in plain language what changed "
                        "relative to a naive baseline and why fitness moved as it did. "
                        "Return JSON with 'source' set to the UNCHANGED current source "
                        "and 'rationale' set to the explanation.")
        result = self._call(genome, fitness, instruction)
        return result.rationale
