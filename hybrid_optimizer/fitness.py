"""fitness.py — Multi-objective fitness evaluation (corrected hard-correctness gate)."""

from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .genome import CodeGenome
from .sandbox import Sandbox, ExecutionResult

RISKY_PATTERNS = [
    r"\beval\s*\(", r"\bexec\s*\(", r"os\.system\s*\(",
    r"subprocess\.\w+\([^)]*shell\s*=\s*True", r"pickle\.loads?\s*\(",
    r"__import__\s*\(",
]

DEFAULT_WEIGHTS = {
    "correctness": 0.5,
    "performance": 0.2,
    "maintainability": 0.15,
    "security": 0.15,
}


@dataclass
class FitnessReport:
    genome_id: int
    correctness: float
    performance: float
    maintainability: float
    security: float
    scalar: float
    passed_tests: int
    total_tests: int
    notes: List[str] = field(default_factory=list)

    def vector(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "performance": self.performance,
            "maintainability": self.maintainability,
            "security": self.security,
        }

    def dominates(self, other: "FitnessReport") -> bool:
        mine, theirs = self.vector(), other.vector()
        at_least_as_good = all(mine[k] >= theirs[k] for k in mine)
        strictly_better = any(mine[k] > theirs[k] for k in mine)
        return at_least_as_good and strictly_better


def _cyclomatic_proxy(source: str) -> int:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return 999
    branch_nodes = (ast.If, ast.For, ast.While, ast.Try, ast.BoolOp,
                     ast.With, ast.ExceptHandler)
    return 1 + sum(isinstance(n, branch_nodes) for n in ast.walk(tree))


def _maintainability_score(source: str) -> float:
    complexity = _cyclomatic_proxy(source)
    loc = max(1, len([l for l in source.splitlines() if l.strip()]))
    names_ok = bool(re.search(r"\bdef\s+[a-z_][a-z0-9_]*\s*\(", source))
    density_score = 1.0 / (1.0 + complexity / max(1, loc / 5))
    return max(0.0, min(1.0, 0.8 * density_score + 0.2 * float(names_ok)))


def _security_score(source: str) -> float:
    hits = sum(bool(re.search(p, source)) for p in RISKY_PATTERNS)
    return max(0.0, 1.0 - 0.34 * hits)


class FitnessEvaluator:
    def __init__(self, entry_point: str, test_cases: List[Dict],
                 baseline_wall_time_s: Optional[float] = None,
                 weights: Optional[Dict[str, float]] = None,
                 sandbox: Optional[Sandbox] = None):
        self.entry_point = entry_point
        self.test_cases = test_cases
        self.baseline_wall_time_s = baseline_wall_time_s
        self.weights = weights or DEFAULT_WEIGHTS
        self.sandbox = sandbox or Sandbox()

    def calibrate_baseline(self, genome: CodeGenome) -> None:
        results = self.sandbox.run_test_suite(genome.source, self.entry_point, self.test_cases)
        times = [r.wall_time_s for r in results if r.success]
        self.baseline_wall_time_s = (sum(times) / len(times)) if times else 1e-3

    def evaluate(self, genome: CodeGenome) -> FitnessReport:
        notes: List[str] = []

        if not genome.is_syntactically_valid():
            return FitnessReport(genome.genome_id, 0, 0, 0, 0, 0.0, 0,
                                  len(self.test_cases), notes=["syntax error"])

        results: List[ExecutionResult] = self.sandbox.run_test_suite(
            genome.source, self.entry_point, self.test_cases)

        passed = 0
        wall_times = []
        for case, res in zip(self.test_cases, results):
            if res.success and res.return_value == case.get("expected"):
                passed += 1
                wall_times.append(res.wall_time_s)
            elif res.timed_out:
                notes.append("timeout on a test case")
            elif not res.success:
                notes.append(f"runtime error: {res.exception.splitlines()[-1] if res.exception else 'unknown'}")

        total = max(1, len(self.test_cases))
        correctness = passed / total

        if wall_times and self.baseline_wall_time_s:
            avg_time = sum(wall_times) / len(wall_times)
            speedup = self.baseline_wall_time_s / max(avg_time, 1e-6)
            performance = max(0.0, min(1.0, math.log2(1 + speedup) / math.log2(4)))
        else:
            performance = 0.0

        maintainability = _maintainability_score(genome.source)
        security = _security_score(genome.source)

        # Correctness gates the scalar: a candidate that fails ANY test case
        # cannot outrank one that passes all of them. Gate is correctness ==
        # 1.0, not merely > 0 (an earlier draft gated on > 0, which let a
        # candidate passing a single test case out of many still claim full
        # performance/maintainability/security credit; caught empirically
        # during evaluation, see paper Results: "A Bug the Experiment Itself
        # Caught").
        w = self.weights
        fully_correct = correctness >= 1.0
        scalar = (w["correctness"] * correctness +
                  w["performance"] * performance * fully_correct +
                  w["maintainability"] * maintainability * fully_correct +
                  w["security"] * security * fully_correct)

        return FitnessReport(
            genome_id=genome.genome_id,
            correctness=correctness, performance=performance,
            maintainability=maintainability, security=security,
            scalar=round(scalar, 6), passed_tests=passed, total_tests=total,
            notes=notes,
        )
