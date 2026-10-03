"""orchestrator.py — The closed-loop controller."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

from .fitness import FitnessEvaluator, FitnessReport
from .genetic_engine import EvaluatedGenome, GeneticEngine, Population
from .genome import CodeGenome
from .llm_agent import LLMAgent


@dataclass
class LoopConfig:
    max_generations: int = 50
    target_fitness: float = 0.97
    plateau_patience: int = 8
    plateau_min_delta: float = 0.002
    llm_refinement_fraction: float = 0.35
    repair_on_failure: bool = True


@dataclass
class RunReport:
    best_genome: Optional[CodeGenome]
    best_fitness: Optional[FitnessReport]
    generations_run: int
    history: List[dict]
    critique: str = ""
    converged_reason: str = ""


class ClosedLoopOptimizer:
    def __init__(self, entry_point: str, test_cases: List[Dict],
                 genetic_engine: Optional[GeneticEngine] = None,
                 llm_agent: Optional[LLMAgent] = None,
                 fitness_evaluator: Optional[FitnessEvaluator] = None,
                 config: Optional[LoopConfig] = None):
        self.entry_point = entry_point
        self.ga = genetic_engine or GeneticEngine()
        self.llm = llm_agent
        self.fitness_eval = fitness_evaluator or FitnessEvaluator(entry_point, test_cases)
        self.cfg = config or LoopConfig()
        self.archive: List[EvaluatedGenome] = []

    def _evaluate_with_repair(self, genome: CodeGenome) -> EvaluatedGenome:
        report = self.fitness_eval.evaluate(genome)
        if (report.correctness == 0.0 and self.cfg.repair_on_failure and self.llm is not None
                and (not genome.is_syntactically_valid() or report.notes)):
            error = "; ".join(report.notes) or "syntax error"
            try:
                new_source, rationale = self.llm.repair(genome, error)
                repaired = genome.child(new_source, operator="llm_repair", rationale=rationale)
                repaired_report = self.fitness_eval.evaluate(repaired)
                if repaired_report.scalar >= report.scalar:
                    return EvaluatedGenome(repaired, repaired_report)
            except Exception:
                pass
        return EvaluatedGenome(genome, report)

    def _apply_llm_refinement(self, candidates: List[CodeGenome],
                               population: Population) -> List[CodeGenome]:
        if self.llm is None or not candidates:
            return candidates
        n_refine = int(len(candidates) * self.cfg.llm_refinement_fraction)
        refined = list(candidates)
        best_so_far = population.best()
        reference_fitness = best_so_far.fitness if best_so_far else None
        for i in range(min(n_refine, len(refined))):
            try:
                new_source, rationale = self.llm.propose_mutation(
                    refined[i], reference_fitness,
                    guidance="Focus on the weakest sub-score if the report is provided.")
                refined[i] = refined[i].child(new_source, operator="llm_refine", rationale=rationale)
            except Exception:
                continue
        return refined

    def _check_convergence(self, population: Population) -> Optional[str]:
        if not population.history:
            return None
        best = population.history[-1]["best"]
        if best >= self.cfg.target_fitness:
            return f"target fitness {self.cfg.target_fitness} reached"
        if population.generation >= self.cfg.max_generations:
            return "max generations reached"
        if len(population.history) > self.cfg.plateau_patience:
            window = [h["best"] for h in population.history[-self.cfg.plateau_patience:]]
            if max(window) - min(window) < self.cfg.plateau_min_delta:
                return "fitness plateau detected"
        return None

    def run(self, seed_source: str, on_generation: Optional[Callable[[Population], None]] = None) -> RunReport:
        seed = CodeGenome(source=seed_source, generation=0, origin_operator="seed")
        self.fitness_eval.calibrate_baseline(seed)

        population = Population()
        population.individuals = [self._evaluate_with_repair(seed)]
        while len(population.individuals) < self.ga.population_size:
            child = self.ga.mutate(seed)
            population.individuals.append(self._evaluate_with_repair(child))
        population.record_generation_stats()
        self.archive.extend(population.individuals)
        if on_generation:
            on_generation(population)

        converged_reason = None
        while converged_reason is None:
            population.generation += 1
            raw_candidates = self.ga.next_generation_candidates(population)
            refined_candidates = self._apply_llm_refinement(raw_candidates, population)
            evaluated = [self._evaluate_with_repair(g) for g in refined_candidates]

            population.individuals = evaluated
            population.record_generation_stats()
            self.archive.extend(evaluated)
            if on_generation:
                on_generation(population)

            converged_reason = self._check_convergence(population)

        best = population.best()
        critique = ""
        if self.llm is not None and best is not None:
            try:
                critique = self.llm.critique(best.genome, best.fitness)
            except Exception:
                critique = "(critique unavailable)"

        return RunReport(
            best_genome=best.genome if best else None,
            best_fitness=best.fitness if best else None,
            generations_run=population.generation,
            history=population.history,
            critique=critique,
            converged_reason=converged_reason,
        )
