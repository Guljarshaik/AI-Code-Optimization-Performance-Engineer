"""genetic_engine.py — Population management and classical genetic operators."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .fitness import FitnessReport
from .genome import CodeGenome


@dataclass
class EvaluatedGenome:
    genome: CodeGenome
    fitness: FitnessReport


@dataclass
class Population:
    individuals: List[EvaluatedGenome] = field(default_factory=list)
    generation: int = 0
    history: List[dict] = field(default_factory=list)

    def best(self) -> Optional[EvaluatedGenome]:
        if not self.individuals:
            return None
        return max(self.individuals, key=lambda e: e.fitness.scalar)

    def record_generation_stats(self) -> None:
        scores = [e.fitness.scalar for e in self.individuals]
        self.history.append({
            "generation": self.generation,
            "best": max(scores) if scores else 0.0,
            "mean": sum(scores) / len(scores) if scores else 0.0,
            "worst": min(scores) if scores else 0.0,
            "pop_size": len(self.individuals),
        })


class GeneticEngine:
    def __init__(self, population_size: int = 20, elite_fraction: float = 0.1,
                 tournament_size: int = 3, crossover_rate: float = 0.6,
                 mutation_rate: float = 0.3, rng: Optional[random.Random] = None):
        self.population_size = population_size
        self.elite_count = max(1, int(population_size * elite_fraction))
        self.tournament_size = tournament_size
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.rng = rng or random.Random()

    def tournament_select(self, population: Population) -> EvaluatedGenome:
        contenders = self.rng.sample(population.individuals,
                                      k=min(self.tournament_size, len(population.individuals)))
        return max(contenders, key=lambda e: e.fitness.scalar)

    def elites(self, population: Population) -> List[EvaluatedGenome]:
        ranked = sorted(population.individuals, key=lambda e: e.fitness.scalar, reverse=True)
        return ranked[: self.elite_count]

    def crossover(self, a: CodeGenome, b: CodeGenome) -> CodeGenome:
        blocks_a, blocks_b = a.statement_blocks(), b.statement_blocks()
        if len(blocks_a) < 2 or len(blocks_b) < 2:
            return a.child(a.source, operator="crossover(no-op:too-small)")
        cut_a = self.rng.randint(1, len(blocks_a) - 1)
        cut_b = self.rng.randint(1, len(blocks_b) - 1)
        new_source = "\n".join(blocks_a[:cut_a] + blocks_b[cut_b:])
        child = a.child(new_source, operator="crossover")
        child.parent_ids = [a.genome_id, b.genome_id]
        return child

    def mutate(self, genome: CodeGenome) -> CodeGenome:
        source = genome.source
        choice = self.rng.choice(["literal", "comparator", "noop"])

        if choice == "literal":
            import re
            def bump(match):
                val = int(match.group(0))
                delta = self.rng.choice([-2, -1, 1, 2])
                return str(max(0, val + delta))
            new_source = re.sub(r"(?<![\w.])\d+(?![\w.])", bump, source, count=1)
        elif choice == "comparator":
            swaps = {"<=": "<", "<": "<=", ">=": ">", ">": ">=", "==": "!="}
            new_source = source
            for old, new in swaps.items():
                if old in source:
                    new_source = source.replace(old, new, 1)
                    break
            else:
                new_source = source
        else:
            new_source = source

        return genome.child(new_source, operator=f"mutate:{choice}")

    def next_generation_candidates(self, population: Population) -> List[CodeGenome]:
        next_gen: List[CodeGenome] = [e.genome for e in self.elites(population)]

        while len(next_gen) < self.population_size:
            if self.rng.random() < self.crossover_rate:
                p1 = self.tournament_select(population).genome
                p2 = self.tournament_select(population).genome
                child = self.crossover(p1, p2)
            else:
                parent = self.tournament_select(population).genome
                child = parent.child(parent.source, operator="clone")

            if self.rng.random() < self.mutation_rate:
                child = self.mutate(child)

            next_gen.append(child)

        return next_gen[: self.population_size]
