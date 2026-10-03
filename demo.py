"""
demo.py — Minimal end-to-end demonstration of the closed-loop optimizer,
separate from the Section VI batch-experiment harness (run_experiments.py).

Evolves a naive exponential-time fib(n) implementation using the disclosed,
network-free RuleBasedSemanticBackend (see hybrid_optimizer/semantic_rules.py
and paper Section V for the full disclosure) so this script is runnable with
no API key. Swap in AnthropicBackend (hybrid_optimizer/llm_agent.py) for a
live model.
"""

from hybrid_optimizer import ClosedLoopOptimizer
from hybrid_optimizer.genetic_engine import GeneticEngine
from hybrid_optimizer.llm_agent import LLMAgent
from hybrid_optimizer.orchestrator import LoopConfig
from hybrid_optimizer.semantic_rules import RuleBasedSemanticBackend
from hybrid_optimizer.benchmarks import BENCHMARKS

FIB = BENCHMARKS[0]  # fibonacci


def main():
    ga = GeneticEngine(population_size=12, elite_fraction=0.15,
                        crossover_rate=0.5, mutation_rate=0.4)
    llm = LLMAgent(RuleBasedSemanticBackend())
    config = LoopConfig(max_generations=15, target_fitness=0.9,
                         llm_refinement_fraction=0.4, plateau_patience=999)

    optimizer = ClosedLoopOptimizer(
        entry_point=FIB.entry_point,
        test_cases=FIB.test_cases,
        genetic_engine=ga,
        llm_agent=llm,
        config=config,
    )

    def log_generation(population):
        stats = population.history[-1]
        print(f"gen {stats['generation']:>2} | "
              f"best={stats['best']:.3f} mean={stats['mean']:.3f} "
              f"pop={stats['pop_size']}")

    report = optimizer.run(FIB.seed_source, on_generation=log_generation)

    print("\n--- RESULT ---")
    print("Converged reason  :", report.converged_reason)
    print("Generations run   :", report.generations_run)
    print("Best fitness      :", report.best_fitness.scalar if report.best_fitness else None)
    if report.best_fitness:
        print("  correctness     :", report.best_fitness.correctness)
        print("  performance     :", report.best_fitness.performance)
        print("  maintainability :", report.best_fitness.maintainability)
        print("  security        :", report.best_fitness.security)
    print("Critique          :", report.critique)
    print("\nBest genome source:\n")
    print(report.best_genome.pretty() if report.best_genome else "(none)")


if __name__ == "__main__":
    main()
