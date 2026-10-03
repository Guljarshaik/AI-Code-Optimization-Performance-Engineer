"""
run_experiments.py — Runs the actual measured evaluation reported in the
paper's Results section: GA-only vs Semantic-only vs Hybrid across the
5-benchmark suite, 5 seeds each (75 runs total).
"""

from __future__ import annotations

import csv
import json
import os
import random
import statistics
import time
from dataclasses import asdict

from hybrid_optimizer.benchmarks import BENCHMARKS
from hybrid_optimizer.fast_sandbox import FastInProcessSandbox
from hybrid_optimizer.fitness import FitnessEvaluator
from hybrid_optimizer.genetic_engine import GeneticEngine
from hybrid_optimizer.genome import CodeGenome
from hybrid_optimizer.llm_agent import LLMAgent
from hybrid_optimizer.orchestrator import ClosedLoopOptimizer, LoopConfig
from hybrid_optimizer.semantic_rules import RuleBasedSemanticBackend

N_SEEDS = 5
MAX_GENERATIONS = 15
POP_SIZE = 10
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def _fast_fitness_evaluator(entry_point, test_cases):
    return FitnessEvaluator(entry_point, test_cases, sandbox=FastInProcessSandbox(timeout_s=1.5))


def run_ga_only(benchmark, seed: int) -> dict:
    rng = random.Random(seed)
    ga = GeneticEngine(population_size=POP_SIZE, elite_fraction=0.2,
                        crossover_rate=0.5, mutation_rate=0.4, rng=rng)
    cfg = LoopConfig(max_generations=MAX_GENERATIONS, target_fitness=0.999,
                      plateau_patience=999, llm_refinement_fraction=0.0)
    opt = ClosedLoopOptimizer(benchmark.entry_point, benchmark.test_cases,
                               genetic_engine=ga, llm_agent=None, config=cfg,
                               fitness_evaluator=_fast_fitness_evaluator(benchmark.entry_point, benchmark.test_cases))
    t0 = time.perf_counter()
    report = opt.run(benchmark.seed_source)
    wall = time.perf_counter() - t0
    return {
        "history": report.history,
        "final_fitness": asdict(report.best_fitness) if report.best_fitness else None,
        "generations_run": report.generations_run,
        "wall_time_s": wall,
        "llm_calls": 0,
    }


def run_semantic_only(benchmark, seed: int) -> dict:
    backend = RuleBasedSemanticBackend()
    agent = LLMAgent(backend)
    fe = _fast_fitness_evaluator(benchmark.entry_point, benchmark.test_cases)
    current = CodeGenome(source=benchmark.seed_source, generation=0, origin_operator="seed")
    fe.calibrate_baseline(current)
    current_report = fe.evaluate(current)

    history = [{"generation": 0, "best": current_report.scalar,
                "mean": current_report.scalar, "worst": current_report.scalar,
                "pop_size": 1}]

    t0 = time.perf_counter()
    for step in range(1, MAX_GENERATIONS + 1):
        try:
            new_source, rationale = agent.propose_mutation(current, current_report)
            candidate = current.child(new_source, operator="semantic_only", rationale=rationale)
            candidate_report = fe.evaluate(candidate)
            if candidate_report.scalar >= current_report.scalar:
                current, current_report = candidate, candidate_report
        except Exception:
            pass
        history.append({"generation": step, "best": current_report.scalar,
                         "mean": current_report.scalar, "worst": current_report.scalar,
                         "pop_size": 1})
    wall = time.perf_counter() - t0

    return {
        "history": history,
        "final_fitness": asdict(current_report),
        "generations_run": MAX_GENERATIONS,
        "wall_time_s": wall,
        "llm_calls": backend.calls_made,
    }


def run_hybrid(benchmark, seed: int) -> dict:
    rng = random.Random(seed)
    ga = GeneticEngine(population_size=POP_SIZE, elite_fraction=0.2,
                        crossover_rate=0.5, mutation_rate=0.4, rng=rng)
    backend = RuleBasedSemanticBackend()
    agent = LLMAgent(backend)
    cfg = LoopConfig(max_generations=MAX_GENERATIONS, target_fitness=0.999,
                      plateau_patience=999, llm_refinement_fraction=0.4)
    opt = ClosedLoopOptimizer(benchmark.entry_point, benchmark.test_cases,
                               genetic_engine=ga, llm_agent=agent, config=cfg,
                               fitness_evaluator=_fast_fitness_evaluator(benchmark.entry_point, benchmark.test_cases))
    t0 = time.perf_counter()
    report = opt.run(benchmark.seed_source)
    wall = time.perf_counter() - t0
    return {
        "history": report.history,
        "final_fitness": asdict(report.best_fitness) if report.best_fitness else None,
        "generations_run": report.generations_run,
        "wall_time_s": wall,
        "llm_calls": backend.calls_made,
    }


CONFIGS = {
    "GA-only": run_ga_only,
    "Semantic-only": run_semantic_only,
    "Hybrid": run_hybrid,
}


def main():
    all_results = {}
    for benchmark in BENCHMARKS:
        all_results[benchmark.name] = {}
        for config_name, runner in CONFIGS.items():
            seed_runs = []
            for seed in range(N_SEEDS):
                result = runner(benchmark, seed)
                seed_runs.append(result)
                ff = result["final_fitness"]["scalar"] if result["final_fitness"] else float("nan")
                print(f"{benchmark.name:18s} | {config_name:14s} | seed={seed} | "
                      f"final={ff:.4f} | gens={result['generations_run']:3d} | "
                      f"llm_calls={result['llm_calls']:3d} | wall={result['wall_time_s']:.3f}s")
            all_results[benchmark.name][config_name] = seed_runs

    with open(os.path.join(OUT_DIR, "raw_results.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    summary_rows = []
    for bname, configs in all_results.items():
        for cname, runs in configs.items():
            finals = [r["final_fitness"]["scalar"] for r in runs if r["final_fitness"]]
            corrects = [r["final_fitness"]["correctness"] for r in runs if r["final_fitness"]]
            perfs = [r["final_fitness"]["performance"] for r in runs if r["final_fitness"]]
            maints = [r["final_fitness"]["maintainability"] for r in runs if r["final_fitness"]]
            secs = [r["final_fitness"]["security"] for r in runs if r["final_fitness"]]
            walls = [r["wall_time_s"] for r in runs]
            calls = [r["llm_calls"] for r in runs]

            def m(xs):
                return statistics.mean(xs) if xs else 0.0

            def s(xs):
                return statistics.stdev(xs) if len(xs) > 1 else 0.0

            summary_rows.append({
                "benchmark": bname, "config": cname,
                "mean_final_fitness": round(m(finals), 4), "std_final_fitness": round(s(finals), 4),
                "mean_correctness": round(m(corrects), 4),
                "mean_performance": round(m(perfs), 4),
                "mean_maintainability": round(m(maints), 4),
                "mean_security": round(m(secs), 4),
                "mean_wall_time_s": round(m(walls), 4),
                "mean_llm_calls": round(m(calls), 2),
                "n_seeds": len(runs),
            })

    with open(os.path.join(OUT_DIR, "summary.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"\nWrote {os.path.join(OUT_DIR, 'raw_results.json')}")
    print(f"Wrote {os.path.join(OUT_DIR, 'summary.csv')}")


if __name__ == "__main__":
    main()
