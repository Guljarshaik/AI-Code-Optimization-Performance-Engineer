"""test_hybrid_optimizer.py — Minimal unit tests (run with: python -m pytest -q)."""

try:
    import pytest
except ImportError:
    pytest = None

from hybrid_optimizer.genome import CodeGenome
from hybrid_optimizer.sandbox import Sandbox
from hybrid_optimizer.fitness import FitnessEvaluator
from hybrid_optimizer.genetic_engine import GeneticEngine, Population, EvaluatedGenome
from hybrid_optimizer.llm_agent import LLMAgent, OfflineHeuristicBackend
from hybrid_optimizer.orchestrator import ClosedLoopOptimizer, LoopConfig

ADD_SOURCE = "def add(a, b):\n    return a + b\n"
BAD_SOURCE = "def add(a, b)\n    return a + b\n"


def test_genome_validity():
    assert CodeGenome(ADD_SOURCE).is_syntactically_valid()
    assert not CodeGenome(BAD_SOURCE).is_syntactically_valid()


def test_genome_fingerprint_stable():
    g1, g2 = CodeGenome(ADD_SOURCE), CodeGenome(ADD_SOURCE + "\n")
    assert g1.fingerprint() == g2.fingerprint()


def test_sandbox_runs_and_isolates_errors():
    sb = Sandbox(timeout_s=2.0)
    ok = sb.run(ADD_SOURCE, "add", 2, 3)
    assert ok.success and ok.return_value == 5
    crash = sb.run("def add(a, b):\n    raise ValueError('boom')\n", "add", 1, 1)
    assert not crash.success and "ValueError" in crash.exception


def test_sandbox_timeout():
    sb = Sandbox(timeout_s=0.5)
    infinite = "def add(a, b):\n    while True:\n        pass\n"
    res = sb.run(infinite, "add", 1, 2)
    assert res.timed_out


def test_fitness_rewards_correct_and_penalizes_broken():
    cases = [{"args": (2, 3), "expected": 5}, {"args": (4, 5), "expected": 9}]
    fe = FitnessEvaluator("add", cases)
    good = CodeGenome(ADD_SOURCE)
    fe.calibrate_baseline(good)
    good_report = fe.evaluate(good)
    assert good_report.correctness == 1.0
    assert good_report.scalar > 0.5

    broken = CodeGenome("def add(a, b):\n    return a * b\n")
    broken_report = fe.evaluate(broken)
    assert broken_report.correctness == 0.0
    assert broken_report.scalar < good_report.scalar


def test_fitness_penalizes_risky_constructs():
    cases = [{"args": (2, 3), "expected": 5}]
    fe = FitnessEvaluator("add", cases)
    risky = CodeGenome("def add(a, b):\n    eval('1+1')\n    return a + b\n")
    fe.calibrate_baseline(risky)
    report = fe.evaluate(risky)
    assert report.security < 1.0


def test_genetic_engine_crossover_and_mutation_preserve_validity_when_possible():
    ga = GeneticEngine(population_size=6, rng=__import__("random").Random(0))
    a = CodeGenome("def f(x):\n    y = 1\n    z = 2\n    return x + y + z\n")
    b = CodeGenome("def g(x):\n    y = 10\n    z = 20\n    return x - y - z\n")
    child = ga.crossover(a, b)
    assert isinstance(child.source, str) and len(child.source) > 0
    mutant = ga.mutate(a)
    assert mutant.origin_operator.startswith("mutate:")


def test_tournament_selection_prefers_higher_fitness():
    import random
    ga = GeneticEngine(population_size=4, tournament_size=4, rng=random.Random(1))
    from hybrid_optimizer.fitness import FitnessReport
    pop = Population()
    for i, score in enumerate([0.1, 0.9, 0.3, 0.5]):
        g = CodeGenome(f"def f():\n    return {i}\n")
        fr = FitnessReport(g.genome_id, score, score, score, score, score, 1, 1)
        pop.individuals.append(EvaluatedGenome(g, fr))
    winner = ga.tournament_select(pop)
    assert winner.fitness.scalar == 0.9


def test_orchestrator_end_to_end_offline():
    seed = "def add(a, b):\n    return a + b\n"
    cases = [{"args": (2, 3), "expected": 5}, {"args": (5, 5), "expected": 10}]
    ga = GeneticEngine(population_size=6, rng=__import__("random").Random(0))
    llm = LLMAgent(OfflineHeuristicBackend())
    cfg = LoopConfig(max_generations=3, target_fitness=0.999, llm_refinement_fraction=0.5)
    opt = ClosedLoopOptimizer("add", cases, genetic_engine=ga, llm_agent=llm, config=cfg)
    report = opt.run(seed)
    assert report.best_fitness is not None
    assert report.best_fitness.correctness == 1.0
    assert report.generations_run <= 3


if __name__ == "__main__":
    import sys
    if pytest is not None:
        sys.exit(pytest.main([__file__, "-q"]))
    else:
        import inspect
        fns = [f for name, f in inspect.getmembers(sys.modules[__name__], inspect.isfunction)
               if name.startswith("test_")]
        passed, failed = 0, 0
        for f in fns:
            try:
                f()
                print(f"PASS {f.__name__}")
                passed += 1
            except Exception as e:
                print(f"FAIL {f.__name__} -> {e}")
                failed += 1
        print(f"\n{passed} passed, {failed} failed")
        sys.exit(0 if failed == 0 else 1)
