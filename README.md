# Closed-Loop Genetic–LLM Hybrid Optimizer — Reference Implementation and Measured Evaluation

Code accompanying the paper *"Closed-Loop Genetic–LLM Hybrid System for
Autonomous Software Optimization."* Contains the reference architecture
implementation (Sections III–IV of the paper) and the full harness used to
produce every table and figure in Section VI (Results).

## Layout

```
code/
├── hybrid_optimizer/
│   ├── __init__.py
│   ├── genome.py            # CodeGenome: candidate representation
│   ├── sandbox.py           # process-isolated execution harness (reference/production design)
│   ├── fast_sandbox.py      # in-process evaluator used only to make the batch experiment tractable
│   ├── fitness.py           # multi-objective fitness evaluator (corrected hard-correctness gate)
│   ├── genetic_engine.py    # selection, crossover, mutation, population management
│   ├── llm_agent.py         # pluggable semantic backend (Anthropic adapter + offline mock)
│   ├── semantic_rules.py    # RuleBasedSemanticBackend — disclosed deterministic proxy used in Section VI
│   ├── benchmarks.py        # the 5-benchmark suite (Table IV), corrected is_prime oracle
│   └── orchestrator.py      # ClosedLoopOptimizer — the closed-loop controller (Section IV-E)
├── demo.py                  # minimal single-benchmark runnable example
├── run_experiments.py       # reproduces Table V and Figures 3-5 (75 runs, ~seconds)
├── analyze_stats.py         # reproduces Table VI (paired t-tests, pooled + per-benchmark)
├── results/
│   ├── raw_results.json     # full per-seed histories, all 75 runs (canonical run shipped with the paper)
│   ├── summary.csv          # Table V, machine-readable
│   └── stats.json           # Table VI, machine-readable
├── test_hybrid_optimizer.py # 9 unit tests
├── requirements.txt
└── README.md
```

## Quick start

```bash
pip install -r requirements.txt   # stdlib only for the core; scipy needed for analyze_stats.py
python demo.py                    # single-benchmark example, no API key required
```

## Reproducing the paper's Section VI results from scratch

```bash
python run_experiments.py   # writes results/raw_results.json, results/summary.csv (Table V, Figures 3-5)
python analyze_stats.py     # writes results/stats.json (Table VI)
```

**Reproducibility note (also stated in the paper, Section VI-E):** the
performance fitness sub-score depends on measured wall-clock execution
time, which is not perfectly deterministic even with fixed per-seed RNG
seeds, because it reflects real process/interpreter timing noise.
Re-running `run_experiments.py` will reproduce the same qualitative
pattern and the same pooled statistical conclusion, but individual
per-benchmark numbers (and occasionally which benchmarks cross the p<0.05
threshold) can shift slightly between runs. The exact numbers in the
paper are the ones frozen in `results/` in this package; regenerating
them from scratch is expected to land close to, but not bit-for-bit
identical to, those values.

## Running the tests

```bash
python -m pytest -q test_hybrid_optimizer.py
# or, if pytest is unavailable:
python test_hybrid_optimizer.py
```

## Wiring in a real LLM

Replace `RuleBasedSemanticBackend` with `AnthropicBackend` (see
`hybrid_optimizer/llm_agent.py`):

```python
import anthropic
from hybrid_optimizer.llm_agent import LLMAgent, AnthropicBackend

client = anthropic.Anthropic(api_key="sk-...")
llm = LLMAgent(AnthropicBackend(client, model="claude-sonnet-4-6"))
```

Everything else (genetic engine, fitness evaluator, orchestrator) is
unchanged. Re-running `run_experiments.py` with a live backend wired in
is the natural first follow-up experiment the paper calls for (Section
X, Future Work item i) — the current Section VI results use the
disclosed rule-based proxy specifically because no live API access was
available in the environment used to produce the paper.

## Adapting to your own codebase

1. Point `entry_point` at the function name to optimize and supply
   `test_cases` as a list of `{"args": ..., "kwargs": ..., "expected": ...}`.
2. Tune `FitnessEvaluator.weights` to reflect your priorities.
3. Tune `LoopConfig` (population size, semantic-refinement fraction,
   plateau patience) to trade off wall-clock cost against exploration depth.
4. In production, use `Sandbox` (process-isolated), not
   `FastInProcessSandbox` — the latter exists only to make the large
   Section VI batch experiment tractable and trades away process
   isolation for speed; see paper Section IV-G for the disclosure.
