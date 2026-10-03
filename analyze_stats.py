"""
analyze_stats.py — Paired significance testing across the 5 seeds, in the
same spirit as the reference paper's fold-level paired Wilcoxon tests.
With n=5 paired seeds, a paired t-test is used (more standard than
Wilcoxon at this sample size); results are reported honestly including
non-significant ones.
"""
import json
from scipy import stats

with open("results/raw_results.json") as f:
    data = json.load(f)

BENCHMARKS = ["fibonacci", "is_prime", "has_duplicates", "contains_sorted", "concat_strings"]

print(f"{'Benchmark':18s} {'Comparison':22s} {'mean diff':>10s} {'t':>8s} {'p-value':>10s}")
rows = []
for b in BENCHMARKS:
    hybrid = [r["final_fitness"]["scalar"] for r in data[b]["Hybrid"]]
    ga = [r["final_fitness"]["scalar"] for r in data[b]["GA-only"]]
    sem = [r["final_fitness"]["scalar"] for r in data[b]["Semantic-only"]]

    t1, p1 = stats.ttest_rel(hybrid, ga)
    diff1 = sum(h - g for h, g in zip(hybrid, ga)) / len(hybrid)
    print(f"{b:18s} {'Hybrid vs GA-only':22s} {diff1:>+10.4f} {t1:>8.3f} {p1:>10.4f}")
    rows.append((b, "Hybrid vs GA-only", diff1, t1, p1))

    t2, p2 = stats.ttest_rel(hybrid, sem)
    diff2 = sum(h - s for h, s in zip(hybrid, sem)) / len(hybrid)
    print(f"{b:18s} {'Hybrid vs Semantic-only':22s} {diff2:>+10.4f} {t2:>8.3f} {p2:>10.4f}")
    rows.append((b, "Hybrid vs Semantic-only", diff2, t2, p2))

# Pooled across all benchmarks (paired by seed x benchmark)
hybrid_all, ga_all, sem_all = [], [], []
for b in BENCHMARKS:
    hybrid_all += [r["final_fitness"]["scalar"] for r in data[b]["Hybrid"]]
    ga_all += [r["final_fitness"]["scalar"] for r in data[b]["GA-only"]]
    sem_all += [r["final_fitness"]["scalar"] for r in data[b]["Semantic-only"]]

t1, p1 = stats.ttest_rel(hybrid_all, ga_all)
diff1 = sum(h - g for h, g in zip(hybrid_all, ga_all)) / len(hybrid_all)
print(f"\n{'POOLED (n=25)':18s} {'Hybrid vs GA-only':22s} {diff1:>+10.4f} {t1:>8.3f} {p1:>10.4f}")

t2, p2 = stats.ttest_rel(hybrid_all, sem_all)
diff2 = sum(h - s for h, s in zip(hybrid_all, sem_all)) / len(hybrid_all)
print(f"{'POOLED (n=25)':18s} {'Hybrid vs Semantic-only':22s} {diff2:>+10.4f} {t2:>8.3f} {p2:>10.4f}")

with open("results/stats.json", "w") as f:
    json.dump({
        "per_benchmark": [{"benchmark": r[0], "comparison": r[1], "mean_diff": r[2], "t": r[3], "p": r[4]} for r in rows],
        "pooled": [
            {"comparison": "Hybrid vs GA-only", "mean_diff": diff1, "t": t1, "p": p1, "n": len(hybrid_all)},
            {"comparison": "Hybrid vs Semantic-only", "mean_diff": diff2, "t": t2, "p": p2, "n": len(hybrid_all)},
        ],
    }, f, indent=2)
print("\nWrote results/stats.json")
