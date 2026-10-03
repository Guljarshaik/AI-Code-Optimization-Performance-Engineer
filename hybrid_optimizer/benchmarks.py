"""benchmarks.py — 5-benchmark suite used in the paper's Results section."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List


@dataclass
class Benchmark:
    name: str
    entry_point: str
    seed_source: str
    test_cases: List[Dict]
    description: str


FIB_SEED = """def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
"""
FIB_CASES = [{"args": (i,), "expected": v} for i, v in
             enumerate([0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144])]

IS_PRIME_SEED = """def is_prime(n):
    if n < 2:
        return False
    for i in range(2, n):
        if n % i == 0:
            return False
    return True
"""


def _reference_is_prime(n: int) -> bool:
    """Independent reference oracle (not reusing the seed implementation)
    used only to generate correct expected test-case labels. An earlier
    draft used a hand-written, truncated prime set here instead of this
    function; see the paper's Results section for the bug that caused."""
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


IS_PRIME_CASES = [{"args": (n,), "expected": _reference_is_prime(n)} for n in range(2, 100, 3)]

HAS_DUP_SEED = """def has_duplicates(items):
    n = len(items)
    for i in range(len(items)):
        for j in range(len(items)):
            if i != j and items[i] == items[j]:
                return True
    return False
"""
HAS_DUP_CASES = [
    {"args": ([1, 2, 3, 4, 5],), "expected": False},
    {"args": ([1, 2, 3, 2, 5],), "expected": True},
    {"args": ([],), "expected": False},
    {"args": ([1],), "expected": False},
    {"args": (list(range(30)),), "expected": False},
    {"args": (list(range(30)) + [15],), "expected": True},
]

CONTAINS_SORTED_SEED = """def contains_sorted(items, target):
    for x in items:
        if x == target:
            return True
    return False
"""
_sorted_list = list(range(0, 200, 2))
CONTAINS_SORTED_CASES = [
    {"args": (_sorted_list, 0), "expected": True},
    {"args": (_sorted_list, 198), "expected": True},
    {"args": (_sorted_list, 100), "expected": True},
    {"args": (_sorted_list, 101), "expected": False},
    {"args": (_sorted_list, -2), "expected": False},
    {"args": (_sorted_list, 300), "expected": False},
]

CONCAT_SEED = """def concat_strings(parts):
    result = ""
    for p in parts:
        result += p
    return result
"""
CONCAT_CASES = [
    {"args": (["a", "b", "c"],), "expected": "abc"},
    {"args": ([],), "expected": ""},
    {"args": (["hello", " ", "world"],), "expected": "hello world"},
    {"args": (["x"] * 50,), "expected": "x" * 50},
]

BENCHMARKS: List[Benchmark] = [
    Benchmark("fibonacci", "fib", FIB_SEED, FIB_CASES,
              "naive exponential recursion -> memoization (functools.lru_cache)"),
    Benchmark("is_prime", "is_prime", IS_PRIME_SEED, IS_PRIME_CASES,
              "trial division to n-1 -> trial division to sqrt(n)"),
    Benchmark("has_duplicates", "has_duplicates", HAS_DUP_SEED, HAS_DUP_CASES,
              "O(n^2) nested-loop pairwise comparison -> O(n) hash-set membership"),
    Benchmark("contains_sorted", "contains_sorted", CONTAINS_SORTED_SEED, CONTAINS_SORTED_CASES,
              "O(n) linear scan -> O(log n) binary search (bisect) exploiting sortedness"),
    Benchmark("concat_strings", "concat_strings", CONCAT_SEED, CONCAT_CASES,
              "O(n^2) repeated string += concatenation -> O(n) str.join"),
]
