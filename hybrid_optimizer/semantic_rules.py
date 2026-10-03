"""
semantic_rules.py — Deterministic rule-based semantic operator.

DISCLOSURE: no live LLM API credentials were available in the execution
environment used to produce this paper's measured results. This module is
a transparent, deterministic, pattern-matching stand-in for the LLM
Reasoning Module — NOT a language model. It implements the LLMBackend
interface so the orchestrator's control flow (selective routing,
repair-on-failure, critique) is exercised exactly as it would be with a
real model. Each rule recognizes a specific naive-implementation pattern
and proposes the corresponding targeted rewrite once per lineage.
"""

from __future__ import annotations

import json
import re
from typing import Callable, Optional

from .llm_agent import LLMBackend

MARKER = "# semantic-op-applied"


def _already_applied(source: str) -> bool:
    return MARKER in source


def _tag(source: str) -> str:
    return source.rstrip() + "\n" + MARKER + "\n"


def _rule_memoize_fib(source: str) -> Optional[str]:
    if "def fib(" not in source or "functools" in source:
        return None
    if not re.search(r"return\s+fib\(", source):
        return None
    new_source = "import functools\n\n\n@functools.lru_cache(maxsize=None)\n" + source
    return _tag(new_source)


def _rule_sqrt_bound_prime(source: str) -> Optional[str]:
    if "def is_prime(" not in source:
        return None
    m = re.search(r"for\s+(\w+)\s+in\s+range\(2,\s*n\s*\)", source)
    if not m:
        return None
    var = m.group(1)
    new_source = source.replace(
        f"for {var} in range(2, n)",
        f"for {var} in range(2, int(n ** 0.5) + 1)",
    )
    return _tag(new_source)


def _rule_set_based_duplicates(source: str) -> Optional[str]:
    if "def has_duplicates(" not in source:
        return None
    if "set(" in source:
        return None
    if not re.search(r"for\s+\w+\s+in\s+range\(len\(", source):
        return None
    new_source = (
        "def has_duplicates(items):\n"
        "    seen = set()\n"
        "    for x in items:\n"
        "        if x in seen:\n"
        "            return True\n"
        "        seen.add(x)\n"
        "    return False\n"
    )
    return _tag(new_source)


def _rule_binary_search(source: str) -> Optional[str]:
    if "def contains_sorted(" not in source:
        return None
    if "bisect" in source:
        return None
    if not re.search(r"for\s+\w+\s+in\s+items", source):
        return None
    new_source = (
        "import bisect\n\n\n"
        "def contains_sorted(items, target):\n"
        "    i = bisect.bisect_left(items, target)\n"
        "    return i < len(items) and items[i] == target\n"
    )
    return _tag(new_source)


def _rule_join_concat(source: str) -> Optional[str]:
    if "def concat_strings(" not in source:
        return None
    if ".join(" in source:
        return None
    if "+=" not in source:
        return None
    new_source = (
        "def concat_strings(parts):\n"
        "    return ''.join(parts)\n"
    )
    return _tag(new_source)


RULES: list[Callable[[str], Optional[str]]] = [
    _rule_memoize_fib,
    _rule_sqrt_bound_prime,
    _rule_set_based_duplicates,
    _rule_binary_search,
    _rule_join_concat,
]


class RuleBasedSemanticBackend(LLMBackend):
    def __init__(self):
        self.calls_made = 0
        self.rules_fired = 0

    def complete(self, system: str, user: str) -> str:
        self.calls_made += 1
        m = re.search(r"CURRENT_SOURCE:\n(.*?)\nFITNESS_REPORT:", user, re.S)
        source = m.group(1).strip("\n") if m else ""

        if _already_applied(source):
            return json.dumps({
                "source": source,
                "rationale": "No further targeted rewrite available from the rule library; candidate returned unchanged.",
            })

        for rule in RULES:
            result = rule(source)
            if result is not None:
                self.rules_fired += 1
                return json.dumps({
                    "source": result,
                    "rationale": f"Applied targeted rewrite: {rule.__name__.replace('_rule_', '').replace('_', ' ')}.",
                })

        return json.dumps({
            "source": source,
            "rationale": "No applicable rewrite pattern recognized for this candidate.",
        })
