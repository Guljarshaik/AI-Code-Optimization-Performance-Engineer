"""genome.py — Candidate program representation."""

from __future__ import annotations

import ast
import hashlib
import itertools
import textwrap
from dataclasses import dataclass, field
from typing import List, Optional

_id_counter = itertools.count(1)


@dataclass
class CodeGenome:
    source: str
    generation: int = 0
    parent_ids: List[int] = field(default_factory=list)
    origin_operator: str = "seed"
    llm_rationale: Optional[str] = None
    genome_id: int = field(default_factory=lambda: next(_id_counter))

    def fingerprint(self) -> str:
        normalized = "\n".join(line.rstrip() for line in self.source.splitlines()).strip()
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:16]

    def is_syntactically_valid(self) -> bool:
        try:
            ast.parse(self.source)
            return True
        except SyntaxError:
            return False

    def statement_blocks(self) -> List[str]:
        try:
            tree = ast.parse(self.source)
        except SyntaxError:
            return [self.source]
        blocks = []
        lines = self.source.splitlines()
        for node in ast.iter_child_nodes(tree):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                block = "\n".join(lines[node.lineno - 1: node.end_lineno])
                blocks.append(block)
        return blocks or [self.source]

    def child(self, new_source: str, operator: str, rationale: Optional[str] = None) -> "CodeGenome":
        return CodeGenome(
            source=new_source,
            generation=self.generation + 1,
            parent_ids=[self.genome_id],
            origin_operator=operator,
            llm_rationale=rationale,
        )

    def pretty(self) -> str:
        return textwrap.dedent(self.source).strip()
