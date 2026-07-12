"""
evaluation/runner.py
----------------------
Runs a list of Evaluators against one PathResult and collects an
EvaluationReport. Add new checks by constructing EvaluationRunner with
more Evaluator instances — nothing here needs to change as the set of
checks grows.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from evaluation.base import EvaluationResult, Evaluator
from shared.types import PathResult


@dataclass
class EvaluationReport:
    """All EvaluationResults produced for one PathResult."""
    results: List[EvaluationResult] = field(default_factory=list)

    def all_passed(self) -> bool:
        return all(r.passed for r in self.results)

    def failures(self) -> List[EvaluationResult]:
        return [r for r in self.results if not r.passed]


class EvaluationRunner:
    """Runs a fixed set of Evaluators against a PathResult."""

    def __init__(self, evaluators: List[Evaluator]) -> None:
        self._evaluators = evaluators

    def run(
        self,
        result:               PathResult,
        start_node_id:        str,
        destination_node_id:  str,
    ) -> EvaluationReport:
        results = [
            evaluator.evaluate(result, start_node_id, destination_node_id)
            for evaluator in self._evaluators
        ]
        return EvaluationReport(results=results)
