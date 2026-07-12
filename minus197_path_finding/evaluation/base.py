"""
evaluation/base.py
-------------------
Shared contract for all path evaluators.

Every evaluator receives the query the caller made (start_node_id,
destination_node_id) alongside the PathResult, since some checks — like
"does the path actually start where it was asked to" — need to compare
the result against the original request, not just inspect the result
in isolation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict

from shared.types import PathResult


@dataclass
class EvaluationResult:
    """Outcome of one Evaluator run against one PathResult."""
    name:    str
    passed:  bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


class Evaluator(ABC):
    """Base class for a single evaluation check on a PathResult."""

    name: str = "evaluator"

    @abstractmethod
    def evaluate(
        self,
        result:               PathResult,
        start_node_id:        str,
        destination_node_id:  str,
    ) -> EvaluationResult:
        """Run this check and return its outcome."""
        raise NotImplementedError
