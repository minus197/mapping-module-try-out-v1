"""
evaluation/endpoint_evaluator.py
----------------------------------
Checks that a found PathResult actually starts at the requested start
node and ends at the requested destination node.

Three things are verified:
  1. result.found is True
  2. result.start_node.node_id / result.destination_node.node_id match
     the ids the caller asked for
  3. the first and last steps' from_node / to_node are consistent with
     those same ids (guards against a step-building bug producing a
     PathResult whose top-level start/destination fields don't match
     what the actual walked route begins/ends at)

A start_node_id == destination_node_id query yields an empty steps list
by design (PathfindingEngine.find_path, "start == dest" branch) — that
case is treated as valid and checked via the top-level nodes only.
"""

from __future__ import annotations

from evaluation.base import EvaluationResult, Evaluator
from shared.types import PathResult


class EndpointEvaluator(Evaluator):
    """Verifies a PathResult starts and ends at the requested node ids."""

    name = "endpoint_check"

    def evaluate(
        self,
        result:               PathResult,
        start_node_id:        str,
        destination_node_id:  str,
    ) -> EvaluationResult:
        if not result.found:
            return EvaluationResult(
                name=self.name,
                passed=False,
                message="No path was found; nothing to verify.",
                details={"found": False},
            )

        if result.start_node is None or result.destination_node is None:
            return EvaluationResult(
                name=self.name,
                passed=False,
                message="Path was found but start_node/destination_node is missing.",
                details={
                    "start_node":       result.start_node,
                    "destination_node": result.destination_node,
                },
            )

        actual_start = result.start_node.node_id
        actual_dest  = result.destination_node.node_id

        errors = []
        if actual_start != start_node_id:
            errors.append(
                f"start_node is '{actual_start}', expected '{start_node_id}'"
            )
        if actual_dest != destination_node_id:
            errors.append(
                f"destination_node is '{actual_dest}', expected '{destination_node_id}'"
            )

        if result.steps:
            first_step_from = result.steps[0].from_node.node_id
            last_step_to    = result.steps[-1].to_node.node_id
            if first_step_from != start_node_id:
                errors.append(
                    f"first step starts from '{first_step_from}', "
                    f"expected '{start_node_id}'"
                )
            if last_step_to != destination_node_id:
                errors.append(
                    f"last step ends at '{last_step_to}', "
                    f"expected '{destination_node_id}'"
                )

        if errors:
            return EvaluationResult(
                name=self.name,
                passed=False,
                message="; ".join(errors),
                details={
                    "expected_start": start_node_id,
                    "expected_dest":  destination_node_id,
                    "actual_start":   actual_start,
                    "actual_dest":    actual_dest,
                },
            )

        return EvaluationResult(
            name=self.name,
            passed=True,
            message="Path starts and ends at the requested nodes.",
            details={"start": actual_start, "destination": actual_dest},
        )
