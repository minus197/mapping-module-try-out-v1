"""
evaluation  —  Post-hoc checks on a PathResult produced by PathfindingEngine.

Each check is an Evaluator: given a PathResult (plus the node_ids the caller
asked for), it returns one EvaluationResult. EvaluationRunner runs a list of
evaluators over one PathResult and collects their results.

    from evaluation import EvaluationRunner, EndpointEvaluator

    runner = EvaluationRunner([EndpointEvaluator()])
    report = runner.run(result, start_node_id="ZONE-A", destination_node_id="ZONE-B")
    report.all_passed()   # bool
"""

from evaluation.base import Evaluator, EvaluationResult
from evaluation.endpoint_evaluator import EndpointEvaluator
from evaluation.runner import EvaluationReport, EvaluationRunner

__all__ = [
    "Evaluator",
    "EvaluationResult",
    "EndpointEvaluator",
    "EvaluationReport",
    "EvaluationRunner",
]
