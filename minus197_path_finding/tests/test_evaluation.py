"""
tests/test_evaluation.py
--------------------------
Tests for the evaluation package (EndpointEvaluator + EvaluationRunner)
against real PathResults produced by PathfindingEngine on tiny_graph.
"""

from shared.types import PathResult
from pathfinding.engine import PathfindingEngine
from evaluation import EndpointEvaluator, EvaluationRunner


def test_endpoint_evaluator_passes_on_valid_path(tiny_graph, fake_wall_checker):
    engine = PathfindingEngine(tiny_graph, fake_wall_checker)
    result = engine.find_path("SKE-START", "ZONE-DEST")

    evaluator = EndpointEvaluator()
    outcome = evaluator.evaluate(result, "SKE-START", "ZONE-DEST")

    assert outcome.passed
    assert outcome.name == "endpoint_check"


def test_endpoint_evaluator_passes_on_same_start_and_destination(tiny_graph, fake_wall_checker):
    engine = PathfindingEngine(tiny_graph, fake_wall_checker)
    result = engine.find_path("SKE-START", "SKE-START")

    outcome = EndpointEvaluator().evaluate(result, "SKE-START", "SKE-START")

    assert outcome.passed
    assert result.steps == []   # same-node query yields no steps by design


def test_endpoint_evaluator_fails_when_path_not_found(tiny_graph, fake_wall_checker):
    engine = PathfindingEngine(tiny_graph, fake_wall_checker)
    result = engine.find_path("SKE-START", "NO-SUCH-NODE")

    outcome = EndpointEvaluator().evaluate(result, "SKE-START", "NO-SUCH-NODE")

    assert not outcome.passed
    assert outcome.details["found"] is False


def test_endpoint_evaluator_fails_when_expected_ids_dont_match_result():
    """
    A hand-built PathResult whose start_node/destination_node don't match
    what the caller asked for should fail, independent of the engine.
    """
    from shared.types import NavigationNode

    fake_result = PathResult(
        found=True,
        start_node=NavigationNode("A", "A", (0.0, 0.0), "junction", None),
        destination_node=NavigationNode("B", "B", (1.0, 0.0), "junction", None),
        steps=[],
    )

    outcome = EndpointEvaluator().evaluate(fake_result, "A", "WRONG-DEST")

    assert not outcome.passed
    assert "WRONG-DEST" in outcome.message


def test_evaluation_runner_aggregates_results(tiny_graph, fake_wall_checker):
    engine = PathfindingEngine(tiny_graph, fake_wall_checker)
    result = engine.find_path("SKE-START", "ZONE-DEST")

    runner = EvaluationRunner([EndpointEvaluator()])
    report = runner.run(result, "SKE-START", "ZONE-DEST")

    assert report.all_passed()
    assert report.failures() == []
    assert len(report.results) == 1


def test_evaluation_runner_reports_failures(tiny_graph, fake_wall_checker):
    engine = PathfindingEngine(tiny_graph, fake_wall_checker)
    result = engine.find_path("SKE-START", "ZONE-DEST")

    runner = EvaluationRunner([EndpointEvaluator()])
    report = runner.run(result, "SKE-START", "WRONG-DEST")

    assert not report.all_passed()
    assert len(report.failures()) == 1
