"""
tests/test_feedback_json.py
---------------------------
Covers to_feedback_json() — the pathfinding → feedback-module boundary.

This serialiser previously discarded most of the PathStep contract: the final
movement row was consumed as the terminal stop's anchor (hiding the door on
door-terminated routes), turn bands collapsed to a bare left/right, distances
were rounded to whole metres, and bearing/instruction/edge attributes were
never emitted at all. These tests pin the repaired contract.
"""

from __future__ import annotations

import math

from pathfinding.engine import to_feedback_json
from shared.types import NavigationEdge, NavigationNode, PathResult, PathStep


# ── Builders ──────────────────────────────────────────────────────────────────

def _node(node_id, x, y, node_type="junction", label="", tags=None):
    return NavigationNode(
        node_id=node_id,
        label=label or node_id,
        position=(x, y),
        node_type=node_type,
        zone_id=None,
        tags=tags or {},
    )


def _edge(src, tgt, dist, shore_linable=True, safety=0.9, landmark=0.25):
    return NavigationEdge(
        edge_id=f"E-{src}-{tgt}",
        source_id=src,
        target_id=tgt,
        distance=dist,
        shore_linable=shore_linable,
        safety_score=safety,
        landmark_score=landmark,
    )


def _step(frm, to, instruction, dist=None):
    dx = to.position[0] - frm.position[0]
    dy = to.position[1] - frm.position[1]
    d = dist if dist is not None else math.hypot(dx, dy)
    return PathStep(
        from_node=frm,
        to_node=to,
        edge=_edge(frm.node_id, to.node_id, d),
        bearing=math.degrees(math.atan2(dx, dy)) % 360,
        distance=d,
        instruction=instruction,
    )


def _result(steps):
    return PathResult(
        found=True,
        start_node=steps[0].from_node,
        destination_node=steps[-1].to_node,
        steps=steps,
    )


def _door_route():
    """… → PATH-0126 → DOOR → CENTROID — the ODEL→Popeyes tail shape."""
    p127 = _node("PATH-0127", 52.002, -32.0692)
    p126 = _node("PATH-0126", 58.002, -32.0692)
    door = _node("FEAT-0Nrd", 55.32, -33.75, node_type="door", label="Popeyes entrance")
    zone = _node("ZONE-233b", 55.5019, -39.5536, node_type="zone_centroid",
                 label="pop eyes", tags={"admin_label": "Popeyes"})
    return _result([
        _step(p127, p126, "Continue straight. Walk 6 m."),
        _step(p126, door, "Turn right. Walk 3 m. You will reach Popeyes entrance."),
        _step(door, zone, "Bear left. Walk 6 m."),
    ])


# ── The door row ──────────────────────────────────────────────────────────────

def test_door_appears_as_its_own_row():
    """The regression this fix exists for: the door must not be swallowed."""
    actions = to_feedback_json(_door_route(), as_list=True)
    node_ids = [a["node_id"] for a in actions]
    assert "FEAT-0Nrd" in node_ids, (
        "door row consumed as the stop anchor — the v1/v3 defect has returned"
    )


def test_stop_is_appended_not_substituted():
    route = _door_route()
    actions = to_feedback_json(route, as_list=True)
    # 3 movement rows + 1 appended stop
    assert len(actions) == len(route.steps) + 1
    assert [a["action"] for a in actions][-1] == "stop"
    assert all(a["action"] != "stop" for a in actions[:-1])


def test_every_movement_step_is_emitted():
    route = _door_route()
    actions = to_feedback_json(route, as_list=True)
    movement = [a for a in actions if a["action"] != "stop"]
    assert [a["node_id"] for a in movement] == [
        s.from_node.node_id for s in route.steps
    ]


# ── Turn bands (were collapsed to sign-of-angle) ──────────────────────────────

def test_bear_and_turn_are_distinguishable():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 0.0, 10.0)
    c = _node("C", 5.0, 15.0)
    route = _result([
        _step(a, b, "Bear right. Walk 10 m."),
        _step(b, c, "Turn right. Walk 7 m."),
    ])
    actions = to_feedback_json(route, terminal=False, as_list=True)
    assert actions[0]["direction"] == actions[1]["direction"] == "right"
    assert actions[0]["band"] == "bear"
    assert actions[1]["band"] == "turn"


def test_turn_around_keeps_its_band():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 0.0, 5.0)
    route = _result([_step(a, b, "Turn around. Walk 5 m.")])
    actions = to_feedback_json(route, terminal=False, as_list=True)
    assert actions[0]["band"] == "around"
    assert actions[0]["direction"] == "around"


def test_continue_has_no_direction():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 0.0, 5.0)
    route = _result([_step(a, b, "Continue straight. Walk 5 m.")])
    actions = to_feedback_json(route, terminal=False, as_list=True)
    assert actions[0]["action"] == "continue"
    assert "direction" not in actions[0]


# ── Float distances (were rounded to whole metres) ────────────────────────────

def test_distance_is_not_rounded_to_whole_metres():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 0.0, 5.65)
    route = _result([_step(a, b, "Continue straight. Walk 6 m.")])
    actions = to_feedback_json(route, terminal=False, as_list=True)
    assert actions[0]["distance"] == 5.65


def test_corridor_leg_does_not_overstate():
    """Nine 5.65 m hops are 50.85 m, not 9 x 6 = 54 m."""
    nodes = [_node(f"P{i}", 0.0, i * 5.65) for i in range(10)]
    steps = [_step(nodes[i], nodes[i + 1], "Continue straight. Walk 6 m.")
             for i in range(9)]
    actions = to_feedback_json(_result(steps), terminal=False, as_list=True)
    total = sum(a["distance"] for a in actions)
    assert abs(total - 50.85) < 0.01


# ── PathStep contract fields (were absent entirely) ───────────────────────────

def test_bearing_and_instruction_are_emitted():
    route = _door_route()
    actions = to_feedback_json(route, as_list=True)
    for a, s in zip(actions, route.steps):
        assert abs(a["bearing"] - round(s.bearing, 2)) < 0.01
        assert a["instruction"] == s.instruction


def test_edge_attributes_are_emitted():
    """Without these no ROUTE_QUALITY_FACTORS metric can be computed."""
    route = _door_route()
    actions = to_feedback_json(route, as_list=True)
    movement = [a for a in actions if a["action"] != "stop"]
    for a in movement:
        assert "edge_id" in a
        assert isinstance(a["shore_linable"], bool)
        assert "safety_score" in a
        assert "landmark_score" in a


def test_to_node_id_lets_hops_be_reconstructed():
    """Consecutive rows are not consecutive hops without this."""
    route = _door_route()
    actions = to_feedback_json(route, as_list=True)
    movement = [a for a in actions if a["action"] != "stop"]
    for a, s in zip(movement, route.steps):
        assert a["to_node_id"] == s.to_node.node_id


# ── Arrival phrasing ──────────────────────────────────────────────────────────

def test_stop_names_a_side():
    actions = to_feedback_json(_door_route(), as_list=True)
    stop = actions[-1]
    assert stop["side"] in ("left", "right", "ahead")
    assert stop["landmark"] == "Popeyes"


def test_side_flips_with_approach_geometry():
    """Both signs of the cross product, per LAST_MILE_FIX 1."""
    approach_from = _node("A", 0.0, 0.0)
    at = _node("B", 0.0, 10.0)            # travelling due north
    left_dest = _node("L", -5.0, 15.0, tags={"admin_label": "Left shop"})
    right_dest = _node("R", 5.0, 15.0, tags={"admin_label": "Right shop"})

    left = _result([
        _step(approach_from, at, "Continue straight. Walk 10 m."),
        _step(at, left_dest, "Bear left. Walk 7 m."),
    ])
    right = _result([
        _step(approach_from, at, "Continue straight. Walk 10 m."),
        _step(at, right_dest, "Bear right. Walk 7 m."),
    ])

    assert to_feedback_json(left, as_list=True)[-1]["side"] == "left"
    assert to_feedback_json(right, as_list=True)[-1]["side"] == "right"


def test_side_is_ahead_when_collinear():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 0.0, 10.0)
    c = _node("C", 0.0, 20.0, tags={"admin_label": "Straight on"})
    route = _result([
        _step(a, b, "Continue straight. Walk 10 m."),
        _step(b, c, "Continue straight. Walk 10 m."),
    ])
    assert to_feedback_json(route, as_list=True)[-1]["side"] == "ahead"


def test_single_step_route_has_no_heading_to_judge_side_from():
    a = _node("A", 0.0, 0.0)
    b = _node("B", 3.0, 4.0, tags={"admin_label": "Only stop"})
    route = _result([_step(a, b, "Continue straight. Walk 5 m.")])
    assert to_feedback_json(route, as_list=True)[-1]["side"] == "ahead"


# ── Multi-floor leg concatenation ─────────────────────────────────────────────

def test_non_terminal_leg_emits_no_stop():
    route = _door_route()
    actions = to_feedback_json(route, terminal=False, as_list=True)
    assert all(a["action"] != "stop" for a in actions)
    assert len(actions) == len(route.steps)


def test_empty_result_serialises_to_empty():
    assert to_feedback_json(PathResult(found=False), as_list=True) == []
    assert to_feedback_json(PathResult(found=True), as_list=True) == []


def test_json_string_mode_is_valid_json():
    import json
    out = to_feedback_json(_door_route())
    assert isinstance(out, str)
    assert len(json.loads(out)) == 4
