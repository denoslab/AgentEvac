"""Unit tests for agentevac.agents.messaging — spatial broadcast filtering."""

import pytest

from agentevac.agents.messaging import AgentMessagingBus, OutboxMessage


def _bus(comm_radius_m: float = 0.0, **kwargs) -> AgentMessagingBus:
    """Create a bus with sensible defaults for testing."""
    defaults = dict(
        enabled=True,
        max_message_chars=400,
        max_inbox_messages=20,
        max_sends_per_agent_per_round=3,
        max_broadcasts_per_round=20,
        ttl_rounds=10,
        comm_radius_m=comm_radius_m,
    )
    defaults.update(kwargs)
    return AgentMessagingBus(**defaults)


def _broadcast(text: str = "hello") -> list:
    return [OutboxMessage(to="*", message=text)]


def _direct(to: str, text: str = "hello") -> list:
    return [OutboxMessage(to=to, message=text)]


# ---------------------------------------------------------------------------
# Radius = 0 (disabled) — original broadcast-to-all behaviour
# ---------------------------------------------------------------------------

class TestRadiusDisabled:
    def test_broadcast_reaches_all_agents(self):
        bus = _bus(comm_radius_m=0)
        agents = ["A", "B", "C", "D", "E"]
        bus.begin_round(1, agents)
        bus.queue_outbox("A", _broadcast("fire nearby"))
        bus.begin_round(2, agents)

        for agent in ["B", "C", "D", "E"]:
            inbox = bus.get_inbox(agent)
            assert len(inbox) == 1
            assert inbox[0]["message"] == "fire nearby"

        assert bus.get_inbox("A") == []  # sender excluded

    def test_broadcast_reaches_all_even_with_positions(self):
        """When radius=0, positions are irrelevant — all agents receive."""
        bus = _bus(comm_radius_m=0)
        agents = ["A", "B", "C"]
        positions = {"A": (0.0, 0.0), "B": (99999.0, 99999.0), "C": (50000.0, 50000.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1
        assert len(bus.get_inbox("C")) == 1


# ---------------------------------------------------------------------------
# Spatial filtering with finite radius
# ---------------------------------------------------------------------------

class TestSpatialFiltering:
    def test_nearby_agents_receive_broadcast(self):
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B", "C"]
        # B is 500m away (within range), C is 2000m away (out of range)
        positions = {"A": (0.0, 0.0), "B": (500.0, 0.0), "C": (2000.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("fire"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1
        assert len(bus.get_inbox("C")) == 0

    def test_agent_exactly_at_boundary_receives(self):
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B"]
        # B is exactly 1000m away — should receive (<=, not <)
        positions = {"A": (0.0, 0.0), "B": (1000.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1

    def test_agent_just_beyond_boundary_excluded(self):
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B"]
        positions = {"A": (0.0, 0.0), "B": (1001.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 0

    def test_diagonal_distance(self):
        """707m on each axis = ~1000m diagonal."""
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B", "C"]
        # B is ~707m diagonal (within 1000m), C is ~1414m diagonal (out of range)
        positions = {"A": (0.0, 0.0), "B": (500.0, 500.0), "C": (1000.0, 1000.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1  # ~707m < 1000m
        assert len(bus.get_inbox("C")) == 0  # ~1414m > 1000m

    def test_multiple_senders_different_reach(self):
        bus = _bus(comm_radius_m=500)
        agents = ["A", "B", "C"]
        # A at origin, B at 300m, C at 800m
        positions = {"A": (0.0, 0.0), "B": (300.0, 0.0), "C": (800.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("from A"))
        bus.queue_outbox("C", _broadcast("from C"))
        bus.begin_round(2, agents, positions=positions)

        # B is within 500m of both A and C
        inbox_b = bus.get_inbox("B")
        assert len(inbox_b) == 2

        # A is 800m from C — out of range
        inbox_a = bus.get_inbox("A")
        assert len(inbox_a) == 0

        # C is 800m from A — out of range
        inbox_c = bus.get_inbox("C")
        assert len(inbox_c) == 0


# ---------------------------------------------------------------------------
# Direct messages bypass spatial filter
# ---------------------------------------------------------------------------

class TestDirectMessagesUnfiltered:
    def test_direct_message_delivered_regardless_of_distance(self):
        bus = _bus(comm_radius_m=100)
        agents = ["A", "B"]
        positions = {"A": (0.0, 0.0), "B": (50000.0, 0.0)}  # 50km apart
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _direct("B", "direct msg"))
        bus.begin_round(2, agents, positions=positions)

        inbox = bus.get_inbox("B")
        assert len(inbox) == 1
        assert inbox[0]["kind"] == "direct"

    def test_direct_and_broadcast_from_same_sender(self):
        bus = _bus(comm_radius_m=500)
        agents = ["A", "B", "C"]
        positions = {"A": (0.0, 0.0), "B": (300.0, 0.0), "C": (5000.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", [
            OutboxMessage(to="*", message="broadcast"),
            OutboxMessage(to="C", message="direct to C"),
        ])
        bus.begin_round(2, agents, positions=positions)

        # B is nearby — gets broadcast
        assert len(bus.get_inbox("B")) == 1
        # C is far — no broadcast, but gets direct message
        inbox_c = bus.get_inbox("C")
        assert len(inbox_c) == 1
        assert inbox_c[0]["kind"] == "direct"


# ---------------------------------------------------------------------------
# Missing positions — fail-open behaviour
# ---------------------------------------------------------------------------

class TestMissingPositions:
    def test_sender_without_position_broadcasts_to_all(self):
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B", "C"]
        # A has no position entry
        positions = {"B": (0.0, 0.0), "C": (5000.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        # Both receive because sender position unknown — fail open
        assert len(bus.get_inbox("B")) == 1
        assert len(bus.get_inbox("C")) == 1

    def test_target_without_position_receives(self):
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B", "C"]
        # C has no position entry
        positions = {"A": (0.0, 0.0), "B": (500.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1
        # C has no position — fail open, receives message
        assert len(bus.get_inbox("C")) == 1

    def test_no_positions_dict_at_all(self):
        """begin_round without positions arg — all broadcasts go through."""
        bus = _bus(comm_radius_m=1000)
        agents = ["A", "B", "C"]
        bus.begin_round(1, agents)  # no positions
        bus.queue_outbox("A", _broadcast("test"))
        bus.begin_round(2, agents)

        assert len(bus.get_inbox("B")) == 1
        assert len(bus.get_inbox("C")) == 1


# ---------------------------------------------------------------------------
# Existing caps still enforced alongside spatial filter
# ---------------------------------------------------------------------------

class TestCapsWithSpatialFilter:
    def test_per_agent_send_cap_enforced(self):
        bus = _bus(comm_radius_m=5000, max_sends_per_agent_per_round=2)
        agents = ["A", "B"]
        positions = {"A": (0.0, 0.0), "B": (100.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", [
            OutboxMessage(to="*", message="msg1"),
            OutboxMessage(to="*", message="msg2"),
            OutboxMessage(to="*", message="msg3"),  # should be dropped (cap=2)
        ])
        bus.begin_round(2, agents, positions=positions)

        inbox = bus.get_inbox("B")
        assert len(inbox) == 2

    def test_global_broadcast_cap_enforced(self):
        bus = _bus(comm_radius_m=5000, max_broadcasts_per_round=1,
                   max_sends_per_agent_per_round=5)
        agents = ["A", "B", "C"]
        positions = {"A": (0.0, 0.0), "B": (100.0, 0.0), "C": (200.0, 0.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", [
            OutboxMessage(to="*", message="first"),
            OutboxMessage(to="*", message="second"),  # dropped: global cap=1
        ])
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1
        assert bus.get_inbox("B")[0]["message"] == "first"


# ---------------------------------------------------------------------------
# Spawn-edge co-location (same position)
# ---------------------------------------------------------------------------

class TestSamePosition:
    def test_agents_at_same_position_communicate(self):
        """Agents on the same spawn edge have the same midpoint — distance 0."""
        bus = _bus(comm_radius_m=100)
        agents = ["A", "B", "C"]
        positions = {"A": (500.0, 200.0), "B": (500.0, 200.0), "C": (5000.0, 5000.0)}
        bus.begin_round(1, agents, positions=positions)
        bus.queue_outbox("A", _broadcast("local"))
        bus.begin_round(2, agents, positions=positions)

        assert len(bus.get_inbox("B")) == 1
        assert len(bus.get_inbox("C")) == 0
