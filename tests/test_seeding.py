"""Tests for the master-seed derivation and integration points.

These tests pin the determinism and master-keying properties that the
``agentevac/utils/seeding.py`` refactor introduced.  In particular they guard
against two specific regressions that motivated the refactor:

1.  ``hash(agent_id)`` is randomized per process unless ``PYTHONHASHSEED`` is
    set, so the legacy seed scheme was non-reproducible across runs.  The
    cross-process determinism test catches a revert to that scheme.

2.  Per-agent profile sampling and rule-based softmax sampling were not coupled
    to the master seed, so multi-seed sweeps only varied SUMO traffic noise --
    silently underestimating run-to-run variance.  The
    ``varies_with_master_seed`` tests catch a revert to that behaviour.
"""

from __future__ import annotations

import statistics
import subprocess
import sys

import pytest

from agentevac.agents.agent_state import sample_profile_params
from agentevac.agents.information_model import sample_environment_signal
from agentevac.utils.seeding import HASH_ALGO, SeedBook, derive_seed, make_rng


class TestDeriveSeed:
    def test_in_process_stable(self):
        a = derive_seed(42, "agent_profile", "veh_001")
        b = derive_seed(42, "agent_profile", "veh_001")
        assert a == b

    def test_cross_process_stable(self):
        """Spawning a fresh interpreter must yield the same derived seed.

        This is the regression test for the ``hash(agent_id)`` bug: built-in
        ``hash()`` is salted per process, so the legacy scheme would fail this
        test almost every run.  ``derive_seed`` uses BLAKE2b and must not.
        """
        code = (
            "from agentevac.utils.seeding import derive_seed; "
            "print(derive_seed(42, 'agent_profile', 'veh_001'))"
        )
        out1 = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        ).stdout.strip()
        out2 = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True,
        ).stdout.strip()
        assert out1 == out2
        assert out1.isdigit() and int(out1) > 0

    def test_label_separator_avoids_collision(self):
        """The ``\\x1f`` unit separator must prevent prefix-style collisions."""
        # Without the separator, both of these would hash the same bytes "ab|c".
        a = derive_seed(42, "ab", "c")
        b = derive_seed(42, "a", "bc")
        assert a != b

    def test_different_master_yields_different_seed(self):
        a = derive_seed(42, "sumo")
        b = derive_seed(43, "sumo")
        assert a != b

    def test_different_stream_label_yields_different_seed(self):
        a = derive_seed(42, "sumo")
        b = derive_seed(42, "python_global")
        assert a != b

    def test_returns_unsigned_64bit_int(self):
        seed = derive_seed(42, "any", "label")
        assert isinstance(seed, int)
        assert 0 <= seed < 2**64

    def test_none_label_rejected(self):
        with pytest.raises(ValueError):
            derive_seed(42, None)


class TestMakeRng:
    def test_reproducible(self):
        r1 = make_rng(42, "rule_policy", "veh_001", 0)
        r2 = make_rng(42, "rule_policy", "veh_001", 0)
        seq1 = [r1.random() for _ in range(8)]
        seq2 = [r2.random() for _ in range(8)]
        assert seq1 == seq2

    def test_master_keyed(self):
        r42 = make_rng(42, "rule_policy", "veh_001", 0)
        r43 = make_rng(43, "rule_policy", "veh_001", 0)
        assert r42.random() != r43.random()

    def test_round_keyed(self):
        """Different decision rounds must yield independent draws (no aliasing)."""
        r0 = make_rng(42, "rule_policy", "veh_001", 0)
        r1 = make_rng(42, "rule_policy", "veh_001", 1)
        assert r0.random() != r1.random()

    def test_streams_independent(self):
        """Different stream labels with same entity yield independent draws.

        Guards against a regression where two semantically distinct uses
        accidentally share an RNG state.
        """
        prof = make_rng(42, "agent_profile", "veh_001")
        info = make_rng(42, "info_noise", "veh_001", 0)
        rule = make_rng(42, "rule_policy", "veh_001", 0)
        # All three first draws should be distinct.
        draws = {prof.random(), info.random(), rule.random()}
        assert len(draws) == 3


_PROFILE_MEANS = {"theta_r": 0.45, "theta_u": 0.30, "lambda_e": 1.0}
_PROFILE_SPREADS = {"theta_r": 0.10, "theta_u": 0.05, "lambda_e": 0.30}
_PROFILE_BOUNDS = {
    "theta_r": (0.05, 0.95),
    "theta_u": (0.05, 0.95),
    "lambda_e": (0.0, 5.0),
}


class TestAgentProfileMasterKeying:
    def test_reproducible_same_master(self):
        a = sample_profile_params(
            "veh_001", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
            master_seed=42,
        )
        b = sample_profile_params(
            "veh_001", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
            master_seed=42,
        )
        assert a == b

    def test_varies_with_master_seed(self):
        """Regression test for the variance-underestimation bug.

        Pre-refactor, ``sample_profile_params`` used ``random.Random(hash(agent_id))``
        which ignored the master seed entirely, so multi-seed sweeps produced
        identical agent profile populations across all replicates.  This test
        asserts that 200-agent populations at master=42 vs master=43 share zero
        identical theta_r draws -- catching any revert to the old behaviour.
        """
        pop_42 = [
            sample_profile_params(
                f"veh_{i:04d}", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
                master_seed=42,
            )["theta_r"]
            for i in range(200)
        ]
        pop_43 = [
            sample_profile_params(
                f"veh_{i:04d}", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
                master_seed=43,
            )["theta_r"]
            for i in range(200)
        ]
        overlap = sum(1 for x, y in zip(pop_42, pop_43) if x == y)
        # Overlap is theoretically possible by coincidence but should be ~0.
        assert overlap < 5, (
            f"Master seed appears to no longer affect profile draws: "
            f"{overlap}/200 agents share theta_r across masters 42 and 43"
        )
        # Aggregate distributions are still in the same general regime.
        assert abs(statistics.mean(pop_42) - statistics.mean(pop_43)) < 0.05

    def test_default_master_zero_is_deterministic(self):
        """The kw-only default ``master_seed=0`` must still be reproducible.

        This default keeps the existing test_agent_state.py suite passing
        without explicit master arguments.
        """
        a = sample_profile_params(
            "veh_001", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
        )
        b = sample_profile_params(
            "veh_001", _PROFILE_MEANS, _PROFILE_SPREADS, _PROFILE_BOUNDS,
        )
        assert a == b


class TestInfoNoiseStream:
    @staticmethod
    def _signal(master: int, vid: str, round_: int) -> dict:
        return sample_environment_signal(
            agent_id=vid,
            sim_t_s=10.0,
            current_edge="e1",
            current_edge_margin_m=2000.0,
            route_head_min_margin_m=2000.0,
            decision_round=round_,
            sigma_info=200.0,
            rng=make_rng(master, "info_noise", vid, round_),
            distance_ref_m=0.0,
        )

    def test_reproducible(self):
        a = self._signal(42, "veh_001", 0)
        b = self._signal(42, "veh_001", 0)
        assert a["noise_delta_m"] == b["noise_delta_m"]

    def test_master_keyed(self):
        a = self._signal(42, "veh_001", 0)
        b = self._signal(43, "veh_001", 0)
        assert a["noise_delta_m"] != b["noise_delta_m"]

    def test_round_keyed(self):
        a = self._signal(42, "veh_001", 0)
        b = self._signal(42, "veh_001", 1)
        assert a["noise_delta_m"] != b["noise_delta_m"]


class TestSeedBook:
    def test_to_dict_includes_documented_streams(self):
        book = SeedBook(master_seed=42, sumo_seed=1, python_seed=2)
        d = book.to_dict()
        assert d["master_seed"] == 42
        assert d["sumo_seed"] == 1
        assert d["python_seed"] == 2
        assert d["hash_algo"] == HASH_ALGO
        # The named streams the rest of the codebase relies on.
        for label in ("sumo", "python_global", "agent_profile",
                      "rule_policy", "info_noise", "llm"):
            assert label in d["stream_labels"]
        assert d["overrides"] == ()

    def test_overrides_recorded(self):
        book = SeedBook(
            master_seed=42, sumo_seed=99, python_seed=100,
            overrides=("sumo", "python_global"),
        )
        d = book.to_dict()
        assert d["overrides"] == ("sumo", "python_global")
