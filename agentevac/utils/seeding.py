"""Deterministic seed derivation for AgentEvac.

A single user-facing ``master_seed`` is split into independent named sub-streams
via BLAKE2b-8, so changing one stream's algorithm cannot perturb another's draws
and reproducibility does not depend on ``PYTHONHASHSEED``.

Stream label conventions used in this codebase:

    ("sumo",)                                  -> SUMO --seed
    ("python_global",)                         -> module-level random.seed
    ("agent_profile", agent_id)                -> per-agent psychological profile
    ("rule_policy", agent_id, decision_round)  -> rule-based softmax sampling
    ("info_noise", agent_id, decision_round)   -> environment / social signal noise
    ("llm", agent_id, decision_round)          -> OpenAI seed= parameter
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Tuple

__all__ = ["derive_seed", "make_rng", "SeedBook", "HASH_ALGO"]

HASH_ALGO = "blake2b-8"
_DIGEST_BYTES = 8  # 64-bit unsigned int output


def _encode(label: Any) -> bytes:
    """Encode a single label component to bytes for hashing.

    Strings, ints, and anything ``str()``-able are accepted.  ``None`` raises,
    because silently coalescing None to an empty string would let two
    semantically different streams collide.
    """
    if label is None:
        raise ValueError("seed label component must not be None")
    if isinstance(label, bytes):
        return label
    return str(label).encode("utf-8")


def derive_seed(master_seed: int, *labels: Any) -> int:
    """Deterministically derive a 64-bit sub-seed from ``master_seed`` and labels.

    Stable across processes, Python versions, and platforms (does not rely on
    the randomized built-in ``hash()``).  The same ``(master_seed, labels)``
    tuple always returns the same int.

    Args:
        master_seed: The user-facing master seed for the run.
        *labels: Stream identifier components.  Convention is
            ``(stream_name, *entity_ids)`` -- e.g. ``("agent_profile", vid)``.

    Returns:
        A non-negative 64-bit int suitable for seeding ``random.Random`` or
        passing to ``numpy.random.default_rng``.
    """
    h = hashlib.blake2b(digest_size=_DIGEST_BYTES)
    h.update(_encode(int(master_seed)))
    for lab in labels:
        h.update(b"\x1f")  # ASCII unit separator -- avoids "a|b" vs "ab" collisions
        h.update(_encode(lab))
    return int.from_bytes(h.digest(), "big", signed=False)


def make_rng(master_seed: int, *labels: Any) -> random.Random:
    """Return a fresh ``random.Random`` seeded by ``derive_seed(master_seed, *labels)``."""
    return random.Random(derive_seed(master_seed, *labels))


@dataclass
class SeedBook:
    """Manifest entry recording how a run was seeded.

    Written into ``run_params.json`` so any run is replayable from the master
    seed alone (assuming no per-stream env overrides).
    """

    master_seed: int
    sumo_seed: int
    python_seed: int
    hash_algo: str = HASH_ALGO
    stream_labels: List[str] = field(default_factory=lambda: [
        "sumo",
        "python_global",
        "agent_profile",
        "rule_policy",
        "info_noise",
        "llm",
    ])
    overrides: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
