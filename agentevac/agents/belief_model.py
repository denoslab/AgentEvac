"""Bayesian belief-update pipeline for wildfire hazard assessment.

Each agent maintains a probability distribution over three hazard states:
    - ``p_safe``   : fire is far enough that the agent's current position is safe.
    - ``p_risky``  : fire is close enough to warrant caution but not immediate danger.
    - ``p_danger`` : fire is imminent; departure is likely warranted.

The update pipeline (exposed via ``update_agent_belief``) runs once per decision period:
    1. Map the observed margin (meters from fire edge) to a prior triplet
       (``categorize_hazard_state``).
    2. If social messages are present, fuse the environment-derived prior with a
       social-signal belief weighted by ``theta_trust`` (``fuse_env_and_social_beliefs``).
    3. Apply temporal smoothing using a fixed inertia factor to avoid belief whiplash
       (``smooth_belief``).
    4. Compute Shannon entropy to quantify uncertainty (``compute_belief_entropy``).
    5. Normalize entropy to [0, 1] and bucket it into Low / Medium / High
       (``normalize_entropy``, ``bucket_uncertainty``).
"""

import math
from typing import Any, Dict


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp ``value`` to the closed interval [``lo``, ``hi``].

    Args:
        value: The value to clamp.
        lo: Lower bound (inclusive).
        hi: Upper bound (inclusive).

    Returns:
        ``value`` clipped to [``lo``, ``hi``].
    """
    return max(lo, min(hi, float(value)))


def _normalize_triplet(belief: Dict[str, float]) -> Dict[str, float]:
    """L1-normalize a {p_safe, p_risky, p_danger} belief dict to sum to 1.

    If the total probability mass is zero or negative (degenerate input), returns a
    uniform distribution so downstream code always receives a valid probability vector.

    Args:
        belief: Dict with keys "p_safe", "p_risky", "p_danger".

    Returns:
        A new dict with the same keys whose values sum to 1.0.
    """
    total = float(
        belief.get("p_safe", 0.0) +
        belief.get("p_risky", 0.0) +
        belief.get("p_danger", 0.0)
    )
    if total <= 0.0:
        return {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}
    return {
        "p_safe": float(belief.get("p_safe", 0.0)) / total,
        "p_risky": float(belief.get("p_risky", 0.0)) / total,
        "p_danger": float(belief.get("p_danger", 0.0)) / total,
    }


def categorize_hazard_state(signal: Dict[str, Any]) -> Dict[str, float]:
    """Map an observed margin (meters from fire edge) to a hazard-state prior.

    Uses ``observed_margin_m`` if available, falling back to ``base_margin_m``.
    If neither is present (e.g., no fire active), returns a uniform prior.

    Threshold rationale:
        - ≤ 0 m    : fire has reached / overtaken the edge  → near-certain danger.
        - ≤ 1200 m : within ember-attack range               → high danger.
        - ≤ 2500 m : smoke / radiant-heat proximity          → risky (watch closely).
        - ≤ 5000 m : fire visible but with buffer            → elevated but manageable.
        - > 5000 m : fire is well clear of the route         → predominantly safe.

    Args:
        signal: Environment signal dict, typically from ``information_model.sample_environment_signal``.

    Returns:
        A normalized {p_safe, p_risky, p_danger} dict representing the categorical prior.
    """
    margin = signal.get("observed_margin_m")
    if margin is None:
        margin = signal.get("base_margin_m")
    if margin is None:
        return {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}

    margin_f = float(margin)
    if margin_f <= 0.0:
        return {"p_safe": 0.02, "p_risky": 0.08, "p_danger": 0.90}
    if margin_f <= 1200.0:
        return {"p_safe": 0.05, "p_risky": 0.20, "p_danger": 0.75}
    if margin_f <= 2500.0:
        return {"p_safe": 0.15, "p_risky": 0.55, "p_danger": 0.30}
    if margin_f <= 5000.0:
        return {"p_safe": 0.35, "p_risky": 0.50, "p_danger": 0.15}
    return {"p_safe": 0.75, "p_risky": 0.20, "p_danger": 0.05}


def fuse_env_and_social_beliefs(
    env_belief: Dict[str, float],
    social_belief: Dict[str, float],
    theta_trust: float,
) -> Dict[str, float]:
    """Fuse an environmental belief with a social-signal belief via weighted average.

    The agent's trust parameter ``theta_trust`` ∈ [0, 1] determines how much weight
    is given to peer messages relative to its own observations:
        fused = (1 - theta_trust) * env_belief + theta_trust * social_belief

    Args:
        env_belief: Belief derived from the agent's own hazard observations.
        social_belief: Belief inferred from neighbor inbox messages.
        theta_trust: Weight given to social signals; 0 = ignore peers, 1 = ignore self.

    Returns:
        A normalized {p_safe, p_risky, p_danger} dict representing the fused belief.
    """
    social_weight = _clamp(theta_trust, 0.0, 1.0)
    env_weight = 1.0 - social_weight
    fused = {
        "p_safe": env_weight * float(env_belief.get("p_safe", 0.0)) + social_weight * float(social_belief.get("p_safe", 0.0)),
        "p_risky": env_weight * float(env_belief.get("p_risky", 0.0)) + social_weight * float(social_belief.get("p_risky", 0.0)),
        "p_danger": env_weight * float(env_belief.get("p_danger", 0.0)) + social_weight * float(social_belief.get("p_danger", 0.0)),
    }
    return _normalize_triplet(fused)


def smooth_belief(
    prev_belief: Dict[str, float],
    new_belief: Dict[str, float],
    inertia: float = 0.35,
) -> Dict[str, float]:
    """Blend the previous belief with a newly computed belief using exponential smoothing.

    Prevents erratic belief flips between decision rounds by retaining a fraction
    (``inertia``) of the prior state:
        smoothed = inertia * prev + (1 - inertia) * new

    A higher inertia value means the agent is slower to update beliefs (more conservative).
    The default of 0.35 balances responsiveness with stability.

    Args:
        prev_belief: The agent's belief from the previous decision round.
        new_belief: The freshly computed belief for the current round.
        inertia: Mixing weight for the previous belief ∈ [0, 0.999].

    Returns:
        A normalized {p_safe, p_risky, p_danger} dict representing the smoothed belief.
    """
    smooth = _clamp(inertia, 0.0, 0.999)
    prev = _normalize_triplet(prev_belief)
    new = _normalize_triplet(new_belief)
    merged = {
        "p_safe": smooth * prev["p_safe"] + (1.0 - smooth) * new["p_safe"],
        "p_risky": smooth * prev["p_risky"] + (1.0 - smooth) * new["p_risky"],
        "p_danger": smooth * prev["p_danger"] + (1.0 - smooth) * new["p_danger"],
    }
    return _normalize_triplet(merged)


def compute_belief_entropy(belief: Dict[str, float]) -> float:
    """Compute the Shannon entropy of a {p_safe, p_risky, p_danger} belief distribution.

    Entropy H = -Σ p_i * log(p_i) over the three states.  The maximum possible value
    for a 3-state uniform distribution is log(3) ≈ 1.099 nats.  A floor of 1e-12 is
    applied to each probability to avoid log(0) singularities.

    Args:
        belief: The belief distribution dict (will be normalized internally).

    Returns:
        Shannon entropy in nats (≥ 0).
    """
    norm = _normalize_triplet(belief)
    total = 0.0
    for key in ("p_safe", "p_risky", "p_danger"):
        p = max(1e-12, float(norm[key]))
        total -= p * math.log(p)
    return total


def normalize_entropy(entropy: float) -> float:
    """Normalize Shannon entropy to the range [0, 1] relative to the 3-state maximum.

    Divides raw entropy by log(3) so that a uniform distribution maps to 1.0 and a
    fully confident belief maps to 0.0.

    Args:
        entropy: Raw Shannon entropy in nats.

    Returns:
        Normalized entropy ∈ [0, 1].
    """
    max_entropy = math.log(3.0)
    if max_entropy <= 0.0:
        return 0.0
    return _clamp(float(entropy) / max_entropy, 0.0, 1.0)


def bucket_uncertainty(entropy_norm: float) -> str:
    """Discretize normalized entropy into a human-readable uncertainty label.

    Thresholds:
        - "Low"    : entropy_norm ≤ 0.33  (agent is fairly confident)
        - "Medium" : entropy_norm ≤ 0.67  (moderate uncertainty)
        - "High"   : entropy_norm >  0.67  (agent is highly uncertain)

    The label is included in the LLM prompt so the model can reason about its own
    epistemic state without having to interpret raw entropy values.

    Args:
        entropy_norm: Normalized entropy ∈ [0, 1].

    Returns:
        One of "Low", "Medium", or "High".
    """
    val = _clamp(entropy_norm, 0.0, 1.0)
    if val <= 0.33:
        return "Low"
    if val <= 0.67:
        return "Medium"
    return "High"


def compute_signal_conflict(
    env_belief: Dict[str, float],
    social_belief: Dict[str, float],
) -> float:
    """Measure disagreement between env and social beliefs via Jensen-Shannon divergence.

    JSD is symmetric, bounded [0, ln 2], and information-theoretic — consistent with
    the entropy framework used elsewhere in this module.  The raw JSD is normalized
    by ln(2) so the return value lies in [0, 1]:

        0 = sources perfectly agree
        1 = sources maximally disagree (e.g., one says safe, the other says danger)

    This score is recorded for post-hoc RQ1 analysis and surfaced in the LLM prompt
    so the agent can reason about contradictions between its own observation and
    neighbor messages.

    Args:
        env_belief: Belief derived from the agent's own hazard observation.
        social_belief: Belief inferred from neighbor inbox messages.

    Returns:
        Normalized JSD ∈ [0, 1].
    """
    keys = ("p_safe", "p_risky", "p_danger")
    env = _normalize_triplet(env_belief)
    soc = _normalize_triplet(social_belief)
    m = {k: 0.5 * env[k] + 0.5 * soc[k] for k in keys}

    def _kl(p: Dict[str, float], q: Dict[str, float]) -> float:
        return sum(
            max(1e-12, p[k]) * math.log(max(1e-12, p[k]) / max(1e-12, q[k]))
            for k in keys
        )

    jsd = 0.5 * _kl(env, m) + 0.5 * _kl(soc, m)
    return _clamp(jsd / math.log(2), 0.0, 1.0)


def update_agent_belief(
    prev_belief: Dict[str, float],
    env_signal: Dict[str, Any],
    social_signal: Dict[str, Any],
    theta_trust: float,
    inertia: float = 0.35,
) -> Dict[str, Any]:
    """Run the full Bayesian belief-update pipeline for one decision round.

    Steps:
        1. Convert environment signal margin to a categorical prior (``categorize_hazard_state``).
        2. If peer messages exist, fuse env prior with social belief via ``theta_trust``
           (``fuse_env_and_social_beliefs``); otherwise use env prior directly.
        3. Apply temporal smoothing against the previous belief (``smooth_belief``).
        4. Compute and normalize Shannon entropy; bucket into Low / Medium / High.

    Args:
        prev_belief: The agent's belief dict from the previous decision round.
        env_signal: Environment signal produced by ``information_model.sample_environment_signal``.
        social_signal: Social signal produced by ``information_model.build_social_signal``.
        theta_trust: Social-signal trust weight ∈ [0, 1] (from agent profile).
        inertia: Temporal smoothing factor ∈ [0, 0.999].

    Returns:
        An enriched belief dict containing:
            - p_safe, p_risky, p_danger    : smoothed posterior probabilities
            - entropy, entropy_norm        : Shannon entropy (raw and normalized)
            - uncertainty_bucket           : "Low", "Medium", or "High"
            - signal_conflict              : JSD between env and social beliefs [0, 1]
            - env_weight, social_weight    : fusion weights applied this round
            - env_belief, social_belief    : component beliefs before fusion
    """
    env_belief = categorize_hazard_state(env_signal)

    social_count = int(social_signal.get("message_count", 0) or 0)
    social_belief_raw = social_signal.get("social_belief") or {}
    if social_count > 0:
        social_belief = _normalize_triplet(social_belief_raw)
        fused = fuse_env_and_social_beliefs(env_belief, social_belief, theta_trust)
        social_weight = _clamp(theta_trust, 0.0, 1.0)
        env_weight = 1.0 - social_weight
        conflict = compute_signal_conflict(env_belief, social_belief)
    else:
        # No messages in inbox: rely entirely on own environmental observation.
        social_belief = {"p_safe": 1.0 / 3.0, "p_risky": 1.0 / 3.0, "p_danger": 1.0 / 3.0}
        fused = dict(env_belief)
        social_weight = 0.0
        env_weight = 1.0
        conflict = 0.0

    smoothed = smooth_belief(prev_belief or env_belief, fused, inertia=inertia)
    entropy = compute_belief_entropy(smoothed)
    entropy_norm = normalize_entropy(entropy)

    return {
        "p_safe": smoothed["p_safe"],
        "p_risky": smoothed["p_risky"],
        "p_danger": smoothed["p_danger"],
        "entropy": round(entropy, 4),
        "entropy_norm": round(entropy_norm, 4),
        "uncertainty_bucket": bucket_uncertainty(entropy_norm),
        "signal_conflict": round(conflict, 4),
        "env_weight": round(env_weight, 4),
        "social_weight": round(social_weight, 4),
        "env_belief": env_belief,
        "social_belief": social_belief,
    }
