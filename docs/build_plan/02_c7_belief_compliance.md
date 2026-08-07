# P1b. C.7 belief-only compliance

Design source `docs/jason_0611_response_assessment.md` Part C.7. Co-developed with [M1 alert engine](01_m1_alert_engine.md), which supplies the active-order flag and channel weight. This is decision 1, locked to belief-only.

## Goal

An evacuation order, whether broadcast by M1 or delivered door-to-door by M3, is a strong but overridable input, not a forced departure. Compliance becomes an emergent, measurable output and is what E4 manipulates through a new authority-trust parameter.

## Why belief-only

As the code stands an order is absolute. The predeparture policy makes a MANDATORY rule "if an official evacuation order is present, depart" at `main.py:2596-2598`, and `theta_trust` weights only peers versus own observation, never authority. A door-knock would force every household out, which contradicts the record where Middle Sackville largely stayed, and reinstates the compliance prior the manuscript criticises.

Belief-only routes the order through the belief, not the prompt, and deletes both the mandatory rule and the order text from the departure step. This was chosen over a hybrid for three reasons. It gives clean attribution, so an E4 compliance shift is attributable to authority trust alone rather than to a constant prompt nudge. It holds exact parity between the LLM and rule-based agents, because `rule_based_predeparture` (`agentevac/agents/rule_based_policy.py:58`) delegates to `should_depart_now` and never sees a prompt, so belief is the only channel that reaches both. And it is the strongest answer to the baked-in-compliance critique that motivates the rebuild. The realism cost is contained, because the alert still drives awareness through M2 and still flows to the routing decision through the normal filters. Only the when-to-leave text is removed.

Fallback. If the lived-experience coding later needs the departure reason to cite the order to be codeable against Jason's PADM frame, switch to factual hybrid, meaning a factual non-imperative order line added to the predeparture prompt, held constant across the `theta_auth` sweep so the confound is a fixed offset, never persuasive. Not built now.

## Current state

- Belief fusion is `fuse_env_and_social_beliefs(env_belief, social_belief, theta_trust)` at `agentevac/agents/belief_model.py:101`, returning `(1 - theta_trust) * env + theta_trust * social`. There is no authority channel.
- Departure is `should_depart_now` in `agentevac/agents/departure_model.py`, whose Clause 1 is `p_danger > theta_r`. It reads belief and never reads any order field.
- The agent profile carries `theta_trust, theta_r, theta_u, gamma, lambda_e, lambda_t` and social-pressure params, sampled by `sample_profile_params` in `agentevac/agents/agent_state.py`. There is no `theta_auth`.
- The order reaches the LLM through the prompt block at `main.py:2613` and the MANDATORY rule text at `main.py:2596-2598`. It reaches the rule-based agent through nothing, so the rule-based mirror is currently blind to orders.

## Changes

### `belief_model.py`, institutional channel after fusion

Add the channel right after `fuse_env_and_social_beliefs` at line 101, inert when no order is active.

```
base  = (1 - theta_trust) * env_belief + theta_trust * social_belief   # unchanged
I     = order triplet for the active instruction
        evacuate_now -> {p_safe: 0.05, p_risky: 0.15, p_danger: 0.80}
a     = clamp(theta_auth * channel_factor, 0.0, 1.0)
fused = (1 - a) * base + a * I,   then normalize
```

`a = 0` when there is no order or zero authority trust, so all current behaviour is unchanged. `channel_factor` is the channel weight, door-to-door about 1.0 in person, broadcast about 0.6, supplied by the M1 and M3 resolvers.

### `agent_state.py`, the theta_auth parameter

Add `theta_auth` to the profile with a default, a spread, and bounds [0, 1], drawn through the existing `sample_profile_params` heterogeneity machinery alongside `theta_trust`. Back-fill it in `ensure_agent_state` like the other profile keys so older states gain it without a reset.

### `main.py`, wire and de-gate

- Add `DEFAULT_THETA_AUTH`, `THETA_AUTH_SPREAD`, and its bounds near the other cognition constants, thread them into the means, spreads, and bounds dicts passed to `sample_profile_params`, and record them in the run-params snapshot.
- Delete the MANDATORY order rule at `main.py:2596-2598` and delete the `official_evacuation_order` prompt block at `main.py:2613`, so no order text reaches the predeparture prompt.
- Feed the active-order flag and `channel_factor` from the resolved `alert_state`, plus the M3 door state, into the belief update.
- Tag the departure `reason` as `order` when the elevated belief is what crossed Clause 1 under an active order, for the compliance metric in Part J.

Departure then emerges through the unchanged `should_depart_now` Clause 1, so a low-`theta_auth` or low-perceived-risk household can decline. Both agent types respond through the same belief path, so parity holds exactly.

## Tests

- Belief test that `a = 0`, meaning no order or `theta_auth = 0`, reproduces the old `fuse_env_and_social_beliefs` output exactly.
- Belief test that `a = 1` with `channel_factor = 1` pulls the fused belief to the order triplet.
- Monotonicity test that fused `p_danger` rises with `theta_auth` for a fixed base and active order.
- Sampling test that `theta_auth` respects its bounds and is deterministic per `(master_seed, agent_id)`.
- Regression test that a run with no active order anywhere is bit-identical to the pre-C.7 belief path.

## Open sub-decisions, calibration, resolve at E0

These are calibration values, so seed them with the design defaults to wire the code, then tune during E0 validation against the record's clearance and compliance. They do not block starting the code.

1. The order triplet `I` for `evacuate_now`, design default {0.05, 0.15, 0.80}.
2. `channel_factor`, door-to-door about 1.0 and broadcast about 0.6.
3. `theta_auth` default, spread, and bounds, and the E4 sweep levels.

## Acceptance criteria

- No order anywhere in a run leaves both agent types bit-identical to the pre-C.7 baseline.
- Under an active order, raising `theta_auth` raises the fraction of ordered households that depart, with no prompt change, for both LLM and rule-based agents.
- The mandatory rule and the order prompt block are gone, so a `git grep` for `official_evacuation_order` in the predeparture path returns nothing.
