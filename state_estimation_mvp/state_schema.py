from __future__ import annotations

from dataclasses import dataclass


ALLOWED_STATE_VALUES: dict[str, list[str]] = {
    "posture": ["standing", "sitting", "lying"],
    "movement": ["stationary", "walking", "running", "sleep"],
    "social_engagement": ["alone", "partner", "group"],
    "relation": ["none", "friend", "family", "colleague", "stranger"],
    "device_interaction_behavior": [
        "no_interaction",
        "passive_viewing",
        "short_taps",
        "continuous_scrolling",
        "active_input",
    ],
    "environment": ["indoor", "outdoor", "dynamic", "quiet", "crowded"],
    "temporal": ["brief", "intermittent", "sustained", "continuous"],
}


@dataclass
class UserSessionState:
    posture: str
    movement: str
    social_engagement: str
    relation: str
    device_interaction_behavior: str
    environment: str
    temporal: str


def validate_state(state: dict[str, str]) -> None:
    for k, allowed in ALLOWED_STATE_VALUES.items():
        if k not in state:
            raise ValueError(f"[MISSING] {k}")
        v = str(state[k]).strip()
        if v not in allowed:
            raise ValueError(f"[INVALID] {k}: {v}")


def state_signature(state: dict[str, str]) -> tuple[str, ...]:
    validate_state(state)
    return (
        state["posture"],
        state["movement"],
        state["social_engagement"],
        state["relation"],
        state["device_interaction_behavior"],
        state["environment"],
        state["temporal"],
    )
