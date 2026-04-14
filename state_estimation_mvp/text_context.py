from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass
class ContextFields:
    physical: str = "resting"
    social: str = "alone"
    task: str = "observing"
    digital: str = "no interruption"
    environment: str = "indoor"
    temporal: str = "brief interaction"


ALLOWED: dict[str, list[str]] = {
    "physical": ["resting", "sitting", "walking", "moving", "navigating"],
    "social": ["alone", "socially engaged", "in conversation", "interacting with others", "being observed"],
    "task": ["observing", "interacting", "multitasking", "multi-step task", "exploratory task"],
    "digital": [
        "no interruption",
        "occasional interruptions",
        "frequent interruptions",
        "continuous interruptions",
        "high information density",
        "visually intrusive updates",
    ],
    "environment": ["indoor", "outdoor", "dynamic environment", "quiet environment", "crowded environment"],
    "temporal": ["brief interaction", "intermittent interaction", "sustained interaction", "continuous exposure"],
}


def validate_context(context: dict[str, str]) -> None:
    required = ["physical", "social", "task", "digital", "environment", "temporal"]
    for key in required:
        if key not in context:
            raise ValueError(f"[MISSING] {key}")
        value = str(context[key]).strip()
        if value not in ALLOWED[key]:
            raise ValueError(f"[INVALID] {key}: {value}")


def build_text(context: dict[str, str]) -> str:
    validate_context(context)
    return (
        f"Physical context: {context['physical']}. "
        f"Social context: {context['social']}. "
        f"Task context: {context['task']}. "
        f"Digital context: {context['digital']}. "
        f"Environment context: {context['environment']}. "
        f"Temporal context: {context['temporal']}."
    )


class TextContextBuilder:
    """Compatibility wrapper around strict schema builder."""

    @staticmethod
    def build(fields: ContextFields) -> str:
        return build_text(
            {
                "physical": fields.physical,
                "social": fields.social,
                "task": fields.task,
                "digital": fields.digital,
                "environment": fields.environment,
                "temporal": fields.temporal,
            }
        )


FORBIDDEN_LABEL_HINTS = [
    "stressed",
    "stress",
    "high cognitive load",
    "low cognitive load",
    "high arousal",
    "low arousal",
    "positive valence",
    "negative valence",
    "anxious",
    "overloaded",
    "excited",
    "uncomfortable",
]


def validate_context_text(text: str, min_words: int = 8) -> tuple[bool, list[str]]:
    issues: list[str] = []
    cleaned = text.strip()

    if not cleaned:
        issues.append("empty text")
        return False, issues

    word_count = len(re.findall(r"\b\w+\b", cleaned))
    if word_count < min_words:
        issues.append(f"too short ({word_count} words)")

    low = cleaned.lower()
    for bad in FORBIDDEN_LABEL_HINTS:
        pattern = r"\b" + re.escape(bad) + r"\b"
        if re.search(pattern, low):
            issues.append(f"label leakage phrase: '{bad}'")

    is_valid = len(issues) == 0
    return is_valid, issues


def context_from_slots(
    physical: str,
    social: str,
    task: str,
    digital: str,
    environment: str,
    temporal: str,
) -> ContextFields:
    context = {
        "physical": physical,
        "social": social,
        "task": task,
        "digital": digital,
        "environment": environment,
        "temporal": temporal,
    }
    validate_context(context)
    return ContextFields(**context)
