"""Versioned paper-inspired prompt rendering."""

TEMPLATE_VERSION = "jev-numbered-v1"
CONTINUATION_PREFIX = "Best answer: ["


def render_prompt(context: str, candidates: list[tuple[str, str]]) -> str:
    options = "\n".join(
        f"[{index}] {description}" for index, (_, description) in enumerate(candidates, 1)
    )
    return (
        "You are selecting the best action for the given context.\n\n"
        f"Context:\n{context.strip()}\n\n"
        f"Options:\n{options}\n\n"
        f"{CONTINUATION_PREFIX}"
    )
