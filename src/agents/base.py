"""Shared interface every agent returns."""
from dataclasses import dataclass, field


@dataclass
class AgentResponse:
    agent: str                                    # which agent produced this
    text: str                                     # natural-language answer
    ok: bool = True                               # could it answer?
    sources: list = field(default_factory=list)   # backing evidence (varies by agent)