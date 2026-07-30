from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Observation:
    """Normalized representation of a single LLM generation call."""
    id: str
    trace_id: str
    name: Optional[str]
    model: Optional[str]
    input_tokens: int
    output_tokens: int
    cache_read_tokens: int
    total_tokens: int
    input_cost: float          # USD
    output_cost: float         # USD
    cache_read_cost: float     # USD
    total_cost: float          # USD
    timestamp: datetime
    session_id: Optional[str] = None
    workflow: Optional[str] = None   # trace name — used as workflow label
    turn_index: int = 0              # position within session (0-based)


@dataclass
class TraceRecord:
    """Top-level trace metadata joined with its aggregated generation stats."""
    trace_id: str
    trace_name: Optional[str]
    session_id: Optional[str]
    user_id: Optional[str]
    timestamp: datetime
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cost: float = 0.0
    generation_count: int = 0
    tags: list[str] = field(default_factory=list)
