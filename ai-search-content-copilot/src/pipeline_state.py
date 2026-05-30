"""
Shared state schema for the LangGraph pipeline.

`coverage_history` uses operator.add so each geo_simulator node appends its
score rather than replacing the list — LangGraph handles this automatically.
"""
from __future__ import annotations
from typing import Annotated, TypedDict
import operator


class GEOState(TypedDict):
    # ── Runtime config (set once at START) ──────────────────────────────
    url: str
    product_line_hint: str
    generator_model: str
    reviewer_model: str
    hf_token: str
    max_iterations: int       # hard stop on revision cycles
    coverage_threshold: float # stop early if avg score reaches this
    min_improvement: float    # stop if last cycle gained less than this

    # ── Layer 1 outputs (set once) ───────────────────────────────────────
    bundle: dict              # evidence bundle from webpage extractor
    facts_dict: dict          # ProductFacts serialized
    product_name: str
    markdown: str             # initial markdown draft
    audit_dict: dict          # ClaimAudit serialized
    gaps_dict: dict           # ContentGaps serialized
    inventory_dict: dict      # InventoryRow serialized
    eval_questions: list      # list[str]

    # ── Layer 2 outputs (updated each iteration) ─────────────────────────
    sim_results_before: list  # GEOSimResult dicts — from first simulation
    sim_results_current: list # GEOSimResult dicts — from latest simulation
    visibility_gaps_list: list
    strategies_list: list
    revised_markdown: str     # current best markdown (updated each cycle)
    coverage_history: Annotated[list, operator.add]  # appended each simulation
    iteration: int            # incremented by revise_content each cycle
