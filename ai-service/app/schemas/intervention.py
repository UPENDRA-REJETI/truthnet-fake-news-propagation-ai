from typing import Literal

from pydantic import BaseModel


class InterventionSummaryItem(BaseModel):

    strategy: str

    timing: Literal[
        "early",
        "mid",
        "late"
    ]

    mean: float
    median: float
    std: float
    min: float
    max: float


class InterventionSummaryResponse(BaseModel):

    experiment: str

    cascades_validated: int

    metric: str

    results: list[InterventionSummaryItem]


class InterventionCausalItem(BaseModel):

    strategy: str

    timing: str

    targets: int

    final_reach: int

    remaining_edges: int

    max_depth: int

    reach_reduction_percent: float

    edge_reduction_percent: float


class InterventionCausalResponse(BaseModel):

    experiment: str

    cascade_id: int

    baseline_reach: int

    baseline_edges: int

    results: list[InterventionCausalItem]