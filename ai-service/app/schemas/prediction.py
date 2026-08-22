from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):

    graph_index: int = Field(
        default=0,
        ge=0,
        description="Index of a graph from the UPFD GossipCop test dataset."
    )


class PredictionResponse(BaseModel):

    prediction: str
    class_id: int
    real_probability: float
    fake_probability: float
    graph_index: int
    model: str
    dataset: str