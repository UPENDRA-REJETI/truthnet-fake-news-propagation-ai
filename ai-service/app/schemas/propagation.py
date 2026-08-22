from typing import Literal

from pydantic import BaseModel, Field


class PropagationRequest(BaseModel):

    window: Literal[1, 6, 24] = Field(
        description="Propagation observation window in hours."
    )

    early_tweets: int = Field(
        ge=0,
        description="Number of tweets observed in the window."
    )

    early_users: int = Field(
        ge=0,
        description="Number of unique users observed in the window."
    )

    early_max_depth: int = Field(
        ge=0,
        description="Maximum propagation depth observed."
    )

    early_likes: int = Field(
        ge=0,
        description="Total likes observed in the window."
    )

    early_retweets: int = Field(
        ge=0,
        description="Total retweets observed in the window."
    )

    engagement: int = Field(
        ge=0,
        description="Total likes plus retweets observed in the window."
    )


class PropagationResponse(BaseModel):

    window_hours: int

    predicted_cascade_size: float

    predicted_log_cascade_size: float

    model: str

    dataset: str

    features_used: list[str]