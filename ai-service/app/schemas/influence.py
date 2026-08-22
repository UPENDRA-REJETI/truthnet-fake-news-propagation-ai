from pydantic import BaseModel


class InfluenceUser(BaseModel):

    user_id: int

    network_role: str

    intervention_priority: str

    influence_score: float

    pagerank: float

    weighted_out_degree: float

    betweenness_centrality: float

    propagation_posts: int


class InfluenceResponse(BaseModel):

    cascade_id: int

    total_users: int

    users: list[InfluenceUser]