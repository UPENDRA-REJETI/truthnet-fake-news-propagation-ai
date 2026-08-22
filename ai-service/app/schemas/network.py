from pydantic import BaseModel


class PropagationNode(BaseModel):
    user_id: int
    posts: int
    propagation_posts: int
    max_depth: int
    total_likes: int
    total_retweets: int


class PropagationEdge(BaseModel):
    source: int
    target: int
    depth: int
    tweet_id: int
    parent_id: int
    likes: int
    retweets: int


class PropagationNetworkResponse(BaseModel):
    cascade_id: int
    total_records: int
    total_edges: int
    nodes: list[PropagationNode]
    edges: list[PropagationEdge]