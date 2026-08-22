from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware


# ============================================================
# PREDICTION
# ============================================================

from app.schemas.prediction import (
    PredictionRequest,
    PredictionResponse
)

from app.services.prediction_service import (
    prediction_service
)


# ============================================================
# PROPAGATION FORECAST
# ============================================================

from app.schemas.propagation import (
    PropagationRequest,
    PropagationResponse
)

from app.services.propagation_service import (
    propagation_service
)


# ============================================================
# REAL PROPAGATION NETWORK
# ============================================================

from app.schemas.network import (
    PropagationNetworkResponse
)

from app.services.network_service import (
    network_service
)


# ============================================================
# INFLUENCE
# ============================================================

from app.schemas.influence import (
    InfluenceResponse
)

from app.services.influence_service import (
    influence_service
)


# ============================================================
# INTERVENTION
# ============================================================

from app.schemas.intervention import (
    InterventionSummaryResponse,
    InterventionCausalResponse
)

from app.services.intervention_service import (
    intervention_service
)

# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="Fake News Propagation AI Service",
    description=(
        "AI service for fake news detection, "
        "propagation prediction and intervention analysis."
    ),
    version="1.0.0"
)

# ============================================================
# CORS — FRONTEND ACCESS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():

    return {
        "status": "OK",
        "service": "Fake News Propagation AI Service",
        "timestamp": datetime.now(
            timezone.utc
        ).isoformat()
    }


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():

    return {
        "message": (
            "Fake News Propagation "
            "AI Service is running."
        )
    }


# ============================================================
# GRAPH SAGE — FAKE NEWS PREDICTION
# ============================================================

@app.post(
    "/predict",
    response_model=PredictionResponse
)
def predict(
    request: PredictionRequest
):

    try:

        result = prediction_service.predict_graph(
            request.graph_index
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# PROPAGATION FORECAST
# ============================================================

@app.post(
    "/propagation",
    response_model=PropagationResponse
)
def predict_propagation(
    request: PropagationRequest
):

    try:

        result = propagation_service.predict(
            window=request.window,
            early_tweets=request.early_tweets,
            early_users=request.early_users,
            early_max_depth=request.early_max_depth,
            early_likes=request.early_likes,
            early_retweets=request.early_retweets,
            engagement=request.engagement
        )

        return result

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# INFLUENCE NETWORK
# ============================================================

@app.get(
    "/influence",
    response_model=InfluenceResponse
)
def get_influence(
    cascade_id: int = 281,
    limit: int = 20
):

    try:

        # ----------------------------------------------------
        # Validate cascade ID
        # ----------------------------------------------------

        if cascade_id < 0:

            raise ValueError(
                "cascade_id must be non-negative"
            )


        # ----------------------------------------------------
        # Validate limit
        # ----------------------------------------------------

        if limit < 1 or limit > 100:

            raise ValueError(
                "limit must be between 1 and 100"
            )


        # ----------------------------------------------------
        # Get influence analysis
        # ----------------------------------------------------

        result = influence_service.get_influence(
            cascade_id=cascade_id,
            limit=limit
        )

        return result


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# REAL PROPAGATION NETWORK
# ============================================================

@app.get(
    "/propagation/network",
    response_model=PropagationNetworkResponse
)
def get_propagation_network(
    cascade_id: int = 281,
    limit: int = 100
):

    try:

        # ----------------------------------------------------
        # Validate cascade ID
        # ----------------------------------------------------

        if cascade_id < 0:

            raise ValueError(
                "cascade_id must be non-negative"
            )


        # ----------------------------------------------------
        # Validate network size
        # ----------------------------------------------------

        if limit < 1 or limit > 500:

            raise ValueError(
                "limit must be between 1 and 500"
            )


        # ----------------------------------------------------
        # Get actual propagation network
        # ----------------------------------------------------

        result = network_service.get_network(
            cascade_id=cascade_id,
            limit=limit
        )

        return result


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# INTERVENTION — MULTI-CASCADE SUMMARY
# ============================================================

@app.get(
    "/intervention/summary",
    response_model=InterventionSummaryResponse
)
def intervention_summary():

    try:

        result = (
            intervention_service
            .get_summary()
        )

        return result


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# INTERVENTION — SINGLE-CASCADE CAUSAL RESULTS
# ============================================================

@app.get(
    "/intervention/causal",
    response_model=InterventionCausalResponse
)
def intervention_causal(
    cascade_id: int = 281
):

    try:

        # ----------------------------------------------------
        # Validate cascade ID
        # ----------------------------------------------------

        if cascade_id < 0:

            raise ValueError(
                "cascade_id must be non-negative"
            )


        # ----------------------------------------------------
        # Get causal intervention results
        # ----------------------------------------------------

        result = (
            intervention_service
            .get_causal_results(
                cascade_id=cascade_id
            )
        )

        return result


    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )


# ============================================================
# INTERVENTION — MULTI-CASCADE CASCADE DETAILS
# ============================================================

@app.get(
    "/intervention/cascade"
)
def intervention_cascade(
    cascade_id: int = 281
):

    try:

        # ----------------------------------------------------
        # Validate cascade ID
        # ----------------------------------------------------

        if cascade_id < 0:

            raise ValueError(
                "cascade_id must be non-negative"
            )


        # ----------------------------------------------------
        # Get detailed cascade results
        # ----------------------------------------------------

        result = (
            intervention_service
            .get_cascade_results(
                cascade_id=cascade_id
            )
        )

        return result


    except ValueError as error:

        raise HTTPException(
            status_code=404,
            detail=str(error)
        )


    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail=str(error)
        )