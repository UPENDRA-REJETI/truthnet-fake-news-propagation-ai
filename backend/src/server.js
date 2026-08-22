const express = require("express");
const cors = require("cors");

const app = express();

const PORT = process.env.PORT || 5000;

const AI_SERVICE_URL =
    process.env.AI_SERVICE_URL || "http://127.0.0.1:8000";

// ============================================================
// MIDDLEWARE
// ============================================================

app.use(cors());
app.use(express.json());


// ============================================================
// ROOT
// ============================================================

app.get("/", (req, res) => {

    res.json({

        service: "TruthNet AI Backend",

        status: "online",

        version: "1.0.0",

        endpoints: {

            health: "/api/health",

            prediction: "/api/predict",

            propagation: "/api/propagation",

            propagationNetwork:
                "/api/propagation/network",

            influence: "/api/influence",

            interventionSummary:
                "/api/intervention/summary",

            interventionCausal:
                "/api/intervention/causal",

            interventionCascade:
                "/api/intervention/cascade"

        }

    });

});


// ============================================================
// HEALTH CHECK
// ============================================================

app.get("/api/health", (req, res) => {

    res.json({

        status: "OK",

        service:
            "Fake News Propagation Backend",

        timestamp:
            new Date().toISOString()

    });

});


// ============================================================
// GRAPH SAGE — FAKE NEWS DETECTION
// ============================================================

app.post("/api/predict", async (req, res) => {

    try {

        const {
            graph_index = 0
        } = req.body;


        // ----------------------------------------------------
        // Validate graph index
        // ----------------------------------------------------

        if (
            !Number.isInteger(graph_index) ||
            graph_index < 0
        ) {

            return res.status(400).json({

                success: false,

                error:
                    "graph_index must be a non-negative integer"

            });

        }


        // ----------------------------------------------------
        // Call FastAPI
        // ----------------------------------------------------

        const response = await fetch(
            `${AI_SERVICE_URL}/predict`,
            {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    graph_index

                })

            }
        );


        const data =
            await response.json();


        // ----------------------------------------------------
        // Handle FastAPI errors
        // ----------------------------------------------------

        if (!response.ok) {

            return res.status(
                response.status
            ).json({

                success: false,

                error:
                    data.detail ||
                    "AI service prediction failed"

            });

        }


        // ----------------------------------------------------
        // Successful response
        // ----------------------------------------------------

        res.json({

            success: true,

            data

        });


    } catch (error) {

        console.error(
            "Prediction error:",
            error
        );


        res.status(500).json({

            success: false,

            error:
                "Unable to connect to AI service"

        });

    }

});


// ============================================================
// PROPAGATION FORECAST
// ============================================================

app.post("/api/propagation", async (req, res) => {

    try {

        const {

            window,

            early_tweets,

            early_users,

            early_max_depth,

            early_likes,

            early_retweets,

            engagement

        } = req.body;


        // ----------------------------------------------------
        // Validate window
        // ----------------------------------------------------

        if (
            ![1, 6, 24].includes(window)
        ) {

            return res.status(400).json({

                success: false,

                error:
                    "window must be 1, 6, or 24"

            });

        }


        // ----------------------------------------------------
        // Validate numeric features
        // ----------------------------------------------------

        const fields = {

            early_tweets,

            early_users,

            early_max_depth,

            early_likes,

            early_retweets,

            engagement

        };


        for (
            const [field, value]
            of Object.entries(fields)
        ) {

            if (
                !Number.isInteger(value) ||
                value < 0
            ) {

                return res.status(400).json({

                    success: false,

                    error:
                        `${field} must be a non-negative integer`

                });

            }

        }


        // ----------------------------------------------------
        // Call FastAPI
        // ----------------------------------------------------

        const response = await fetch(
            `${AI_SERVICE_URL}/propagation`,
            {

                method: "POST",

                headers: {

                    "Content-Type":
                        "application/json"

                },

                body: JSON.stringify({

                    window,

                    early_tweets,

                    early_users,

                    early_max_depth,

                    early_likes,

                    early_retweets,

                    engagement

                })

            }
        );


        const data =
            await response.json();


        // ----------------------------------------------------
        // Handle FastAPI errors
        // ----------------------------------------------------

        if (!response.ok) {

            return res.status(
                response.status
            ).json({

                success: false,

                error:
                    data.detail ||
                    "Propagation prediction failed"

            });

        }


        // ----------------------------------------------------
        // Successful response
        // ----------------------------------------------------

        res.json({

            success: true,

            data

        });


    } catch (error) {

        console.error(
            "Propagation prediction error:",
            error
        );


        res.status(500).json({

            success: false,

            error:
                "Unable to connect to propagation AI service"

        });

    }

});


// ============================================================
// PROPAGATION NETWORK
// ============================================================
// This endpoint exposes the actual user-to-user propagation
// network from the FastAPI AI service.
//
// Frontend:
// GET /api/propagation/network?cascade_id=281&limit=50
//
// FastAPI:
// GET /propagation/network?cascade_id=281&limit=50
// ============================================================

app.get(
    "/api/propagation/network",
    async (req, res) => {

        try {

            const cascadeId = Number(
                req.query.cascade_id || 281
            );

            const limit = Number(
                req.query.limit || 50
            );


            // ------------------------------------------------
            // Validate cascade ID
            // ------------------------------------------------

            if (
                !Number.isInteger(cascadeId) ||
                cascadeId < 0
            ) {

                return res.status(400).json({

                    success: false,

                    error:
                        "cascade_id must be a non-negative integer"

                });

            }


            // ------------------------------------------------
            // Validate limit
            // ------------------------------------------------

            if (
                !Number.isInteger(limit) ||
                limit < 1 ||
                limit > 100
            ) {

                return res.status(400).json({

                    success: false,

                    error:
                        "limit must be between 1 and 100"

                });

            }


            // ------------------------------------------------
            // Call FastAPI propagation network endpoint
            // ------------------------------------------------

            const response = await fetch(
                `${AI_SERVICE_URL}/propagation/network?cascade_id=${cascadeId}&limit=${limit}`,
                {

                    method: "GET"

                }
            );


            const data =
                await response.json();


            // ------------------------------------------------
            // Handle FastAPI errors
            // ------------------------------------------------

            if (!response.ok) {

                return res.status(
                    response.status
                ).json({

                    success: false,

                    error:
                        data.detail ||
                        "Propagation network analysis failed"

                });

            }


            // ------------------------------------------------
            // Successful response
            // ------------------------------------------------

            res.json({

                success: true,

                data

            });


        } catch (error) {

            console.error(
                "Propagation network error:",
                error
            );


            res.status(500).json({

                success: false,

                error:
                    "Unable to connect to propagation network AI service"

            });

        }

    }
);


// ============================================================
// INFLUENCE NETWORK
// ============================================================

app.get("/api/influence", async (req, res) => {

    try {

        const cascadeId = Number(
            req.query.cascade_id || 281
        );

        const limit = Number(
            req.query.limit || 20
        );


        // ----------------------------------------------------
        // Validate cascade ID
        // ----------------------------------------------------

        if (
            !Number.isInteger(cascadeId) ||
            cascadeId < 0
        ) {

            return res.status(400).json({

                success: false,

                error:
                    "cascade_id must be a non-negative integer"

            });

        }


        // ----------------------------------------------------
        // Validate limit
        // ----------------------------------------------------

        if (
            !Number.isInteger(limit) ||
            limit < 1 ||
            limit > 100
        ) {

            return res.status(400).json({

                success: false,

                error:
                    "limit must be between 1 and 100"

            });

        }


        // ----------------------------------------------------
        // Call FastAPI
        // ----------------------------------------------------

        const response = await fetch(
            `${AI_SERVICE_URL}/influence?cascade_id=${cascadeId}&limit=${limit}`,
            {

                method: "GET"

            }
        );


        const data =
            await response.json();


        // ----------------------------------------------------
        // Handle FastAPI errors
        // ----------------------------------------------------

        if (!response.ok) {

            return res.status(
                response.status
            ).json({

                success: false,

                error:
                    data.detail ||
                    "Influence analysis failed"

            });

        }


        // ----------------------------------------------------
        // Successful response
        // ----------------------------------------------------

        res.json({

            success: true,

            data

        });


    } catch (error) {

        console.error(
            "Influence analysis error:",
            error
        );


        res.status(500).json({

            success: false,

            error:
                "Unable to connect to influence AI service"

        });

    }

});


// ============================================================
// INTERVENTION — MULTI-CASCADE SUMMARY
// ============================================================

app.get(
    "/api/intervention/summary",
    async (req, res) => {

        try {

            // ------------------------------------------------
            // Call FastAPI
            // ------------------------------------------------

            const response = await fetch(
                `${AI_SERVICE_URL}/intervention/summary`,
                {
                    method: "GET"
                }
            );


            const data =
                await response.json();


            // ------------------------------------------------
            // Handle FastAPI errors
            // ------------------------------------------------

            if (!response.ok) {

                return res.status(
                    response.status
                ).json({

                    success: false,

                    error:
                        data.detail ||
                        "Intervention summary failed"

                });

            }


            // ------------------------------------------------
            // Successful response
            // ------------------------------------------------

            res.json({

                success: true,

                data

            });


        } catch (error) {

            console.error(
                "Intervention summary error:",
                error
            );


            res.status(500).json({

                success: false,

                error:
                    "Unable to connect to intervention AI service"

            });

        }

    }
);


// ============================================================
// INTERVENTION — CAUSAL RESULTS
// ============================================================

app.get(
    "/api/intervention/causal",
    async (req, res) => {

        try {

            const cascadeId = Number(
                req.query.cascade_id || 281
            );


            // ------------------------------------------------
            // Validate cascade ID
            // ------------------------------------------------

            if (
                !Number.isInteger(cascadeId) ||
                cascadeId < 0
            ) {

                return res.status(400).json({

                    success: false,

                    error:
                        "cascade_id must be a non-negative integer"

                });

            }


            // ------------------------------------------------
            // Call FastAPI
            // ------------------------------------------------

            const response = await fetch(
                `${AI_SERVICE_URL}/intervention/causal?cascade_id=${cascadeId}`,
                {
                    method: "GET"
                }
            );


            const data =
                await response.json();


            // ------------------------------------------------
            // Handle FastAPI errors
            // ------------------------------------------------

            if (!response.ok) {

                return res.status(
                    response.status
                ).json({

                    success: false,

                    error:
                        data.detail ||
                        "Causal intervention analysis failed"

                });

            }


            // ------------------------------------------------
            // Successful response
            // ------------------------------------------------

            res.json({

                success: true,

                data

            });


        } catch (error) {

            console.error(
                "Causal intervention error:",
                error
            );


            res.status(500).json({

                success: false,

                error:
                    "Unable to connect to intervention AI service"

            });

        }

    }
);


// ============================================================
// INTERVENTION — CASCADE DETAILS
// ============================================================

app.get(
    "/api/intervention/cascade",
    async (req, res) => {

        try {

            const cascadeId = Number(
                req.query.cascade_id || 281
            );


            // ------------------------------------------------
            // Validate cascade ID
            // ------------------------------------------------

            if (
                !Number.isInteger(cascadeId) ||
                cascadeId < 0
            ) {

                return res.status(400).json({

                    success: false,

                    error:
                        "cascade_id must be a non-negative integer"

                });

            }


            // ------------------------------------------------
            // Call FastAPI
            // ------------------------------------------------

            const response = await fetch(
                `${AI_SERVICE_URL}/intervention/cascade?cascade_id=${cascadeId}`,
                {
                    method: "GET"
                }
            );


            const data =
                await response.json();


            // ------------------------------------------------
            // Handle FastAPI errors
            // ------------------------------------------------

            if (!response.ok) {

                return res.status(
                    response.status
                ).json({

                    success: false,

                    error:
                        data.detail ||
                        "Cascade intervention analysis failed"

                });

            }


            // ------------------------------------------------
            // Successful response
            // ------------------------------------------------

            res.json({

                success: true,

                data

            });


        } catch (error) {

            console.error(
                "Cascade intervention error:",
                error
            );


            res.status(500).json({

                success: false,

                error:
                    "Unable to connect to intervention AI service"

            });

        }

    }
);


// ============================================================
// START SERVER
// ============================================================

app.listen(PORT, () => {

    console.log(
        `Backend running on http://localhost:${PORT}`
    );

    console.log(
        `AI service configured at ${AI_SERVICE_URL}`
    );

});