import { useState, useMemo, useEffect, useRef } from "react";
import ForceGraph2D from "react-force-graph-2d";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from "recharts";
import "./App.css";

const API_URL = "http://127.0.0.1:5000";
const PROPAGATION_API_URL = "http://127.0.0.1:8000";

function App() {
  // ============================================================
  // GRAPH SAGE DETECTION
  // ============================================================

  const [graphIndex, setGraphIndex] = useState(0);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  // ============================================================
  // PROPAGATION FORECAST
  // ============================================================

  const [propagationWindow, setPropagationWindow] = useState(6);

  const [propagationInputs, setPropagationInputs] = useState({
    early_tweets: 24,
    early_users: 21,
    early_max_depth: 2,
    early_likes: 350,
    early_retweets: 142,
    engagement: 492,
  });

  const [propagationResult, setPropagationResult] = useState(null);
  const [propagationLoading, setPropagationLoading] = useState(false);
  const [propagationError, setPropagationError] = useState("");

  // ============================================================
  // INFLUENCE NETWORK
  // ============================================================

  const [influenceCascade, setInfluenceCascade] = useState(281);
  const [influenceUsers, setInfluenceUsers] = useState([]);
  const [influenceTotalUsers, setInfluenceTotalUsers] = useState(0);
  const [influenceLoading, setInfluenceLoading] = useState(false);
  const [influenceError, setInfluenceError] = useState("");

  // ============================================================
  // REAL PROPAGATION NETWORK
  // ============================================================

  const [propagationNetwork, setPropagationNetwork] = useState({
    cascade_id: 281,
    total_records: 0,
    total_edges: 0,
    nodes: [],
    edges: [],
  });
  const [propagationNetworkLoading, setPropagationNetworkLoading] =
    useState(false);
  const [propagationNetworkError, setPropagationNetworkError] =
    useState("");
  const [selectedPropagationNode, setSelectedPropagationNode] =
    useState(null);
  const [hoveredPropagationNode, setHoveredPropagationNode] =
    useState(null);
  const propagationGraphRef = useRef(null);
  const [propagationGraphWidth, setPropagationGraphWidth] = useState(900);

  // ============================================================
  // INTERVENTION INTELLIGENCE
  // ============================================================

  const [interventionCascade, setInterventionCascade] = useState(281);
  const [interventionSummary, setInterventionSummary] = useState(null);
  const [interventionCausal, setInterventionCausal] = useState(null);
  const [interventionCascadeData, setInterventionCascadeData] =
    useState(null);
  const [interventionLoading, setInterventionLoading] = useState(false);
  const [interventionError, setInterventionError] = useState("");

  // ============================================================
  // RESEARCH EXPERIMENTS
  // ============================================================

  const [experimentData, setExperimentData] = useState(null);
  const [experimentLoading, setExperimentLoading] = useState(false);
  const [experimentError, setExperimentError] = useState("");

  // ============================================================
  // NAVIGATION
  // ============================================================

  const scrollToSection = (sectionId) => {
    const section = document.getElementById(sectionId);

    if (section) {
      section.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  };

  // ============================================================
  // GRAPH SAGE PREDICTION
  // ============================================================

  const analyzeGraph = async () => {
    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await fetch(`${API_URL}/api/predict`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          graph_index: Number(graphIndex),
        }),
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed");
      }

      setResult(data.data);
    } catch (err) {
      setError(
        err.message || "Unable to connect to the AI service."
      );
    } finally {
      setLoading(false);
    }
  };

  // ============================================================
  // PROPAGATION INPUT
  // ============================================================

  const handlePropagationInput = (event) => {
    const { name, value } = event.target;

    setPropagationInputs((previous) => ({
      ...previous,
      [name]: Number(value),
    }));
  };

  // ============================================================
  // PROPAGATION FORECAST
  // ============================================================

  const predictPropagation = async () => {
    setPropagationLoading(true);
    setPropagationError("");
    setPropagationResult(null);

    try {
      const response = await fetch(
        `${API_URL}/api/propagation`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            window: Number(propagationWindow),
            ...propagationInputs,
          }),
        }
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Propagation prediction failed"
        );
      }

      setPropagationResult(data.data);
    } catch (err) {
      setPropagationError(
        err.message ||
          "Unable to connect to the propagation AI service."
      );
    } finally {
      setPropagationLoading(false);
    }
  };

  // ============================================================
  // INFLUENCE NETWORK
  // ============================================================

  const loadInfluenceNetwork = async () => {
    setInfluenceLoading(true);
    setInfluenceError("");
    setInfluenceUsers([]);

    try {
      const response = await fetch(
        `${API_URL}/api/influence?cascade_id=${Number(
          influenceCascade
        )}&limit=20`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error || "Influence analysis failed"
        );
      }

      setInfluenceUsers(data.data?.users || []);
      setInfluenceTotalUsers(data.data?.total_users || 0);

      // Load the actual observed user-to-user propagation edges
      // for the same cascade.
      await loadPropagationNetwork(Number(influenceCascade));
    } catch (err) {
      setInfluenceError(
        err.message || "Unable to load influence network."
      );

      setInfluenceUsers([]);
      setInfluenceTotalUsers(0);
    } finally {
      setInfluenceLoading(false);
    }
  };

  // ============================================================
  // REAL PROPAGATION NETWORK
  // ============================================================

  const loadPropagationNetwork = async (cascadeId = influenceCascade) => {
    setPropagationNetworkLoading(true);
    setPropagationNetworkError("");

    try {
      const response = await fetch(
        `${PROPAGATION_API_URL}/propagation/network?cascade_id=${Number(
          cascadeId
        )}&limit=100`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            data.detail ||
            "Unable to load real propagation network."
        );
      }

      setPropagationNetwork(
        data.data ||
          data || {
            cascade_id: Number(cascadeId),
            total_records: 0,
            total_edges: 0,
            nodes: [],
            edges: [],
          }
      );
      setSelectedPropagationNode(null);
    } catch (err) {
      setPropagationNetworkError(
        err.message ||
          "Unable to load real propagation network."
      );
      setPropagationNetwork({
        cascade_id: Number(cascadeId),
        total_records: 0,
        total_edges: 0,
        nodes: [],
        edges: [],
      });
    } finally {
      setPropagationNetworkLoading(false);
    }
  };

  // ============================================================
  // INTERVENTION ANALYSIS
  // ============================================================

  const loadIntervention = async () => {
    setInterventionLoading(true);
    setInterventionError("");

    setInterventionSummary(null);
    setInterventionCausal(null);
    setInterventionCascadeData(null);

    try {
      const cascadeId = Number(interventionCascade);

      const [
        summaryResponse,
        causalResponse,
        cascadeResponse,
      ] = await Promise.all([
        fetch(`${API_URL}/api/intervention/summary`),

        fetch(
          `${API_URL}/api/intervention/causal?cascade_id=${cascadeId}`
        ),

        fetch(
          `${API_URL}/api/intervention/cascade?cascade_id=${cascadeId}`
        ),
      ]);

      const summaryData = await summaryResponse.json();
      const causalData = await causalResponse.json();
      const cascadeData = await cascadeResponse.json();

      if (!summaryResponse.ok) {
        throw new Error(
          summaryData.error ||
            "Unable to load intervention summary."
        );
      }

      if (!causalResponse.ok) {
        throw new Error(
          causalData.error ||
            "Unable to load causal intervention data."
        );
      }

      if (!cascadeResponse.ok) {
        throw new Error(
          cascadeData.error ||
            "Unable to load cascade intervention data."
        );
      }

      setInterventionSummary(summaryData.data);
      setInterventionCausal(causalData.data);
      setInterventionCascadeData(cascadeData.data);
    } catch (err) {
      setInterventionError(
        err.message ||
          "Unable to load intervention analysis."
      );
    } finally {
      setInterventionLoading(false);
    }
  };

  // ============================================================
  // RESEARCH EXPERIMENTS
  // ============================================================

  const loadExperimentResults = async () => {
    setExperimentLoading(true);
    setExperimentError("");
    setExperimentData(null);

    try {
      const response = await fetch(
        `${API_URL}/api/intervention/summary`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.error ||
            "Unable to load experiment results."
        );
      }

      setExperimentData(data.data);
    } catch (err) {
      setExperimentError(
        err.message ||
          "Unable to load experiment results."
      );
    } finally {
      setExperimentLoading(false);
    }
  };

  // ============================================================
  // EXPERIMENT DATA
  // ============================================================

  const experimentResults =
    experimentData?.results || [];

  const earlyResults = experimentResults.filter(
    (item) =>
      String(item.timing).toLowerCase() === "early"
  );

  const midResults = experimentResults.filter(
    (item) =>
      String(item.timing).toLowerCase() === "mid"
  );

  const lateResults = experimentResults.filter(
    (item) =>
      String(item.timing).toLowerCase() === "late"
  );

  const average = (items) => {
    if (!items.length) return 0;

    return (
      items.reduce(
        (sum, item) =>
          sum +
          Number(
            item.mean ||
              item.reach_reduction_percent ||
              0
          ),
        0
      ) / items.length
    );
  };

  const earlyAverage = average(earlyResults);
  const midAverage = average(midResults);
  const lateAverage = average(lateResults);

  const bestExperiment =
    experimentResults.length > 0
      ? experimentResults.reduce(
          (best, current) =>
            Number(
              current.mean ||
                current.reach_reduction_percent ||
                0
            ) >
            Number(
              best.mean ||
                best.reach_reduction_percent ||
                0
            )
              ? current
              : best
        )
      : null;

  // ============================================================
  // CHART DATA — TIMING
  // ============================================================

  const timingChartData = [
    {
      timing: "Early",
      reduction: Number(earlyAverage.toFixed(2)),
    },
    {
      timing: "Mid",
      reduction: Number(midAverage.toFixed(2)),
    },
    {
      timing: "Late",
      reduction: Number(lateAverage.toFixed(2)),
    },
  ];

  // ============================================================
  // CHART DATA — STRATEGY
  // ============================================================

  const strategyNames = [
    "fact_checking",
    "content_moderation",
    "targeted_intervention",
  ];

  const strategyChartData = strategyNames.map(
    (strategy) => {
      const rows = experimentResults.filter(
        (item) =>
          String(item.strategy).toLowerCase() ===
          strategy
      );

      return {
        strategy: strategy
          .replace(/_/g, " ")
          .replace(/\b\w/g, (letter) =>
            letter.toUpperCase()
          ),
        reduction: Number(average(rows).toFixed(2)),
      };
    }
  );

  // ============================================================
  // CHART DATA — INTERVENTION
  // ============================================================

  const interventionResults =
    interventionCausal?.results || [];

  const interventionChartData =
    interventionResults.map((item) => ({
      name: `${String(
        item.strategy || ""
      )
        .replace(/_/g, " ")
        .replace(/\b\w/g, (letter) =>
          letter.toUpperCase()
        )} - ${String(
        item.timing || ""
      ).toUpperCase()}`,
      reachReduction: Number(
        item.reach_reduction_percent || 0
      ),
      edgeReduction: Number(
        item.edge_reduction_percent || 0
      ),
    }));

  // ============================================================
  // CHART DATA — INFLUENCE
  // ============================================================

  const influenceChartData = influenceUsers
    .slice(0, 10)
    .map((user, index) => ({
      user: `#${user.user_id}`,
      influence: Number(
        user.influence_score || 0
      ),
      rank: index + 1,
    }));

  // ============================================================
  // DETECTION CONFIDENCE
  // ============================================================

  const confidence = result
    ? Math.max(
        Number(result.real_probability || 0),
        Number(result.fake_probability || 0)
      ) * 100
    : 0;

  // ============================================================
  // PROPAGATION RISK
  // ============================================================

  const cascadeSize = Number(
    propagationResult?.predicted_cascade_size || 0
  );

  let propagationRisk = "LOW";

  if (cascadeSize >= 1000) {
    propagationRisk = "CRITICAL";
  } else if (cascadeSize >= 500) {
    propagationRisk = "HIGH";
  } else if (cascadeSize >= 100) {
    propagationRisk = "MODERATE";
  }

  // ============================================================
  // TOP INFLUENCE USER
  // ============================================================

  const topInfluenceUser =
    influenceUsers.length > 0
      ? influenceUsers[0]
      : null;

  // ============================================================
  // INTERACTIVE INFLUENCE GRAPH
  // ============================================================

  const influenceGraphData = useMemo(() => {
    if (!influenceUsers.length) {
      return {
        nodes: [],
        links: [],
      };
    }

    const nodes = influenceUsers.map((user) => ({
      id: String(user.user_id),
      user_id: user.user_id,
      role: user.network_role,
      priority: user.intervention_priority,
      influence: Number(user.influence_score || 0),
      posts: Number(user.propagation_posts || 0),
    }));

    const links = [];

    // The current influence API returns ranked nodes rather than
    // raw user-to-user edges. Therefore this is a visual influence
    // topology centered on the highest-ranked displayed node.
    if (nodes.length > 1) {
      const root = nodes[0];

      nodes.slice(1).forEach((node) => {
        links.push({
          source: root.id,
          target: node.id,
          value: node.influence,
        });
      });
    }

    return {
      nodes,
      links,
    };
  }, [influenceUsers]);

  // ============================================================
  // REAL PROPAGATION GRAPH DATA
  // ============================================================

  const realPropagationGraphData = useMemo(() => {
    if (
      !propagationNetwork ||
      !Array.isArray(propagationNetwork.nodes) ||
      !Array.isArray(propagationNetwork.edges)
    ) {
      return { nodes: [], links: [] };
    }

    const nodeMap = new Map();

    propagationNetwork.nodes.forEach((node) => {
      const userId = Number(node.user_id);

      if (!Number.isFinite(userId)) return;

      nodeMap.set(String(userId), {
        id: String(userId),
        user_id: userId,
        posts: Number(node.posts || 0),
        propagation_posts: Number(
          node.propagation_posts || 0
        ),
        max_depth: Number(node.max_depth || 0),
        total_likes: Number(node.total_likes || 0),
        total_retweets: Number(
          node.total_retweets || 0
        ),
        in_degree: 0,
        out_degree: 0,
      });
    });

    const links = propagationNetwork.edges
      .map((edge) => ({
        source: String(edge.source),
        target: String(edge.target),
        depth: Number(edge.depth || 0),
        tweet_id: edge.tweet_id,
        parent_id: edge.parent_id,
        likes: Number(edge.likes || 0),
        retweets: Number(edge.retweets || 0),
        engagement:
          Number(edge.likes || 0) +
          Number(edge.retweets || 0),
      }))
      .filter(
        (edge) =>
          nodeMap.has(edge.source) &&
          nodeMap.has(edge.target) &&
          edge.source !== edge.target
      );

    links.forEach((link) => {
      const source = nodeMap.get(link.source);
      const target = nodeMap.get(link.target);

      if (source) source.out_degree += 1;
      if (target) target.in_degree += 1;
    });

    const nodes = Array.from(nodeMap.values()).map((node) => ({
      ...node,
      degree: node.in_degree + node.out_degree,
    }));

    return {
      nodes,
      links,
    };
  }, [propagationNetwork]);

  // Keep the real propagation graph responsive to its container.
  useEffect(() => {
    const wrapper = document.querySelector(".force-graph-wrapper");

    if (!wrapper) {
      return;
    }

    const updateGraphWidth = () => {
      setPropagationGraphWidth(
        Math.max(320, Math.floor(wrapper.clientWidth))
      );
    };

    updateGraphWidth();

    const resizeObserver = new ResizeObserver(updateGraphWidth);
    resizeObserver.observe(wrapper);

    return () => {
      resizeObserver.disconnect();
    };
  }, []);

  // Configure the D3 forces after the real network data arrives.
  // The API already provides the real propagation edges; this only
  // improves their visual layout in the browser.
  useEffect(() => {
    const graph = propagationGraphRef.current;

    if (
      !graph ||
      !realPropagationGraphData.nodes.length
    ) {
      return;
    }

    const charge = graph.d3Force("charge");
    const linkForce = graph.d3Force("link");

    if (charge) {
      charge
        .strength(-240)
        .distanceMax(700);
    }

    if (linkForce) {
      linkForce
        .distance((link) => {
          const depth = Number(link.depth || 1);

          if (depth === 1) return 95;
          if (depth === 2) return 75;
          return 60;
        })
        .strength(0.85);
    }

    graph.d3ReheatSimulation();

    const fitTimer = window.setTimeout(() => {
      graph.zoomToFit(700, 55);
    }, 1000);

    return () => {
      window.clearTimeout(fitTimer);
    };
  }, [realPropagationGraphData]);

  // ============================================================
  // UI
  // ============================================================

  return (
    <div className="app-shell">

      {/* ======================================================
          SIDEBAR
      ====================================================== */}

      <aside className="sidebar">

        <div className="brand">
          <div className="brand-mark">
            TN
          </div>

          <div>
            <h1>TRUTHNET</h1>
            <span>AI INTELLIGENCE</span>
          </div>
        </div>

        <nav className="navigation">

          <button
            className="nav-item active"
            onClick={() =>
              scrollToSection("dashboard")
            }
          >
            <span>◈</span>
            Dashboard
          </button>

          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("detection")
            }
          >
            <span>◉</span>
            Detection
          </button>

          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("propagation")
            }
          >
            <span>⌁</span>
            Propagation
          </button>

          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("influence")
            }
          >
            <span>◌</span>
            Influence Network
          </button>

          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("intervention")
            }
          >
            <span>◍</span>
            Intervention
          </button>

          <button
            className="nav-item"
            onClick={() =>
              scrollToSection("experiments")
            }
          >
            <span>▦</span>
            Experiments
          </button>

        </nav>

        <div className="sidebar-footer">
          <div className="status-dot"></div>

          <div>
            <strong>
              AI SYSTEM ONLINE
            </strong>

            <span>
              All services operational
            </span>
          </div>
        </div>

      </aside>

      {/* ======================================================
          MAIN
      ====================================================== */}

      <main
        className="main-content"
        id="dashboard"
      >

        {/* ====================================================
            TOP BAR
        ==================================================== */}

        <header className="topbar">

          <div>
            <p className="eyebrow">
              MISINFORMATION PROPAGATION
              INTELLIGENCE
            </p>

            <h2>
              Detection Command Center
            </h2>
          </div>

          <div className="system-status">
            <span className="online-dot"></span>
            SYSTEM ONLINE
          </div>

        </header>

        {/* ====================================================
            HERO
        ==================================================== */}

        <section className="hero">

          <div className="hero-copy">

            <span className="hero-label">
              GRAPH NEURAL NETWORK ANALYSIS
            </span>

            <h3>
              Detect misinformation.
              <br />
              Understand its propagation.
            </h3>

            <p>
              Analyze information cascades using a
              trained GraphSAGE model and investigate
              how misinformation spreads through
              social networks.
            </p>

          </div>

          <div className="hero-orbit">

            <div className="orbit orbit-one"></div>
            <div className="orbit orbit-two"></div>

            <div className="orbit-core">
              <span>AI</span>
            </div>

            <div className="orbit-node node-one"></div>
            <div className="orbit-node node-two"></div>
            <div className="orbit-node node-three"></div>

          </div>

        </section>

        {/* ====================================================
            MODULE 01 — DETECTION
        ==================================================== */}

        <section
          className="dashboard-grid"
          id="detection"
        >

          <div className="panel analysis-panel">

            <div className="panel-heading">

              <div>
                <span className="panel-kicker">
                  ANALYSIS ENGINE
                </span>

                <h3>
                  Analyze Information
                </h3>
              </div>

              <span className="panel-number">
                01
              </span>

            </div>

            <p className="panel-description">
              Select a propagation graph from the
              GossipCop evaluation dataset and run
              GraphSAGE inference.
            </p>

            <label className="input-label">
              GRAPH INDEX
            </label>

            <div className="input-row">

              <input
                type="number"
                min="0"
                max="3825"
                value={graphIndex}
                onChange={(event) =>
                  setGraphIndex(
                    event.target.value
                  )
                }
              />

              <button
                className="analyze-button"
                onClick={analyzeGraph}
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="spinner"></span>
                    ANALYZING
                  </>
                ) : (
                  <>
                    ANALYZE
                    <span>→</span>
                  </>
                )}
              </button>

            </div>

            <div className="input-hint">
              Test range: 0 — 3825
            </div>

            {error && (
              <div className="error-message">
                <strong>
                  Analysis failed
                </strong>

                <span>{error}</span>
              </div>
            )}

          </div>

          <div className="panel model-panel">

            <div className="panel-heading">

              <div>
                <span className="panel-kicker">
                  MODEL STATUS
                </span>

                <h3>
                  GraphSAGE
                </h3>
              </div>

              <span className="live-badge">
                LIVE
              </span>

            </div>

            <div className="metric-list">

              <div className="metric-row">
                <span>Test Accuracy</span>
                <strong>89.94%</strong>
              </div>

              <div className="metric-row">
                <span>ROC-AUC</span>
                <strong>96.00%</strong>
              </div>

              <div className="metric-row">
                <span>Architecture</span>
                <strong>GraphSAGE</strong>
              </div>

              <div className="metric-row">
                <span>Dataset</span>
                <strong>GossipCop</strong>
              </div>

            </div>

            <div className="model-bar">
              <div></div>
            </div>

            <span className="model-note">
              Trained propagation-aware graph
              classifier
            </span>

          </div>

        </section>

        {/* ====================================================
            GRAPH SAGE RESULT
        ==================================================== */}

        <section className="verdict-panel">

          <div className="verdict-header">

            <div>
              <span className="panel-kicker">
                AI VERDICT
              </span>

              <h3>
                Classification Result
              </h3>
            </div>

            {result && (
              <span className="graph-tag">
                GRAPH #{result.graph_index}
              </span>
            )}

          </div>

          {!result && !loading && (
            <div className="empty-verdict">

              <div className="empty-icon">
                ◌
              </div>

              <h4>
                Awaiting analysis
              </h4>

              <p>
                Run a graph analysis to generate
                an AI classification and confidence
                assessment.
              </p>

            </div>
          )}

          {loading && (
            <div className="empty-verdict">

              <div className="large-spinner"></div>

              <h4>
                Analyzing propagation graph
              </h4>

              <p>
                GraphSAGE is processing the selected
                network structure...
              </p>

            </div>
          )}

          {result && !loading && (
            <div className="result-content">

              <div
                className={`verdict-badge ${
                  result.class_id === 1
                    ? "fake"
                    : "real"
                }`}
              >

                <span className="verdict-symbol">
                  {result.class_id === 1
                    ? "!"
                    : "✓"}
                </span>

                <div>
                  <span>
                    CLASSIFICATION
                  </span>

                  <strong>
                    {String(
                      result.prediction || ""
                    ).toUpperCase()}
                  </strong>
                </div>

              </div>

              <div className="confidence-section">

                <div className="confidence-header">

                  <span>
                    MODEL CONFIDENCE
                  </span>

                  <strong>
                    {confidence.toFixed(2)}%
                  </strong>

                </div>

                <div className="confidence-track">

                  <div
                    className={`confidence-fill ${
                      result.class_id === 1
                        ? "fake-fill"
                        : "real-fill"
                    }`}
                    style={{
                      width: `${Math.min(
                        confidence,
                        100
                      )}%`,
                    }}
                  ></div>

                </div>

                <div className="probability-grid">

                  <div className="probability-card">
                    <span>REAL</span>

                    <strong>
                      {(
                        Number(
                          result.real_probability ||
                            0
                        ) * 100
                      ).toFixed(2)}
                      %
                    </strong>
                  </div>

                  <div className="probability-card">
                    <span>FAKE</span>

                    <strong>
                      {(
                        Number(
                          result.fake_probability ||
                            0
                        ) * 100
                      ).toFixed(2)}
                      %
                    </strong>
                  </div>

                </div>

              </div>

              <div className="result-meta">

                <div>
                  <span>MODEL</span>
                  <strong>
                    {result.model}
                  </strong>
                </div>

                <div>
                  <span>DATASET</span>
                  <strong>
                    {result.dataset}
                  </strong>
                </div>

                <div>
                  <span>GRAPH</span>
                  <strong>
                    #{result.graph_index}
                  </strong>
                </div>

              </div>

            </div>
          )}

        </section>

        {/* ====================================================
            MODULE 02 — PROPAGATION
        ==================================================== */}

        <section
          className="propagation-section"
          id="propagation"
        >

          <div className="section-heading">

            <div>
              <span className="panel-kicker">
                MODULE 02 · PROPAGATION INTELLIGENCE
              </span>

              <h3>
                Propagation Forecast
              </h3>

              <p>
                Predict future cascade growth from
                early propagation signals using the
                trained FibVID Random Forest models.
              </p>
            </div>

            <span className="panel-number">
              02
            </span>

          </div>

          <div className="propagation-layout">

            <div className="propagation-input-panel">

              <label className="input-label">
                OBSERVATION WINDOW
              </label>

              <div className="window-selector">

                {[1, 6, 24].map((window) => (
                  <button
                    key={window}
                    className={
                      propagationWindow === window
                        ? "window-button selected"
                        : "window-button"
                    }
                    onClick={() =>
                      setPropagationWindow(
                        window
                      )
                    }
                  >
                    {window}H
                  </button>
                ))}

              </div>

              <div className="propagation-form">

                {[
                  [
                    "early_tweets",
                    "EARLY TWEETS",
                  ],
                  [
                    "early_users",
                    "UNIQUE USERS",
                  ],
                  [
                    "early_max_depth",
                    "MAX PROPAGATION DEPTH",
                  ],
                  [
                    "early_likes",
                    "TOTAL LIKES",
                  ],
                  [
                    "early_retweets",
                    "TOTAL RETWEETS",
                  ],
                  [
                    "engagement",
                    "ENGAGEMENT",
                  ],
                ].map(([name, label]) => (
                  <div
                    className="form-field"
                    key={name}
                  >
                    <label>{label}</label>

                    <input
                      type="number"
                      min="0"
                      name={name}
                      value={
                        propagationInputs[name]
                      }
                      onChange={
                        handlePropagationInput
                      }
                    />
                  </div>
                ))}

              </div>

              <button
                className="propagation-button"
                onClick={predictPropagation}
                disabled={propagationLoading}
              >
                {propagationLoading ? (
                  <>
                    <span className="spinner"></span>
                    FORECASTING
                  </>
                ) : (
                  <>
                    GENERATE FORECAST
                    <span>→</span>
                  </>
                )}
              </button>

              {propagationError && (
                <div className="error-message">
                  <strong>
                    Forecast failed
                  </strong>

                  <span>
                    {propagationError}
                  </span>
                </div>
              )}

            </div>

            <div className="propagation-result-panel">

              {!propagationResult &&
                !propagationLoading && (
                  <div className="propagation-empty">

                    <span className="propagation-icon">
                      ↗
                    </span>

                    <h4>
                      Forecast awaiting input
                    </h4>

                    <p>
                      Provide early propagation
                      signals and generate a
                      cascade forecast.
                    </p>

                  </div>
                )}

              {propagationLoading && (
                <div className="propagation-empty">

                  <div className="large-spinner"></div>

                  <h4>
                    Running propagation model
                  </h4>

                  <p>
                    Random Forest is estimating
                    future cascade growth...
                  </p>

                </div>
              )}

              {propagationResult &&
                !propagationLoading && (
                  <div className="propagation-result">

                    <div className="forecast-header">

                      <div>
                        <span className="panel-kicker">
                          FORECAST RESULT
                        </span>

                        <h4>
                          Predicted Cascade Size
                        </h4>
                      </div>

                      <span className="forecast-window">
                        {
                          propagationResult.window_hours
                        }H
                      </span>

                    </div>

                    <div className="cascade-value">

                      <strong>
                        {Math.round(
                          Number(
                            propagationResult
                              .predicted_cascade_size ||
                              0
                          )
                        ).toLocaleString()}
                      </strong>

                      <span>
                        estimated final reach
                      </span>

                    </div>

                    <div className="risk-display">

                      <span>
                        PROPAGATION RISK
                      </span>

                      <strong
                        className={`risk-${propagationRisk.toLowerCase()}`}
                      >
                        {propagationRisk}
                      </strong>

                    </div>

                    <div className="forecast-metrics">

                      <div>
                        <span>
                          LOG CASCADE
                        </span>

                        <strong>
                          {
                            propagationResult
                              .predicted_log_cascade_size
                          }
                        </strong>
                      </div>

                      <div>
                        <span>MODEL</span>

                        <strong>
                          {
                            propagationResult.model
                          }
                        </strong>
                      </div>

                      <div>
                        <span>DATASET</span>

                        <strong>
                          {
                            propagationResult.dataset
                          }
                        </strong>
                      </div>

                    </div>

                    <div className="forecast-note">
                      Prediction generated from{" "}
                      {
                        propagationResult.window_hours
                      }
                      -hour early propagation
                      signals.
                    </div>

                  </div>
                )}

            </div>

          </div>

        </section>

        {/* ====================================================
            MODULE 03 — INFLUENCE
        ==================================================== */}

        <section
          className="influence-section"
          id="influence"
        >

          <div className="section-heading">

            <div>
              <span className="panel-kicker">
                MODULE 03 · NETWORK INTELLIGENCE
              </span>

              <h3>
                Influence Network
              </h3>

              <p>
                Identify high-impact users,
                propagation bridges and critical
                intervention targets using the
                FibVID propagation network.
              </p>
            </div>

            <span className="panel-number">
              03
            </span>

          </div>

          <div className="influence-controls">

            <div>

              <label className="input-label">
                CASCADE / CLAIM
              </label>

              <input
                type="number"
                min="0"
                value={influenceCascade}
                onChange={(event) =>
                  setInfluenceCascade(
                    Number(event.target.value)
                  )
                }
              />

            </div>

            <button
              className="propagation-button"
              onClick={loadInfluenceNetwork}
              disabled={influenceLoading}
            >
              {influenceLoading ? (
                <>
                  <span className="spinner"></span>
                  ANALYZING NETWORK
                </>
              ) : (
                <>
                  ANALYZE INFLUENCE
                  <span>→</span>
                </>
              )}
            </button>

          </div>

          {influenceError && (
            <div className="error-message">

              <strong>
                Influence analysis failed
              </strong>

              <span>
                {influenceError}
              </span>

            </div>
          )}

          {influenceUsers.length > 0 && (
            <>

              <div className="influence-summary">

                <div className="influence-stat">
                  <span>CASCADE</span>
                  <strong>
                    #{influenceCascade}
                  </strong>
                </div>

                <div className="influence-stat">
                  <span>NETWORK USERS</span>
                  <strong>
                    {influenceTotalUsers.toLocaleString()}
                  </strong>
                </div>

                <div className="influence-stat">
                  <span>TOP INFLUENCE</span>
                  <strong>
                    {Number(
                      topInfluenceUser?.influence_score ||
                        0
                    ).toFixed(4)}
                  </strong>
                </div>

                <div className="influence-stat">
                  <span>CRITICAL TARGETS</span>
                  <strong>
                    {
                      influenceUsers.filter(
                        (user) =>
                          user.intervention_priority ===
                          "Critical"
                      ).length
                    }
                  </strong>
                </div>

              </div>

              {/* ==================================================
                  INFLUENCE CHART
              ================================================== */}

              <div className="network-graph-panel">

                <div className="table-heading">

                  <div>
                    <span className="panel-kicker">
                      REAL PROPAGATION TOPOLOGY
                    </span>

                    <h4>
                      Actual Cascade Propagation Network
                    </h4>

                    <p className="graph-description">
                      Observed user-to-user propagation edges extracted
                      directly from the FibVID cascade dataset.
                    </p>
                  </div>

                  <div
                    style={{
                      display: "flex",
                      gap: "10px",
                      alignItems: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <span className="network-node-count">
                      {realPropagationGraphData.nodes.length} USERS
                    </span>

                    <span className="network-node-count">
                      {realPropagationGraphData.links.length} EDGES
                    </span>

                    <button
                      className="propagation-button"
                      onClick={() =>
                        loadPropagationNetwork(
                          Number(influenceCascade)
                        )
                      }
                      disabled={propagationNetworkLoading}
                    >
                      {propagationNetworkLoading
                        ? "LOADING..."
                        : "REFRESH NETWORK →"}
                    </button>
                  </div>

                </div>

                {propagationNetworkError && (
                  <div className="error-message">
                    <strong>
                      Propagation network failed
                    </strong>

                    <span>
                      {propagationNetworkError}
                    </span>
                  </div>
                )}

                <div
                  className="force-graph-wrapper"
                  style={{
                    position: "relative",
                    width: "100%",
                    overflow: "hidden",
                  }}
                >

                  {propagationNetworkLoading &&
                  realPropagationGraphData.nodes.length === 0 ? (
                    <div className="influence-empty">
                      <div className="large-spinner"></div>

                      <h4>
                        Loading real propagation network
                      </h4>

                      <p>
                        Building the observed user-to-user cascade graph...
                      </p>
                    </div>
                  ) : realPropagationGraphData.nodes.length === 0 ? (
                    <div className="influence-empty">
                      <div className="propagation-icon">
                        ◎
                      </div>

                      <h4>
                        Propagation network awaiting data
                      </h4>

                      <p>
                        Refresh the network to load the observed
                        FibVID propagation edges.
                      </p>
                    </div>
                  ) : (
                    <>
                      <ForceGraph2D
                        ref={propagationGraphRef}
                        graphData={realPropagationGraphData}
                        width={propagationGraphWidth}
                        height={420}
                        backgroundColor="transparent"

                        /* ------------------------------
                           PHYSICS / LAYOUT
                        ------------------------------ */

                        cooldownTicks={180}
                        warmupTicks={80}
                        d3AlphaDecay={0.022}
                        d3VelocityDecay={0.28}
                        nodeRelSize={5}

                        enableNodeDrag={true}
                        enableZoomInteraction={true}
                        enablePanInteraction={true}

                        /* ------------------------------
                           LINKS
                        ------------------------------ */

                        linkWidth={(link) => {
                          const engagement =
                            Number(link.engagement || 0);

                          return Math.max(
                            1,
                            Math.min(
                              5,
                              1 +
                                Math.log10(
                                  engagement + 1
                                ) * 0.65
                            )
                          );
                        }}

                        linkColor={(link) => {
                          const depth =
                            Number(link.depth || 1);

                          if (depth === 1) {
                            return "rgba(0, 217, 255, 0.72)";
                          }

                          if (depth === 2) {
                            return "rgba(0, 217, 255, 0.48)";
                          }

                          return "rgba(255, 77, 109, 0.55)";
                        }}

                        linkDirectionalArrowLength={7}
                        linkDirectionalArrowRelPos={0.94}

                        linkDirectionalParticles={(link) =>
                          Number(link.engagement || 0) > 100000
                            ? 2
                            : 1
                        }

                        linkDirectionalParticleWidth={2}

                        linkDirectionalParticleSpeed={0.004}

                        onEngineStop={() => {
                          requestAnimationFrame(() => {
                            propagationGraphRef.current?.zoomToFit(500, 70);
                          });
                        }}

                        linkCurvature={(link) => {
                          const tweetId =
                            Number(link.tweet_id || 0);

                          return (
                            ((tweetId % 5) - 2) * 0.055
                          );
                        }}

                        /* ------------------------------
                           NODE SIZE
                        ------------------------------ */

                        nodeVal={(node) => {
                          const degree =
                            Number(node.degree || 0);

                          const propagationPosts =
                            Number(
                              node.propagation_posts || 0
                            );

                          return Math.max(
                            4,
                            Math.min(
                              18,
                              4 +
                                Math.sqrt(
                                  degree +
                                    propagationPosts
                                ) *
                                  2.2
                            )
                          );
                        }}

                        /* ------------------------------
                           DISABLE THE DEFAULT BLACK
                           BROWSER TOOLTIP
                        ------------------------------ */

                        nodeLabel={() => ""}

                        /* ------------------------------
                           INTERACTION
                        ------------------------------ */

                        onNodeHover={(node) => {
                          setHoveredPropagationNode(
                            node || null
                          );
                        }}

                        onNodeClick={(node) => {
                          setSelectedPropagationNode(node);
                        }}

                        onBackgroundClick={() => {
                          setSelectedPropagationNode(null);
                          setHoveredPropagationNode(null);
                        }}

                        /* ------------------------------
                           CUSTOM CANVAS NODES
                        ------------------------------ */

                        nodeCanvasObject={(
                          node,
                          ctx,
                          globalScale
                        ) => {
                          const degree =
                            Number(node.degree || 0);

                          const propagationPosts =
                            Number(
                              node.propagation_posts || 0
                            );

                          const isSelected =
                            selectedPropagationNode?.id ===
                            node.id;

                          const isHovered =
                            hoveredPropagationNode?.id ===
                            node.id;

                          const isHub =
                            degree >= 8 ||
                            propagationPosts >= 5;

                          const radius = Math.max(
                            4.5,
                            Math.min(
                              17,
                              4.5 +
                                Math.sqrt(
                                  degree +
                                    propagationPosts
                                ) *
                                  1.9
                            )
                          );

                          /* Outer glow for important nodes */
                          if (
                            isHub ||
                            isSelected ||
                            isHovered
                          ) {
                            ctx.beginPath();

                            ctx.arc(
                              node.x,
                              node.y,
                              radius + 5,
                              0,
                              2 * Math.PI
                            );

                            ctx.fillStyle =
                              isSelected
                                ? "rgba(255,255,255,0.16)"
                                : "rgba(0,217,255,0.10)";

                            ctx.fill();
                          }

                          /* Node body */
                          ctx.beginPath();

                          ctx.arc(
                            node.x,
                            node.y,
                            radius,
                            0,
                            2 * Math.PI
                          );

                          if (isSelected) {
                            ctx.fillStyle = "#ffffff";
                          } else if (isHub) {
                            ctx.fillStyle = "#ff4d6d";
                          } else {
                            ctx.fillStyle = "#00d9ff";
                          }

                          ctx.fill();

                          ctx.strokeStyle = isSelected
                            ? "#00d9ff"
                            : "rgba(255,255,255,0.82)";

                          ctx.lineWidth = isSelected
                            ? 2.5
                            : 1;

                          ctx.stroke();

                          /* Labels only for important nodes
                             or when zoomed in sufficiently. */
                          if (
                            isSelected ||
                            isHovered ||
                            isHub ||
                            globalScale > 2.4
                          ) {
                            const fontSize = Math.max(
                              9,
                              Math.min(
                                14,
                                12 / globalScale
                              )
                            );

                            ctx.font = `600 ${fontSize}px Inter, sans-serif`;
                            ctx.textAlign = "center";
                            ctx.textBaseline = "top";

                            ctx.fillStyle =
                              "#ffffff";

                            ctx.shadowColor =
                              "rgba(0,0,0,0.85)";

                            ctx.shadowBlur = 4;

                            ctx.fillText(
                              `#${node.user_id}`,
                              node.x,
                              node.y +
                                radius +
                                5
                            );

                            ctx.shadowBlur = 0;
                          }
                        }}

                        nodePointerAreaPaint={(
                          node,
                          color,
                          ctx
                        ) => {
                          const degree =
                            Number(node.degree || 0);

                          const posts =
                            Number(
                              node.propagation_posts || 0
                            );

                          const radius = Math.max(
                            7,
                            Math.min(
                              22,
                              7 +
                                Math.sqrt(
                                  degree + posts
                                ) *
                                  2
                            )
                          );

                          ctx.fillStyle = color;

                          ctx.beginPath();

                          ctx.arc(
                            node.x,
                            node.y,
                            radius,
                            0,
                            2 * Math.PI
                          );

                          ctx.fill();
                        }}
                      />

                      {/* --------------------------------
                          CLEAN HTML HOVER CARD
                      -------------------------------- */}

                      {hoveredPropagationNode && (
                        <div
                          style={{
                            position: "absolute",
                            top: "18px",
                            right: "18px",
                            width: "245px",
                            padding: "14px 16px",
                            background:
                              "rgba(8, 14, 20, 0.94)",
                            border:
                              "1px solid rgba(0,217,255,0.35)",
                            boxShadow:
                              "0 12px 30px rgba(0,0,0,0.35)",
                            pointerEvents: "none",
                            zIndex: 10,
                          }}
                        >
                          <div
                            style={{
                              color: "#00d9ff",
                              fontSize: "10px",
                              letterSpacing:
                                "0.18em",
                              fontWeight: 700,
                              marginBottom: "8px",
                            }}
                          >
                            PROPAGATION NODE
                          </div>

                          <div
                            style={{
                              color: "#ffffff",
                              fontSize: "18px",
                              fontWeight: 700,
                              marginBottom: "10px",
                            }}
                          >
                            #{hoveredPropagationNode.user_id}
                          </div>

                          <div
                            style={{
                              display: "grid",
                              gridTemplateColumns:
                                "1fr 1fr",
                              gap: "8px",
                              color:
                                "rgba(255,255,255,0.72)",
                              fontSize: "11px",
                            }}
                          >
                            <span>
                              POSTS{" "}
                              <strong
                                style={{
                                  color: "#ffffff",
                                  display: "block",
                                }}
                              >
                                {Number(
                                  hoveredPropagationNode.posts ||
                                    0
                                ).toLocaleString()}
                              </strong>
                            </span>

                            <span>
                              PROPAGATION{" "}
                              <strong
                                style={{
                                  color: "#ffffff",
                                  display: "block",
                                }}
                              >
                                {Number(
                                  hoveredPropagationNode.propagation_posts ||
                                    0
                                ).toLocaleString()}
                              </strong>
                            </span>

                            <span>
                              DEPTH{" "}
                              <strong
                                style={{
                                  color: "#ffffff",
                                  display: "block",
                                }}
                              >
                                {Number(
                                  hoveredPropagationNode.max_depth ||
                                    0
                                )}
                              </strong>
                            </span>

                            <span>
                              DEGREE{" "}
                              <strong
                                style={{
                                  color: "#ffffff",
                                  display: "block",
                                }}
                              >
                                {Number(
                                  hoveredPropagationNode.degree ||
                                    0
                                )}
                              </strong>
                            </span>
                          </div>
                        </div>
                      )}
                    </>
                  )}

                </div>

                <div className="network-legend">

                  <div className="network-legend-item">
                    <span className="legend-dot normal"></span>
                    Depth 1 propagation
                  </div>

                  <div className="network-legend-item">
                    <span className="legend-dot critical"></span>
                    Deeper propagation
                  </div>

                  <span className="network-legend-hint">
                    Arrow direction = observed propagation ·
                    Drag · Zoom · Click node
                  </span>

                </div>

                <div className="influence-summary">

                  <div className="influence-stat">
                    <span>CASCADE RECORDS</span>

                    <strong>
                      {Number(
                        propagationNetwork.total_records || 0
                      ).toLocaleString()}
                    </strong>
                  </div>

                  <div className="influence-stat">
                    <span>OBSERVED EDGES</span>

                    <strong>
                      {Number(
                        propagationNetwork.total_edges || 0
                      ).toLocaleString()}
                    </strong>
                  </div>

                  <div className="influence-stat">
                    <span>NETWORK USERS</span>

                    <strong>
                      {realPropagationGraphData.nodes.length.toLocaleString()}
                    </strong>
                  </div>

                  <div className="influence-stat">
                    <span>SELECTED USER</span>

                    <strong>
                      {selectedPropagationNode
                        ? `#${selectedPropagationNode.user_id}`
                        : "—"}
                    </strong>
                  </div>

                </div>

                {selectedPropagationNode && (
                  <div className="influence-highlight">

                    <div>
                      <span className="panel-kicker">
                        SELECTED PROPAGATION NODE
                      </span>

                      <h4>
                        USER #
                        {selectedPropagationNode.user_id}
                      </h4>

                      <p>
                        {
                          selectedPropagationNode.propagation_posts
                        }{" "}
                        propagation posts · max depth{" "}
                        {
                          selectedPropagationNode.max_depth
                        }{" "}
                        · degree{" "}
                        {
                          selectedPropagationNode.degree ||
                          0
                        }
                      </p>
                    </div>

                    <div className="influence-score-large">
                      <span>RETWEETS</span>

                      <strong>
                        {Number(
                          selectedPropagationNode.total_retweets ||
                            0
                        ).toLocaleString()}
                      </strong>
                    </div>

                  </div>
                )}

              </div>

              <div className="analytics-panel">

                <div className="table-heading">

                  <div>
                    <span className="panel-kicker">
                      NETWORK ANALYTICS
                    </span>

                    <h4>
                      Top Influence Scores
                    </h4>
                  </div>

                </div>

                <div className="chart-container">

                  <ResponsiveContainer
                    width="100%"
                    height={350}
                  >
                    <BarChart
                      data={influenceChartData}
                      margin={{
                        top: 20,
                        right: 20,
                        left: 10,
                        bottom: 20,
                      }}
                    >

                      <CartesianGrid
                        strokeDasharray="3 3"
                        opacity={0.15}
                      />

                      <XAxis
                        dataKey="user"
                      />

                      <YAxis />

                      <Tooltip />

                      <Legend />

                      <Bar
                        dataKey="influence"
                        name="Influence Score"
                        fill="#00d9ff"
                      />

                    </BarChart>
                  </ResponsiveContainer>

                </div>

              </div>

              <div className="influence-highlight">

                <div>

                  <span className="panel-kicker">
                    HIGHEST INFLUENCE
                  </span>

                  <h4>
                    USER #
                    {topInfluenceUser?.user_id}
                  </h4>

                  <p>
                    {topInfluenceUser?.network_role}
                    {" · "}
                    {
                      topInfluenceUser?.intervention_priority
                    }
                  </p>

                </div>

                <div className="influence-score-large">

                  <span>
                    INFLUENCE SCORE
                  </span>

                  <strong>
                    {Number(
                      topInfluenceUser?.influence_score ||
                        0
                    ).toFixed(6)}
                  </strong>

                </div>

              </div>

              <div className="influence-table-container">

                <div className="table-heading">

                  <div>

                    <span className="panel-kicker">
                      RANKED NETWORK NODES
                    </span>

                    <h4>
                      Top Influential Users
                    </h4>

                  </div>

                  <span>
                    TOP {influenceUsers.length}
                  </span>

                </div>

                <div className="influence-table-wrapper">

                  <table className="influence-table">

                    <thead>
                      <tr>
                        <th>RANK</th>
                        <th>USER</th>
                        <th>ROLE</th>
                        <th>INFLUENCE</th>
                        <th>PAGERANK</th>
                        <th>WEIGHTED DEGREE</th>
                        <th>POSTS</th>
                        <th>PRIORITY</th>
                      </tr>
                    </thead>

                    <tbody>

                      {influenceUsers.map(
                        (user, index) => (
                          <tr
                            key={`${user.user_id}-${index}`}
                          >

                            <td>
                              <span className="rank-number">
                                {index + 1}
                              </span>
                            </td>

                            <td>
                              <strong>
                                #{user.user_id}
                              </strong>
                            </td>

                            <td>
                              <span
                                className={`role-badge role-${String(
                                  user.network_role || ""
                                )
                                  .toLowerCase()
                                  .replace(
                                    /\s+/g,
                                    "-"
                                  )}`}
                              >
                                {user.network_role}
                              </span>
                            </td>

                            <td>
                              <strong className="influence-value">
                                {Number(
                                  user.influence_score ||
                                    0
                                ).toFixed(4)}
                              </strong>
                            </td>

                            <td>
                              {Number(
                                user.pagerank || 0
                              ).toFixed(6)}
                            </td>

                            <td>
                              {Number(
                                user.weighted_out_degree ||
                                  0
                              ).toLocaleString()}
                            </td>

                            <td>
                              {user.propagation_posts}
                            </td>

                            <td>
                              <span
                                className={`priority-badge priority-${String(
                                  user.intervention_priority ||
                                    ""
                                ).toLowerCase()}`}
                              >
                                {
                                  user.intervention_priority
                                }
                              </span>
                            </td>

                          </tr>
                        )
                      )}

                    </tbody>

                  </table>

                </div>

              </div>

            </>
          )}

          {!influenceLoading &&
            influenceUsers.length === 0 &&
            !influenceError && (
              <div className="influence-empty">

                <div className="propagation-icon">
                  ◎
                </div>

                <h4>
                  Influence analysis awaiting input
                </h4>

                <p>
                  Select a FibVID cascade and run
                  the network analysis to identify
                  influential propagation nodes.
                </p>

              </div>
            )}

        </section>

        {/* ====================================================
            MODULE 04 — INTERVENTION
        ==================================================== */}

        <section
          className="intervention-section"
          id="intervention"
        >

          <div className="section-heading">

            <div>
              <span className="panel-kicker">
                MODULE 04 · INTERVENTION INTELLIGENCE
              </span>

              <h3>
                Intervention Simulator
              </h3>

              <p>
                Compare fact checking, content
                moderation and targeted intervention
                across different propagation timelines.
              </p>
            </div>

            <span className="panel-number">
              04
            </span>

          </div>

          <div className="intervention-layout">

            <div className="intervention-control-panel">

              <span className="panel-kicker">
                CASCADE ANALYSIS
              </span>

              <h4>
                Select Propagation Cascade
              </h4>

              <p>
                Load causal intervention results for
                the selected cascade.
              </p>

              <label className="input-label">
                CASCADE / CLAIM ID
              </label>

              <input
                className="large-input"
                type="number"
                min="0"
                value={interventionCascade}
                onChange={(event) =>
                  setInterventionCascade(
                    Number(event.target.value)
                  )
                }
              />

              <button
                className="propagation-button"
                onClick={loadIntervention}
                disabled={interventionLoading}
              >
                {interventionLoading ? (
                  <>
                    <span className="spinner"></span>
                    LOADING ANALYSIS
                  </>
                ) : (
                  <>
                    RUN INTERVENTION ANALYSIS
                    <span>→</span>
                  </>
                )}
              </button>

              {interventionError && (
                <div className="error-message">

                  <strong>
                    Intervention analysis failed
                  </strong>

                  <span>
                    {interventionError}
                  </span>

                </div>
              )}

            </div>

            <div className="intervention-result-panel">

              {!interventionCausal &&
                !interventionLoading && (
                  <div className="intervention-empty">

                    <div className="propagation-icon">
                      ◍
                    </div>

                    <h4>
                      Intervention analysis awaiting
                      input
                    </h4>

                    <p>
                      Select a cascade and run the
                      intervention analysis to compare
                      mitigation strategies.
                    </p>

                  </div>
                )}

              {interventionLoading && (
                <div className="intervention-empty">

                  <div className="large-spinner"></div>

                  <h4>
                    Running intervention analysis
                  </h4>

                  <p>
                    Loading validated causal
                    intervention results...
                  </p>

                </div>
              )}

              {interventionCausal &&
                !interventionLoading && (
                  <>

                    <div className="intervention-header">

                      <div>

                        <span className="panel-kicker">
                          CAUSAL INTERVENTION
                        </span>

                        <h4>
                          Cascade #
                          {
                            interventionCausal.cascade_id
                          }
                        </h4>

                      </div>

                      <span className="live-badge">
                        VALIDATED
                      </span>

                    </div>

                    <div className="intervention-baseline">

                      <div>
                        <span>
                          BASELINE REACH
                        </span>

                        <strong>
                          {Number(
                            interventionCausal
                              .baseline_reach || 0
                          ).toLocaleString()}
                        </strong>
                      </div>

                      <div>
                        <span>
                          BASELINE EDGES
                        </span>

                        <strong>
                          {Number(
                            interventionCausal
                              .baseline_edges || 0
                          ).toLocaleString()}
                        </strong>
                      </div>

                    </div>

                    <div className="strategy-grid">

                      {interventionResults.map(
                        (item, index) => (
                          <div
                            className="strategy-card"
                            key={`${item.strategy}-${item.timing}-${index}`}
                          >

                            <div className="strategy-card-top">

                              <span>
                                {String(
                                  item.strategy || ""
                                ).replace(
                                  /_/g,
                                  " "
                                )}
                              </span>

                              <strong>
                                {String(
                                  item.timing || ""
                                ).toUpperCase()}
                              </strong>

                            </div>

                            <div className="strategy-value">
                              {Number(
                                item.reach_reduction_percent ||
                                  0
                              ).toFixed(2)}
                              %
                            </div>

                            <span className="strategy-label">
                              REACH REDUCTION
                            </span>

                            <div className="strategy-progress">

                              <div
                                style={{
                                  width: `${Math.min(
                                    Number(
                                      item.reach_reduction_percent ||
                                        0
                                    ),
                                    100
                                  )}%`,
                                }}
                              ></div>

                            </div>

                            <div className="strategy-meta">
                              <span>
                                Final reach
                              </span>

                              <strong>
                                {Number(
                                  item.final_reach || 0
                                ).toLocaleString()}
                              </strong>
                            </div>

                            <div className="strategy-meta">
                              <span>
                                Edge reduction
                              </span>

                              <strong>
                                {Number(
                                  item.edge_reduction_percent ||
                                    0
                                ).toFixed(2)}
                                %
                              </strong>
                            </div>

                          </div>
                        )
                      )}

                    </div>

                    {/* ==================================================
                        INTERVENTION CHART
                    ================================================== */}

                    <div className="analytics-panel">

                      <div className="table-heading">

                        <div>

                          <span className="panel-kicker">
                            CAUSAL EFFECTIVENESS
                          </span>

                          <h4>
                            Reach vs Edge Reduction
                          </h4>

                        </div>

                      </div>

                      <div className="chart-container">

                        <ResponsiveContainer
                          width="100%"
                          height={400}
                        >
                          <BarChart
                            data={interventionChartData}
                            margin={{
                              top: 20,
                              right: 20,
                              left: 10,
                              bottom: 100,
                            }}
                          >

                            <CartesianGrid
                              strokeDasharray="3 3"
                              opacity={0.15}
                            />

                            <XAxis
                              dataKey="name"
                              angle={-35}
                              textAnchor="end"
                              interval={0}
                              height={100}
                            />

                            <YAxis
                              domain={[0, 100]}
                            />

                            <Tooltip />

                            <Legend />

                            <Bar
                              dataKey="reachReduction"
                              name="Reach Reduction %"
                              fill="#00d9ff"
                            />

                            <Bar
                              dataKey="edgeReduction"
                              name="Edge Reduction %"
                              fill="#7c5cff"
                            />

                          </BarChart>
                        </ResponsiveContainer>

                      </div>

                    </div>

                  </>
                )}

            </div>

          </div>

          {interventionSummary && (
            <div className="intervention-summary-panel">

              <div className="table-heading">

                <div>

                  <span className="panel-kicker">
                    MULTI-CASCADE VALIDATION
                  </span>

                  <h4>
                    Intervention Effectiveness
                  </h4>

                </div>

                <span>
                  {
                    interventionSummary
                      .cascades_validated
                  } CASCADES
                </span>

              </div>

              <div className="summary-metrics">

                {(
                  interventionSummary.results || []
                ).map((item, index) => (
                  <div
                    className="summary-metric"
                    key={`${item.strategy}-${item.timing}-${index}`}
                  >

                    <span>
                      {String(
                        item.strategy || ""
                      ).replace(
                        /_/g,
                        " "
                      )}
                    </span>

                    <strong>
                      {String(
                        item.timing || ""
                      ).toUpperCase()}
                    </strong>

                    <em>
                      {Number(
                        item.mean || 0
                      ).toFixed(2)}
                      %
                    </em>

                  </div>
                ))}

              </div>

            </div>
          )}

        </section>

        {/* ====================================================
            MODULE 05 — RESEARCH EXPERIMENTS
        ==================================================== */}

        <section
          className="experiments-section"
          id="experiments"
        >

          <div className="experiments-header">

            <div>

              <span className="panel-kicker">
                MODULE 05 · RESEARCH VALIDATION
              </span>

              <h3>
                Research Experiments
              </h3>

              <p>
                Explore validated multi-cascade
                intervention experiments and compare
                mitigation effectiveness across
                propagation timelines.
              </p>

            </div>

            <span className="panel-number">
              05
            </span>

          </div>

          <div className="experiments-layout">

            {/* LEFT CONTROL */}

            <div className="experiment-control">

              <span className="panel-kicker">
                EXPERIMENT DATASET
              </span>

              <h3>
                Multi-Cascade Causal Validation
              </h3>

              <p>
                Load the validated Experiment 03
                summary and compare intervention
                strategy performance across early,
                mid, and late intervention windows.
              </p>

              <div className="experiment-field">

                <label>
                  EXPERIMENT
                </label>

                <div className="experiment-tag">
                  EXPERIMENT 03 · MULTI-CASCADE
                  VALIDATION
                </div>

              </div>

              <div className="experiment-field">

                <label>
                  METRIC
                </label>

                <div className="experiment-tag">
                  REACH REDUCTION PERCENT
                </div>

              </div>

              <button
                className="experiment-button"
                onClick={loadExperimentResults}
                disabled={experimentLoading}
              >

                {experimentLoading ? (
                  <>
                    <span className="spinner"></span>
                    LOADING RESULTS
                  </>
                ) : (
                  <>
                    LOAD EXPERIMENT RESULTS
                    <span>→</span>
                  </>
                )}

              </button>

              {experimentError && (
                <div className="error-message">

                  <strong>
                    Experiment loading failed
                  </strong>

                  <span>
                    {experimentError}
                  </span>

                </div>
              )}

            </div>

            {/* RIGHT RESULTS */}

            <div className="experiment-results">

              {!experimentData &&
                !experimentLoading && (
                  <div className="experiment-empty">

                    <span className="propagation-icon">
                      ▦
                    </span>

                    <h4>
                      Experiment results awaiting
                      input
                    </h4>

                    <p>
                      Load the validated experiment
                      results to compare intervention
                      performance.
                    </p>

                  </div>
                )}

              {experimentLoading && (
                <div className="experiment-empty">

                  <div className="large-spinner"></div>

                  <h4>
                    Loading validated experiment
                  </h4>

                  <p>
                    Retrieving multi-cascade
                    intervention results...
                  </p>

                </div>
              )}

              {experimentData &&
                !experimentLoading && (
                  <>

                    <div className="experiment-result-header">

                      <div>

                        <span className="panel-kicker">
                          VALIDATED EXPERIMENT
                        </span>

                        <h3>
                          Experiment 03 —
                          Multi-Cascade Causal
                          Validation
                        </h3>

                        <p>
                          Metric:{" "}
                          {experimentData.metric}
                        </p>

                      </div>

                      <span className="validation-badge">
                        {
                          experimentData
                            .cascades_validated
                        } CASCADES VALIDATED
                      </span>

                    </div>

                    {/* ==================================================
                        TIMING CHART
                    ================================================== */}

                    <div className="analytics-panel">

                      <div className="table-heading">

                        <div>

                          <span className="panel-kicker">
                            TEMPORAL ANALYSIS
                          </span>

                          <h4>
                            Intervention Timing
                            Effectiveness
                          </h4>

                        </div>

                      </div>

                      <div className="chart-container">

                        <ResponsiveContainer
                          width="100%"
                          height={350}
                        >
                          <BarChart
                            data={timingChartData}
                            margin={{
                              top: 20,
                              right: 20,
                              left: 10,
                              bottom: 20,
                            }}
                          >

                            <CartesianGrid
                              strokeDasharray="3 3"
                              opacity={0.15}
                            />

                            <XAxis
                              dataKey="timing"
                            />

                            <YAxis
                              domain={[0, 100]}
                            />

                            <Tooltip />

                            <Legend />

                            <Bar
                              dataKey="reduction"
                              name="Mean Reach Reduction %"
                              fill="#00d9ff"
                            />

                          </BarChart>
                        </ResponsiveContainer>

                      </div>

                    </div>

                    {/* ==================================================
                        STRATEGY CHART
                    ================================================== */}

                    <div className="analytics-panel">

                      <div className="table-heading">

                        <div>

                          <span className="panel-kicker">
                            STRATEGY ANALYSIS
                          </span>

                          <h4>
                            Average Strategy
                            Effectiveness
                          </h4>

                        </div>

                      </div>

                      <div className="chart-container">

                        <ResponsiveContainer
                          width="100%"
                          height={350}
                        >
                          <BarChart
                            data={strategyChartData}
                            margin={{
                              top: 20,
                              right: 20,
                              left: 10,
                              bottom: 30,
                            }}
                          >

                            <CartesianGrid
                              strokeDasharray="3 3"
                              opacity={0.15}
                            />

                            <XAxis
                              dataKey="strategy"
                            />

                            <YAxis
                              domain={[0, 100]}
                            />

                            <Tooltip />

                            <Legend />

                            <Bar
                              dataKey="reduction"
                              name="Average Reach Reduction %"
                              fill="#7c5cff"
                            />

                          </BarChart>
                        </ResponsiveContainer>

                      </div>

                    </div>

                    {/* ==================================================
                        AVERAGES
                    ================================================== */}

                    <div className="experiment-average-grid">

                      <div className="average-card">

                        <span>
                          EARLY AVERAGE
                        </span>

                        <strong>
                          {earlyAverage.toFixed(2)}%
                        </strong>

                      </div>

                      <div className="average-card">

                        <span>
                          MID AVERAGE
                        </span>

                        <strong>
                          {midAverage.toFixed(2)}%
                        </strong>

                      </div>

                      <div className="average-card">

                        <span>
                          LATE AVERAGE
                        </span>

                        <strong>
                          {lateAverage.toFixed(2)}%
                        </strong>

                      </div>

                    </div>

                    {/* ==================================================
                        BEST RESULT
                    ================================================== */}

                    <div className="experiment-stats">

                      <div>
                        <span>SCENARIOS</span>

                        <strong>
                          {experimentResults.length}
                        </strong>
                      </div>

                      <div>
                        <span>BEST MEAN</span>

                        <strong>
                          {Number(
                            bestExperiment?.mean ||
                              bestExperiment?.reach_reduction_percent ||
                              0
                          ).toFixed(2)}
                          %
                        </strong>
                      </div>

                      <div>
                        <span>BEST STRATEGY</span>

                        <strong>
                          {String(
                            bestExperiment?.strategy ||
                              "-"
                          )
                            .replace(
                              /_/g,
                              " "
                            )
                            .toUpperCase()}
                        </strong>
                      </div>

                      <div>
                        <span>BEST TIMING</span>

                        <strong>
                          {String(
                            bestExperiment?.timing ||
                              "-"
                          ).toUpperCase()}
                        </strong>
                      </div>

                    </div>

                    {/* ==================================================
                        EARLY EFFECT
                    ================================================== */}

                    <div className="early-effect">

                      <div className="early-effect-header">

                        <span>
                          EARLY INTERVENTION EFFECT
                        </span>

                        <strong>
                          {earlyAverage.toFixed(2)}%
                        </strong>

                      </div>

                      <div className="early-effect-track">

                        <div
                          style={{
                            width: `${Math.min(
                              earlyAverage,
                              100
                            )}%`,
                          }}
                        ></div>

                      </div>

                    </div>

                    {/* ==================================================
                        TABLE
                    ================================================== */}

                    <div className="experiment-table-container">

                      <div className="table-heading">

                        <div>

                          <span className="panel-kicker">
                            INTERVENTION STRATEGY
                            PERFORMANCE
                          </span>

                        </div>

                        <span>
                          {
                            experimentResults.length
                          } SCENARIOS
                        </span>

                      </div>

                      <div className="experiment-table-wrapper">

                        <table className="experiment-table">

                          <thead>

                            <tr>

                              <th>
                                STRATEGY
                              </th>

                              <th>
                                TIMING
                              </th>

                              <th>
                                MEAN
                              </th>

                              <th>
                                MEDIAN
                              </th>

                              <th>
                                STD
                              </th>

                              <th>
                                MIN
                              </th>

                              <th>
                                MAX
                              </th>

                            </tr>

                          </thead>

                          <tbody>

                            {experimentResults.map(
                              (item, index) => (
                                <tr
                                  key={`${item.strategy}-${item.timing}-${index}`}
                                  className={
                                    item ===
                                    bestExperiment
                                      ? "best-row"
                                      : ""
                                  }
                                >

                                  <td>
                                    <strong>
                                      {String(
                                        item.strategy ||
                                          ""
                                      )
                                        .replace(
                                          /_/g,
                                          " "
                                        )
                                        .toUpperCase()}
                                    </strong>
                                  </td>

                                  <td>
                                    <span className="timing-badge">
                                      {String(
                                        item.timing ||
                                          ""
                                      ).toUpperCase()}
                                    </span>
                                  </td>

                                  <td>
                                    {Number(
                                      item.mean ||
                                        item.reach_reduction_percent ||
                                        0
                                    ).toFixed(2)}
                                    %
                                  </td>

                                  <td>
                                    {Number(
                                      item.median || 0
                                    ).toFixed(2)}
                                    %
                                  </td>

                                  <td>
                                    ±
                                    {Number(
                                      item.std || 0
                                    ).toFixed(2)}
                                  </td>

                                  <td>
                                    {Number(
                                      item.min || 0
                                    ).toFixed(2)}
                                    %
                                  </td>

                                  <td className="max-value">
                                    {Number(
                                      item.max || 0
                                    ).toFixed(2)}
                                    %
                                  </td>

                                </tr>
                              )
                            )}

                          </tbody>

                        </table>

                      </div>

                    </div>

                    {/* ==================================================
                        RESEARCH FINDING
                    ================================================== */}

                    <div className="research-finding">

                      <strong>
                        RESEARCH FINDING
                      </strong>

                      <span>
                        Early intervention produces
                        the strongest average cascade
                        reduction in the validated
                        experiment. Effectiveness
                        decreases as intervention is
                        delayed, supporting early
                        mitigation as the preferred
                        response window for
                        misinformation propagation.
                      </span>

                    </div>

                  </>
                )}

            </div>

          </div>

        </section>

        {/* ====================================================
            FOOTER
        ==================================================== */}

        <footer className="footer">

          <span>
            TRUTHNET AI · Fake News Propagation
            Intelligence
          </span>

          <span>
            GraphSAGE · UPFD GossipCop · FibVID · v1.0
          </span>

        </footer>

      </main>

    </div>
  );
}

export default App;