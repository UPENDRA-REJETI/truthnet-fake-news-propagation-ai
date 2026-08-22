# TRUTHNET - Fake News Propagation Intelligence

An AI-powered full-stack platform for detecting misinformation, predicting its propagation, identifying influential network users, and evaluating causal intervention strategies.

---

## 1. Project Overview

TRUTHNET is an AI-driven misinformation intelligence platform designed to analyze how fake news propagates through social networks.

The system combines graph-based deep learning, propagation forecasting, network analysis, and causal intervention analysis into a single interactive dashboard.

The platform provides five major intelligence modules:

1. Fake News Detection
2. Propagation Forecasting
3. Influence Network Analysis
4. Intervention Simulation
5. Research Experiment Validation

---

## 2. Key Objectives

The project aims to:

- Detect misinformation using graph-based deep learning.
- Predict future misinformation cascade growth from early propagation signals.
- Identify high-impact users within propagation networks.
- Detect critical nodes and propagation bridges.
- Compare different misinformation intervention strategies.
- Analyze the effect of intervention timing.
- Validate intervention effectiveness across multiple cascades.
- Provide an interactive AI intelligence dashboard.

---

## 3. System Architecture

```text
                         +----------------------+
                         |      TRUTHNET UI     |
                         |    React + Vite      |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |    Node.js Backend   |
                         |      REST API        |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         |      AI Service      |
                         |       FastAPI        |
                         +----------+-----------+
                                    |
              +---------------------+---------------------+
              |                     |                     |
              v                     v                     v
       +-------------+       +-------------+       +-------------+
       |  GraphSAGE  |       |Random Forest|       |   Network   |
       | Fake News   |       | Propagation |       |   Analysis  |
       | Detection   |       | Forecasting |       |  Influence  |
       +------+------+       +------+------+       +------+------+
              |                     |                     |
              v                     v                     v
        UPFD-GossipCop           FibVID          Influence Metrics
              |                     |                     |
              +---------------------+---------------------+
                                    |
                                    v
                         +----------------------+
                         | Intervention Analysis|
                         |   Causal Simulation  |
                         +----------+-----------+
                                    |
                                    v
                         +----------------------+
                         | Research Validation |
                         | Multi-Cascade Study |
                         +----------------------+