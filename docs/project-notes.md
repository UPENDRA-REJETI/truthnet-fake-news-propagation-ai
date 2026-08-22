# Multi-Agent AI Framework for Fake News Propagation Prediction and Intervention in Social Networks

## Project Development Notes

---

# Phase 1 — Project Initialization

## 1. Project Overview

The project aims to develop an AI-driven full-stack platform for analyzing and simulating the propagation of fake news through social networks.

Unlike conventional fake-news detection systems that only classify information as fake or real, this project focuses on understanding how misinformation propagates through a network, identifying influential users and communities, simulating intervention strategies, and generating explainable decision-support insights.

The system will combine graph-based machine learning, agent-based simulation, graph analytics, explainable AI, and an LLM-based explanation layer.

---

## 2. Problem Statement

The rapid spread of fake news on social media can influence public opinion, create panic, and cause social and political instability. Existing fake news detection systems primarily focus on identifying whether a piece of news is fake but do not model how misinformation spreads or determine the most effective intervention strategies.

The proposed system will simulate the propagation of fake news across a social network using graph-based modeling, predict propagation risk, identify influential users and communities, and evaluate intervention strategies such as fact-checking, content moderation, awareness campaigns, and targeted intervention.

---

## 3. Project Objectives

The major objectives are:

1. Detect and analyze fake news using AI-based techniques.
2. Represent misinformation propagation as a graph.
3. Apply a Graph Neural Network to learn propagation-related patterns.
4. Identify influential users within propagation networks.
5. Detect communities within the social network.
6. Simulate user behavior using a multi-agent model.
7. Evaluate multiple misinformation intervention strategies.
8. Compare intervention effectiveness quantitatively.
9. Provide explainable insights into propagation and intervention outcomes.
10. Use an LLM to convert analytical results into human-readable decision-support reports.
11. Develop a full-stack application through which users can interact with the AI system.

---

## 4. Proposed System

The proposed platform follows the pipeline:

Data
→ Graph Construction
→ Graph Neural Network
→ Propagation Risk Analysis
→ Influential User Identification
→ Community Analysis
→ Multi-Agent Simulation
→ Intervention Strategies
→ Outcome Comparison
→ Explainability
→ LLM Decision Support

---

## 5. Development Strategy

A local-first hybrid development strategy was selected.

### Local Development

The main application will be developed using VS Code on the local machine.

Local development will include:

- React frontend
- Node.js and Express backend
- Python FastAPI AI service
- Graph processing
- Agent-based simulation
- Database development
- API integration
- Testing

### Google Colab

Google Colab will primarily be used for computationally intensive AI experiments and GNN training where GPU acceleration is beneficial.

The trained models will subsequently be integrated into the local AI service for inference.

### GitHub

GitHub will be used as the central source-control and project repository.

The development workflow will be:

Local Development
→ Git Commit
→ GitHub
→ Final Cloud Deployment

### Cloud Deployment

Cloud deployment will be performed near the completion of the project for final demonstration and showcase.

A local version will also be maintained as a backup for demonstrations.

---

## 6. Technology Stack

### Frontend

- React
- Vite
- Tailwind CSS
- Recharts
- Cytoscape.js

### Backend

- Node.js
- Express.js
- REST APIs

### AI Service

- Python 3.11
- FastAPI
- PyTorch
- PyTorch Geometric
- scikit-learn
- NetworkX

### Graph and AI

- Graph Neural Networks
- GraphSAGE
- Community Detection
- Centrality Analysis
- Agent-Based Modeling
- Explainable AI

### Database

- MongoDB

### LLM

- API-based Large Language Model

### Development Tools

- Visual Studio Code
- Git
- GitHub
- Google Colab

---

## 7. AI Scope Reduction

To keep the project achievable within the available development time, the number of independent AI models was intentionally reduced.

The project will focus on:

### Primary GNN

GraphSAGE

### Fake News Analysis

The graph-based UPFD dataset already contains fake/real labels, so a separate complex fake-news deep-learning model will not initially be developed.

### Agent-Based Modeling

Rule-based behavioral agents will simulate different types of users.

### Intervention Engine

Interventions will initially be implemented using rule-based and graph-derived strategies rather than additional machine-learning models.

### LLM

The LLM will act as an explanation and decision-support layer rather than performing the primary numerical prediction.

---

## 8. Planned Agent Types

The initial simulation will contain several behavioral agent types:

- Normal User
- Influencer
- Skeptic
- Highly Susceptible User
- Bot-like Amplifier

Each agent will have parameters such as:

- Influence
- Activity level
- Susceptibility
- Trust level
- Probability of sharing
- Probability of verification

These parameters will influence propagation behavior during simulation.

---

## 9. Planned Intervention Strategies

The initial intervention framework will include:

### 1. No Intervention

Baseline scenario used for comparison.

### 2. Fact-Checking

Reduce the probability of further sharing after misinformation is identified.

### 3. Content Moderation

Reduce visibility and propagation probability of suspicious content.

### 4. Influencer-Targeted Intervention

Target high-impact users identified through graph analysis.

### 5. Adaptive Intervention

If time permits, dynamically select intervention targets based on propagation risk, user influence, and community vulnerability.

The adaptive strategy is a potential research contribution of the project.

---

## 10. Proposed System Architecture

The planned architecture consists of four major layers:

### Frontend Layer

React-based web interface responsible for:

- Dashboard
- Fake-news analysis
- Network visualization
- Propagation visualization
- Simulation controls
- Intervention comparison
- Explainability
- AI-generated reports

### Backend Layer

Node.js + Express responsible for:

- REST API
- Application logic
- Database communication
- Request orchestration
- Communication with the AI service

### AI Layer

Python + FastAPI responsible for:

- Graph processing
- GraphSAGE inference
- Influence analysis
- Community detection
- Agent simulation
- Intervention evaluation
- Explainability

### Data Layer

MongoDB for application and experiment-related data.

The graph itself will primarily be processed using NetworkX/PyTorch Geometric rather than introducing a dedicated graph database initially.

---

## 11. Repository Structure

The initial repository structure is:

fake-news-propagation-ai/

├── frontend/
├── backend/
├── ai-service/
│ └── .venv/
├── data/
│ ├── raw/
│ └── processed/
├── models/
├── notebooks/
├── experiments/
├── docs/
│ └── project-notes.md
├── .gitignore
└── README.md

### Folder Responsibilities

#### frontend/

React application.

#### backend/

Node.js and Express application.

#### ai-service/

Python AI and simulation service.

#### data/raw/

Original datasets. Raw datasets will not be modified.

#### data/processed/

Cleaned and transformed datasets.

#### models/

Trained machine-learning and GNN model files.

#### notebooks/

AI experiments and dataset analysis.

#### experiments/

Experiment configurations and results.

#### docs/

Project documentation and development notes.

---

## 12. Environment Setup

The following development environment was selected:

### Operating Environment

Windows

### Code Editor

Visual Studio Code

### Python

Python 3.11

### Node.js

Node.js 24 LTS

### Package Manager

npm

### Version Control

Git

### AI Environment

A project-specific Python virtual environment is maintained at:

ai-service/.venv/

This environment isolates AI dependencies from globally installed Python packages.

---

## 13. Environment Verification

The following components were verified before beginning implementation:

- Git
- Node.js
- npm
- Python 3.11
- pip
- VS Code
- Python virtual environment

Node.js was upgraded from an older version to Node.js 24 LTS to provide a current and stable JavaScript development environment.

The Python project environment uses Python 3.11.

---

## 14. Development Principle

The project will follow:

Understand
→ Implement
→ Test
→ Record
→ Integrate

Each major module will document:

- Objective
- Input
- Processing
- Output
- Evaluation
- Observations
- Limitations
- Next step

This documentation will later be used to prepare the final project report, presentation, and viva material.

---

## 15. Initial Development Roadmap

### Phase 0 — Environment Setup

Completed.

### Phase 1 — Project Initialization

Completed.

### Phase 2 — Dataset Research and Audit

In progress.

### Phase 3 — Data Preprocessing

Pending.

### Phase 4 — GraphSAGE Model

Pending.

### Phase 5 — Influence and Community Analysis

Pending.

### Phase 6 — Multi-Agent Simulation

Pending.

### Phase 7 — Intervention Engine

Pending.

### Phase 8 — Python AI API

Pending.

### Phase 9 — Node.js Backend

Pending.

### Phase 10 — React Frontend

Pending.

### Phase 11 — Full Integration

Pending.

### Phase 12 — Testing and Evaluation

Pending.

### Phase 13 — Deployment

Pending.

### Phase 14 — Final Report and Presentation

Pending.

---

## 16. Current Status at End of Phase 1

The project repository and development environment have been initialized.

The local-first hybrid architecture has been selected.

The AI scope has been reduced to a manageable set of core technologies.

The next major task is dataset auditing and selection before beginning model development.

---

# Phase 2 — Dataset Audit

## Dataset Selection

After auditing UPFD PolitiFact and UPFD GossipCop, UPFD GossipCop was selected as the primary dataset and UPFD PolitiFact as a secondary/generalization dataset.

### UPFD GossipCop

- Total graphs: 5,464
- Training graphs: 1,092
- Validation graphs: 546
- Test graphs: 3,826
- Node features: 10-dimensional profile features
- Training node count:
  - Minimum: 3
  - Maximum: 195
  - Mean: 58.13
  - Median: 50
- Test node count:
  - Minimum: 3
  - Maximum: 199
  - Mean: 57.50
  - Median: 49
- Training labels:
  - Label 0: 557
  - Label 1: 535
- Validation labels:
  - Label 0: 265
  - Label 1: 281
- Test labels:
  - Label 0: 1,910
  - Label 1: 1,916

### UPFD PolitiFact

- Total graphs: 314
- Training graphs: 62
- Validation graphs: 31
- Test graphs: 221
- Node features: 10-dimensional profile features
- Training node count:
  - Minimum: 4
  - Maximum: 492
  - Mean: 97.94
  - Median: 51
- Test node count:
  - Minimum: 3
  - Maximum: 497
  - Mean: 141.19
  - Median: 92
- Training labels:
  - Label 0: 36
  - Label 1: 26
- Validation labels:
  - Label 0: 13
  - Label 1: 18
- Test labels:
  - Label 0: 108
  - Label 1: 113

## Dataset Decision

UPFD GossipCop will be used as the primary dataset because it provides substantially more training graphs and a highly balanced class distribution while maintaining relatively small propagation graphs.

UPFD PolitiFact will be retained as a secondary dataset for generalization analysis.

The two datasets will initially be evaluated separately rather than blindly merged.

## Current Feature Representation

The initial audit uses the 10-dimensional `profile` node representation.

Alternative UPFD representations such as `content` may be evaluated later if time and computational resources permit.

## Next Investigation

Inspect UPFD timestamp and node-ID mappings to determine whether early propagation information can be used for propagation/cascade prediction.

# Phase 2B — FibVID Initial Schema Audit

FibVID was downloaded from the official dataset archive and extracted without modifying the original ZIP file.

## Files Identified

### news_claim.csv

- File size: approximately 0.17 MB
- Rows: 1,353
- Columns:
  - text
  - source
  - claim_num
  - group

### origin_tweet.csv

- File size: approximately 0.50 MB
- Rows: 4,290
- Contains original tweet information associated with claims.
- Includes timestamp, tweet ID, engagement metrics, claim number, group, hashtag and user ID.

### claim_propagation.csv

- File size: approximately 46.47 MB
- Rows: 323,592
- Columns:
  - tweet_user
  - tweet_id
  - like_count
  - depth
  - parent_user
  - create_date
  - parent_id
  - retweet_count
  - post_text
  - claim_number
  - group
  - hashtag

This file provides the core propagation structure required for temporal graph construction.

### user_information.csv

- File size: approximately 18.34 MB
- Rows: 189,421
- Columns:
  - following_count
  - follower_count
  - creation_date
  - description
  - user_id

This file provides user-level attributes that can be used as graph node features.

## Initial Assessment

FibVID provides substantially richer temporal and propagation information than the preprocessed UPFD graphs. The presence of parent tweet IDs, parent users, timestamps, propagation depth, claim identifiers and user metadata makes FibVID suitable for constructing temporal misinformation propagation graphs.

The dataset therefore passes the initial feasibility audit.

The exact semantic mapping of the `group` field to claim veracity categories will be verified before model development.

## The next step is a quantitative propagation and graph-integrity audit.

# Phase 2C — FibVID Propagation and Graph Integrity Audit

The FibVID propagation data was quantitatively audited to verify whether it
could support temporal propagation modeling, graph construction, influence
analysis, and intervention simulation.

## Claim Audit

Total claims:

- 1,353

Claims by group:

- Group 0: 203
- Group 1: 569
- Group 2: 150
- Group 3: 431

Source distribution:

- PolitiFact: 1,033
- Snopes: 320

All claim fields were complete with no missing values.

## Propagation Audit

Total propagation records:

- 221,253

Unique claims with propagation:

- 295

Unique tweets:

- 221,253

Unique users:

- 144,741

Propagation depth:

- Minimum: 0
- Maximum: 6
- Mean: 1.84
- Median: 2.00

## Cascade Size

Number of cascades:

- 295

Cascade size:

- Minimum: 1
- Median: 216
- Mean: 750.01
- Maximum: 15,096

Percentiles:

- 25th: 16.50
- 50th: 216.00
- 75th: 820.50
- 90th: 2,123.20
- 95th: 2,898.60
- 99th: 6,348.12

## Temporal Integrity

Invalid timestamps:

- 0

Temporal range:

- 2020-02-01 to 2020-12-31

Total time span:

- Approximately 334 days

## Graph Integrity

Root events:

- 1,774

Self-parenting records:

- 0

Non-root parent IDs missing from propagation data:

- 0

Duplicate tweet IDs:

- 0

These results indicate that the propagation data provides a structurally
consistent basis for constructing parent-child propagation graphs.

## User Coverage

Propagation users:

- 144,741

Users with profile information:

- 144,741

User feature coverage:

- 100%

Missing profile fields were limited to:

- creation_date: 2,445
- description: 19,564

## Claim Coverage

Total claims:

- 1,353

Claims with propagation records:

- 295

Propagation coverage:

- 21.80%

## Decision

FibVID was confirmed as suitable for propagation-oriented experiments.

Because only 295 of the 1,353 claims contain propagation records, the
propagation experiments are explicitly limited to those 295 cascades.

---

# Phase 2D — Propagation Feature Engineering

A claim-level propagation feature table was generated from the FibVID
propagation records.

## Generated Features

The feature table contains 30 columns covering:

- cascade start time
- cascade end time
- cascade duration
- final cascade size
- unique users
- maximum propagation depth
- total likes
- total retweets
- 1-hour activity
- 6-hour activity
- 24-hour activity
- growth ratios
- engagement measures
- logarithmic cascade size

Early propagation windows were explicitly calculated for:

- 1 hour
- 6 hours
- 24 hours

## Early Propagation Coverage

All 295 propagation cascades had activity within each evaluated window.

### 1-Hour Window

- Mean tweets: 15.39
- Median tweets: 2.00
- Mean users: 15.18

### 6-Hour Window

- Mean tweets: 36.64
- Median tweets: 7.00
- Mean users: 35.84

### 24-Hour Window

- Mean tweets: 80.76
- Median tweets: 17.00
- Mean users: 78.69

## Target

The prediction target was:

`log_final_cascade_size`

Target statistics:

- Mean: 4.7765
- Median: 5.3799
- Minimum: 0.6931
- Maximum: 9.6223

## Output

Feature table:

`data/processed/fibvid_propagation_features.csv`

Model-ready dataset:

`data/processed/fibvid_propagation_model_data.csv`

---

# Experiment 02 — Propagation Prediction Baselines

## Objective

Determine whether early propagation behavior can be used to predict the
eventual size of an information cascade.

Three observation windows were evaluated:

- 1 hour
- 6 hours
- 24 hours

Three prediction approaches were compared:

- Median baseline
- Ridge Regression
- Random Forest

## Dataset

Total cascades:

- 295

Training cascades:

- 236

Testing cascades:

- 59

## 1-Hour Results

### Median Baseline

- Log MAE: 2.0632
- Log RMSE: 2.5619
- Log R²: -0.0431

### Ridge Regression

- Log MAE: 1.7067
- Log RMSE: 1.9616
- Log R²: 0.3884

### Random Forest

- Log MAE: 1.4379
- Log RMSE: 1.7278
- Log R²: 0.5255

## 6-Hour Results

### Median Baseline

- Log MAE: 2.0632
- Log RMSE: 2.5619
- Log R²: -0.0431

### Ridge Regression

- Log MAE: 1.6191
- Log RMSE: 1.8969
- Log R²: 0.4281

### Random Forest

- Log MAE: 1.3286
- Log RMSE: 1.5471
- Log R²: 0.6196

## 24-Hour Results

### Median Baseline

- Log MAE: 2.0632
- Log RMSE: 2.5619
- Log R²: -0.0431

### Ridge Regression

- Log MAE: 1.6720
- Log RMSE: 1.9722
- Log R²: 0.3818

The original-scale Ridge metrics were unstable because of the highly
skewed cascade-size distribution.

### Random Forest

- Log MAE: 1.3262
- Log RMSE: 1.5631
- Log R²: 0.6117

## Decision

Random Forest consistently outperformed the median baseline and Ridge
Regression on the logarithmic prediction target.

The 6-hour model produced the strongest single held-out test R²:

- R² = 0.6196

The 24-hour model was also highly competitive.

---

# Experiment 02.1 — Random Forest Robustness Validation

Five-fold cross-validation was performed to determine whether the
propagation prediction results were robust to different train/test splits.

## 1-Hour Random Forest

R² scores:

- 0.5312
- 0.1348
- 0.2326
- 0.0823
- 0.5186

Mean R²:

- 0.2999

Standard deviation:

- 0.1900

Mean MAE:

- 1.5796

Mean RMSE:

- 1.9822

## 6-Hour Random Forest

R² scores:

- 0.6296
- 0.2207
- 0.1944
- 0.2304
- 0.5776

Mean R²:

- 0.3705

Standard deviation:

- 0.1914

Mean MAE:

- 1.4860

Mean RMSE:

- 1.8726

## 24-Hour Random Forest

R² scores:

- 0.6102
- 0.2112
- 0.2149
- 0.2858
- 0.6234

Mean R²:

- 0.3891

Standard deviation:

- 0.1878

Mean MAE:

- 1.4378

Mean RMSE:

- 1.8440

## Robustness Decision

The 24-hour model achieved the strongest cross-validation mean R²:

- 0.3891

The 6-hour model remained highly competitive and achieved the strongest
single test-set R².

All three models were retained because the different observation windows
represent different operational intervention horizons.

---

# Experiment 02.2 — Final Propagation Models

Final Random Forest models were trained for all three observation windows.

## Models

### 1-Hour

`models/propagation/propagation_rf_1h.joblib`

Important features:

- early_max_depth_1h
- early_retweets_1h
- early_likes_1h
- engagement_1h
- early_users_1h
- early_tweets_1h

### 6-Hour

`models/propagation/propagation_rf_6h.joblib`

Important features:

- early_max_depth_6h
- early_retweets_6h
- early_likes_6h
- engagement_6h
- early_users_6h
- early_tweets_6h

### 24-Hour

`models/propagation/propagation_rf_24h.joblib`

Important features:

- early_max_depth_24h
- early_retweets_24h
- early_likes_24h
- early_tweets_24h
- engagement_24h
- early_users_24h

Metadata:

`models/propagation/metadata.json`

---

# Phase 3 — Influence Analysis

## Objective

Identify users that can substantially affect information propagation through
network structure and propagation behavior.

Influence analysis was performed using:

- degree centrality
- weighted out-degree
- PageRank
- betweenness centrality
- propagation activity
- follower count

The analysis was progressively refined to produce more meaningful
intervention candidates.

---

# Experiment 03A — Initial Influence Analysis

The largest propagation cascade was selected for initial influence analysis.

Cascade:

- Claim: 281
- Records: 15,096

Graph:

- Nodes: 13,493
- Edges: 14,540
- Density: 0.000080
- Connected components: 2

The top influential users were ranked using a combined influence score.

Output:

`data/processed/fibvid_influential_users.csv`

---

# Experiment 03B — Weighted Influence Analysis

The influence analysis was extended using weighted propagation relationships
and user-profile information.

Additional information included:

- follower count
- propagation posts
- weighted out-degree

A major high-reach amplifier was identified:

- User 51591
- Weighted out-degree: 2,600
- Propagation posts: 15
- Follower count: 88,576,360

Output:

`data/processed/fibvid_weighted_influence_users.csv`

---

# Experiment 03C — Refined Influence Analysis

The influence scoring system was refined to classify users into network
roles.

Roles included:

- Participant
- Amplifier
- High-Influence Node
- High-Reach Node

The analysis showed that high structural influence is not determined by a
single metric. A combination of propagation activity and graph influence
provides a more useful ranking for intervention purposes.

Output:

`data/processed/fibvid_refined_influence_users.csv`

---

# Experiment 03D — All-Cascade Influence Analysis

Influence analysis was extended from a single cascade to all available
propagation cascades.

## Dataset

- Propagation records: 221,253
- Propagation cascades: 295
- Valid influence rows: 207,791
- Unique cascades represented: 236
- Unique users: 144,635

## Network Role Distribution

- Participant: 205,484
- High-Reach Node: 1,824
- Amplifier: 315
- Bridge: 168

## Intervention Priority

- Low: 186,907
- Moderate: 10,392
- High: 8,288
- Critical: 2,204

Output:

`data/processed/fibvid_all_cascade_influence.csv`

This analysis provides the foundation for dynamic intervention targeting.

---

# Phase 4 — Intervention Simulation

## Objective

Evaluate how different intervention strategies affect the future
propagation of misinformation.

The following strategies were implemented:

1. No intervention
2. Fact-checking
3. Content moderation
4. Targeted intervention

Interventions were evaluated at:

- Early
- Mid
- Late

The initial simulations were performed on Claim 281.

---

# Experiment 04A — Initial Intervention Simulation

The initial intervention simulator compared the effect of removing or
disrupting propagation through selected influential users.

For Claim 281:

### No intervention

- Final reach: 13,493
- Remaining edges: 14,540
- Maximum depth: 8

### Fact-checking

- Final reach: 10,288
- Reach reduction: 23.75%

### Content moderation

- Final reach: 8,338
- Reach reduction: 38.20%

### Targeted intervention

- Final reach: 5,088
- Reach reduction: 62.29%

This initial result demonstrated the potential effectiveness of targeted
intervention, but the simulation was subsequently refined to account for
temporal activity and causal propagation structure.

---

# Experiment 04B — Temporal Intervention Simulation

Intervention timing was introduced to compare early, mid and late
intervention.

The results demonstrated that intervention effectiveness decreases as the
cascade progresses.

However, the first implementation revealed that some selected intervention
targets were not active at the corresponding intervention time.

This motivated a dynamic-target selection approach.

---

# Experiment 04C — Dynamic Intervention Simulation

Instead of using a fixed list of influential users, intervention targets
were selected from users who were actually active at the intervention
checkpoint.

For Claim 281:

### Early checkpoint

Active candidates:

- 2,795

### Mid checkpoint

Active candidates:

- 6,788

### Late checkpoint

Active candidates:

- 10,875

The dynamic system selected the highest-scoring active users rather than
blindly targeting users identified from the complete cascade.

This improved the methodological validity of the intervention framework.

---

# Experiment 04D — Causal Dynamic Intervention V1

A causal simulation was developed to determine whether intervention effects
could be measured through actual propagation links.

The simulator evaluated:

- final reach
- remaining propagation edges
- maximum propagation depth
- percentage reduction in reach

Three intervention strategies were evaluated at three temporal checkpoints.

Initial causal results passed the implemented sanity checks but motivated
a further graph-linkage validation.

---

# Experiment 04E — Causal Dynamic Intervention V2

## Propagation Linkage Validation

The propagation graph was explicitly validated before calculating
intervention effects.

Validation results:

- Duplicate tweet IDs: 0
- Non-root parent IDs not found: 0
- Self-parenting events: 0
- Root events: 26

The correct propagation relationship was established as:

`parent_id → tweet_id`

The user-level fields were treated separately for intervention targeting:

`parent_user → tweet_user`

This correction ensures that intervention simulation follows actual
tweet-level propagation paths.

## Corrected Baseline

For Claim 281:

- Final reach: 12,310
- Remaining edges: 13,702
- Maximum depth: 5

## Causal Results

### Fact-checking

- Early reach reduction: 53.27%
- Mid reach reduction: 26.49%
- Late reach reduction: 6.16%

### Content moderation

- Early reach reduction: 51.49%
- Mid reach reduction: 33.02%
- Late reach reduction: 15.15%

### Targeted intervention

- Early reach reduction: 16.08%
- Mid reach reduction: 12.83%
- Late reach reduction: 6.40%

Edge reduction showed the same general temporal pattern.

## Sanity Checks

All intervention scenarios passed the implemented sanity checks.

Temporal effect check:

`early >= mid >= late`

passed for:

- Fact-checking
- Content moderation
- Targeted intervention

This confirmed that intervention timing is a meaningful variable in the
causal simulation.

Output:

`experiments/experiment_03_causal_intervention_v2.csv`

---

# Experiment 04F — Multi-Cascade Causal Validation

## Objective

Determine whether the causal intervention behavior observed on Claim 281
generalizes across multiple propagation cascades.

The 15 largest available propagation cascades were selected.

## Cascades

1. Claim 281 — 15,096 records
2. Claim 170 — 13,560 records
3. Claim 298 — 12,601 records
4. Claim 114 — 5,949 records
5. Claim 273 — 5,529 records
6. Claim 124 — 4,984 records
7. Claim 229 — 4,252 records
8. Claim 231 — 4,090 records
9. Claim 46 — 3,828 records
10. Claim 208 — 3,191 records
11. Claim 141 — 3,150 records
12. Claim 173 — 3,034 records
13. Claim 152 — 3,009 records
14. Claim 122 — 3,008 records
15. Claim 131 — 2,914 records

Total valid cascades processed:

- 15

Total experiment configurations:

- 135

Each cascade was evaluated under:

- Fact-checking
- Content moderation
- Targeted intervention

at:

- Early
- Mid
- Late

---

# Multi-Cascade Reach Reduction Results

## Fact-Checking

- Early mean: 28.60%
- Mid mean: 17.20%
- Late mean: 5.99%

Median:

- Early: 27.39%
- Mid: 15.08%
- Late: 5.73%

Standard deviation:

- Early: 9.12%
- Mid: 7.07%
- Late: 1.29%

## Content Moderation

- Early mean: 43.06%
- Mid mean: 25.63%
- Late mean: 10.47%

Median:

- Early: 44.19%
- Mid: 25.62%
- Late: 10.03%

Standard deviation:

- Early: 10.20%
- Mid: 6.05%
- Late: 2.88%

## Targeted Intervention

- Early mean: 17.88%
- Mid mean: 15.69%
- Late mean: 6.48%

Median:

- Early: 17.52%
- Mid: 13.33%
- Late: 6.45%

Standard deviation:

- Early: 14.95%
- Mid: 12.13%
- Late: 4.85%

---

# Multi-Cascade Edge Reduction Results

## Fact-Checking

- Early mean: 29.63%
- Mid mean: 17.79%
- Late mean: 6.22%

## Content Moderation

- Early mean: 44.31%
- Mid mean: 26.40%
- Late mean: 10.74%

## Targeted Intervention

- Early mean: 18.47%
- Mid mean: 16.10%
- Late mean: 6.67%

The edge-level results support the reach-level findings.

---

# Experiment 04G — Temporal Robustness Validation

The expected temporal relationship was:

`Early intervention >= Mid intervention >= Late intervention`

The condition passed for all three intervention strategies.

### Fact-checking

`28.60% >= 17.20% >= 5.99%`

PASS

### Content moderation

`43.06% >= 25.63% >= 10.47%`

PASS

### Targeted intervention

`17.88% >= 15.69% >= 6.48%`

PASS

This demonstrates that the temporal intervention effect observed in the
single-cascade experiment also appears across the multi-cascade validation
set.

---

# Experiment 04 — Final Interpretation

The intervention experiments demonstrate that propagation containment is
strongly dependent on intervention timing.

Across the 15 evaluated cascades:

- Content moderation produced the highest average reach reduction.
- Fact-checking produced the second-highest average reduction.
- Targeted intervention produced lower average reduction but showed
  considerably greater variability across cascades.
- Early intervention consistently outperformed mid and late intervention.

The variability of targeted intervention indicates that selective
intervention effectiveness depends strongly on the structure and influence
distribution of individual cascades.

The results should be interpreted as simulation-based evidence under the
defined intervention assumptions rather than direct measurements of
real-world platform moderation outcomes.

---

# Experiment 04 — Output Artifacts

Influence analysis:

`data/processed/fibvid_influential_users.csv`

Weighted influence:

`data/processed/fibvid_weighted_influence_users.csv`

Refined influence:

`data/processed/fibvid_refined_influence_users.csv`

All-cascade influence:

`data/processed/fibvid_all_cascade_influence.csv`

Multi-cascade causal results:

`experiments/experiment_03_multicascade_validation.csv`

Multi-cascade summary:

`experiments/experiment_03_multicascade_summary.csv`

---

# Current Project Status

## Completed

- Project initialization
- Development environment setup
- UPFD dataset investigation
- UPFD GossipCop selection
- FibVID schema audit
- FibVID propagation integrity audit
- Propagation feature engineering
- GraphSAGE fake-news detection
- GraphSAGE evaluation
- Propagation prediction baselines
- Random Forest robustness validation
- Final propagation models
- User influence analysis
- Weighted influence analysis
- Refined influence analysis
- All-cascade influence analysis
- Intervention simulation
- Temporal intervention simulation
- Dynamic intervention simulation
- Causal intervention simulation
- Causal graph-linkage correction
- Multi-cascade causal validation
- Temporal robustness validation

## Current Research Status

The core experimental and analytical phase is complete.

The next phase is system integration and product implementation.

---

# Phase 5 — Full-Stack System Integration

The validated research components will now be integrated into the actual
application.

## AI Service

Python + FastAPI will expose:

1. Fake-news classification
2. Propagation prediction
3. Influence analysis
4. Intervention simulation
5. Scenario comparison

## Node.js Backend

Node.js + Express will act as the application backend and orchestration
layer.

Responsibilities include:

- REST API
- request validation
- communication with FastAPI
- application data management
- experiment result retrieval
- frontend-facing APIs

## Frontend

The frontend will be designed as a major component of the project rather
than a simple form-based interface.

The interface will include:

- professional dashboard
- fake-news prediction screen
- confidence visualization
- propagation forecast
- interactive propagation graph
- influential-user ranking
- network-role visualization
- intervention simulator
- early/mid/late comparison
- reach reduction charts
- explainable result panels
- AI-generated analytical summary

The frontend should visually communicate the underlying AI and graph
analytics rather than merely expose API responses.

---

# Phase 5 — Integration Principle

The implementation will proceed incrementally:

`AI API`
→ `Node.js API`
→ `Frontend`
→ `End-to-End Integration`
→ `Testing`
→ `UI Refinement`
→ `Final Demonstration`

Each integration stage will be tested before proceeding to the next stage.

If an existing model or analytical component shows a significant limitation
during integration, model improvement will be performed before continuing.

---

# Phase 5 — Immediate Next Step

The first integration task is to expose the already-trained models through
the existing FastAPI service.

No new model should be trained at this point unless integration testing
reveals a specific model limitation.

The existing trained artifacts will be reused:

`models/graphsage_gossipcop.pt`

`models/propagation/propagation_rf_1h.joblib`

`models/propagation/propagation_rf_6h.joblib`

`models/propagation/propagation_rf_24h.joblib`

The objective is to convert the completed research work into a working
end-to-end application.
