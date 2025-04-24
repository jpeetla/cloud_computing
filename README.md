# CS 4740 @ UVA - Cloud Computing Final Project 
***AI-Driven Multi-Cloud Serverless Scheduler***

This repository explores **reinforcement-learning–based scheduling** of serverless functions across AWS Lambda, Google Cloud Functions, and Azure Functions. The scheduler learns which cloud to invoke (and when) to minimise end-to-end latency, energy consumption, and cost.

---

## Features
- **Q-learning agent** that adapts to workload patterns.  
- Baseline **Round-Robin** and **Least-Connection** schedulers for comparison.  
- Unified wrapper for invoking each provider’s serverless API.  
- CSV + console metrics for latency, energy-use (mocked), and dollars spent.

---

## Quick Start

### 1 — Install

```bash
git clone https://github.com/jpeetla/cloud_computing.git
cd cloud_computing
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2 — Export Cloud Credentials

Set the following environment variables before running the training script:

| Provider | Required Variables |
|----------|--------------------|
| **AWS**  | `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_REGION`, `AWS_LAMBDA_NAME` |
| **GCP**  | `GCP_SA_JSON_PATH`, `GCP_PROJECT`, `GCP_FUNC_NAME` |
| **Azure**| `AZURE_SUB_ID`, `AZURE_RG`, `AZURE_FUNC_APP` |

> 💡 Tip: create a `.env` file with all variables and run `source .env` to quickly load them.

---

### 3 — Train & Evaluate

To train the reinforcement learning agent and evaluate its performance:

```bash
python train.py --episodes 200 --max_steps 100 --decay 0.99
```
This script performs three actions:
- Trains the agent using Q-learning and saves the learned Q-table (qtable.pkl).
- Benchmarks the Q-learning scheduler against:
  - Round-Robin scheduler
  - Least-Connection scheduler
- Outputs performance metrics (latency, cost, energy) to evaluation_summary.csv.

---

#### Configuration
Modify hyperparameters in config.py as needed:

| Parameter | Value  | Description         |
|-----------|--------|---------------------|
| `ALPHA`   | 0.05   | Learning rate       |
| `GAMMA`   | 0.95   | Discount factor     |
| `EPSILON` | 0.20   | Exploration rate    |

---

Final cloud computing project by Jayanth Peetla and Varun Pavuloori
