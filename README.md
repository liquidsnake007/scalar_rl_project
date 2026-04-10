---
title: Distributed Failure Analyzer
emoji: 🚀
colorFrom: blue
colorTo: purple
sdk: docker

pinned: false
---
# 🔍 Distributed Systems Failure Analyzer

An **OpenEnv RL environment** where an AI agent analyzes logs from distributed
microservices and identifies failures, root causes, and degradation patterns.

Built for the **Meta × PyTorch × Scaler OpenEnv Hackathon 2026**.

---

## 🎯 Problem Statement

Real-world distributed systems fail in complex ways. This environment simulates
realistic service logs at three difficulty levels — from an obvious single-service
crash to a subtle intermittent failure buried across six noisy services.

The agent must act like a real **Site Reliability Engineer (SRE)** and diagnose
what went wrong.

---

## 📋 Three Tasks

### ✅ Task 1 — Easy
| Property | Value |
|---|---|
| Services | 1 service |
| Error rate | 99% (obvious spike) |
| Log quality | Full clean logs |
| Agent output | `{"service_name": "...", "error_code": "..."}` |
| Grader | service_name=0.5 + error_code=0.5 = **1.0** |

### ✅ Task 2 — Medium
| Property | Value |
|---|---|
| Services | 3 services |
| Error rate | ~40% (subtle) |
| Log quality | Partial — some entries missing |
| Agent output | `{"root_service": "...", "affected_service": "..."}` |
| Grader | root_service=0.6 + affected_service=0.4 = **1.0** |

### ✅ Task 3 — Hard
| Property | Value |
|---|---|
| Services | 6 services |
| Error rate | ~8% (intermittent) |
| Log quality | Noisy — contains red herrings |
| Agent output | `{"root_service":"...","endpoint":"...","failure_pattern":"...","severity":"..."}` |
| Grader | root=0.4 + endpoint=0.2 + pattern=0.2 + severity=0.2 = **1.0** |

---

## 🏆 Reward Function

| Result | Score |
|---|---|
| All fields correct | **1.0** |
| Most fields correct | **0.6 – 0.8** |
| Some fields correct | **0.2 – 0.5** |
| All fields wrong | **0.0** |

Partial credit is awarded per field — a partially right answer always gets a
meaningful non-zero score, which gives RL agents a useful learning signal.

---

## 🚀 Quick Start

### 1. Install
```bash
pip install -r requirements.txt
```

### 2. Run server locally
```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### 3. Test the API
```bash
# Reset to easy task
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'

# Submit an answer
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"service_name": "payment-api", "error_code": "HTTP_500"}'

# Check current state
curl http://localhost:7860/state
```

### 4. Run the inference script
```bash
export HF_TOKEN=hf_your_token_here
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export API_BASE_URL=https://api-inference.huggingface.co/v1

python inference.py
```

---

## 🐳 Docker

```bash
# Build the image
docker build -t failure-analyzer .

# Run it
docker run -p 7860:7860 failure-analyzer

# Test it
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task": "easy"}'
```

---

## 📁 Project Structure

```
failure_analyzer/
├── models.py              ← Pydantic Action + Observation models
├── __init__.py
├── environment.py     ← Game logic, log generators, reward graders
│app.py             ← FastAPI server (/reset /step /state)
├── inference.py           ← LLM agent with exact required log format
├── requirements.txt       ← Python dependencies
├── Dockerfile             ← Container for HF Spaces (port 7860)
├── openenv.yaml           ← OpenEnv spec config
└── README.md              ← This file
```

---

## 🔌 API Reference

| Endpoint | Method | Body | Description |
|---|---|---|---|
| `/` | GET | — | Health check |
| `/reset` | POST | `{"task": "easy"}` | Start new episode |
| `/step` | POST | task-specific JSON | Submit answer |
| `/state` | GET | — | Current env state |

---

## 📐 Action Spaces

### Easy
```json
{"service_name": "payment-api", "error_code": "HTTP_500"}
```

### Medium
```json
{"root_service": "auth-service", "affected_service": "checkout-api"}
```

### Hard
```json
{
  "root_service": "user-db",
  "endpoint": "/api/v1/query",
  "failure_pattern": "intermittent_timeout",
  "severity": "high"
}
```

**Valid failure_pattern values:**
`intermittent_timeout` | `error_spike` | `memory_leak` | `connection_pool_exhausted` | `cascading_crash`

**Valid severity values:** `low` | `medium` | `high` | `critical`

---

## 📊 Example Inference Log Output

```
[START] task=easy env=failure_analyzer model=mistralai/Mistral-7B-Instruct-v0.3
[STEP] step=1 action={"service_name":"payment-api","error_code":"HTTP_500"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=medium env=failure_analyzer model=mistralai/Mistral-7B-Instruct-v0.3
[STEP] step=1 action={"root_service":"auth-service","affected_service":"checkout-api"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00

[START] task=hard env=failure_analyzer model=mistralai/Mistral-7B-Instruct-v0.3
[STEP] step=1 action={"root_service":"user-db","endpoint":"/api/v1/query","failure_pattern":"intermittent_timeout","severity":"high"} reward=1.00 done=true error=null
[END] success=true steps=1 score=1.00 rewards=1.00
```
