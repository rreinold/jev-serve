# jev-serve

[![Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue?style=for-the-badge)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org)
[![MLX](https://img.shields.io/badge/MLX-ready-brightgreen?style=for-the-badge)](https://github.com/ml-explore/mlx)

**Typed probabilistic decisions from any LLM. No text generation.**

`jev-serve` exposes a `/v1/systemone` endpoint that scores structured decisions via **first-token logit readout** — the same approach openjev.com uses in the browser, running server-side on any MLX model or OpenAI-compatible API.

- **34× faster** than structured JSON generation (0.23s vs 7.80s per decision)
- **Full probability distributions** — not a point estimate, a calibrated `p` per option
- **Three question types**: `choice` (pick one), `noul` (0–1 probability), `score` (ordinal level)
- **Two backends**: direct MLX inference or any OpenAI-compatible API (ollama, etc.)
- Apache 2.0 — derived from [kev](https://github.com/jaredpalmer/kev) by Jared Palmer

---

## Install

```bash
# MLX backend (Apple Silicon)
uv pip install -e ".[mlx]"

# OpenAI-compatible API backend
uv pip install -e ".[api]"
```

## Start

```bash
# MLX — direct inference on any local model
jev-serve --mlx lmstudio-community/Qwen3.8-27B-MLX-6bit

# API — logit readout via ollama, LM Studio, OpenAI, etc.
jev-serve --api http://localhost:11434/v1 --api-model gemma4:e4b-mlx
```

## Usage

```bash
curl -X POST http://localhost:8008/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "state": "2010 Infiniti G37 S, 6MT, daily driver, performance-oriented owner.",
    "questions": {
      "top_mod": {
        "type": "choice",
        "instructions": "What is the most popular upgrade category?",
        "criteria": {
          "exhaust": "Exhaust",
          "intake":  "Intake",
          "wheels":  "Wheels",
          "audio":   "Audio"
        }
      },
      "wants_power": {
        "type": "noul",
        "instructions": "Is this owner primarily interested in power gains?"
      }
    }
  }'
```

```json
{
  "answers": {
    "top_mod": {
      "type": "choice",
      "choice": "exhaust",
      "confidence": 0.51,
      "probabilities": { "exhaust": 0.64, "intake": 0.03, "audio": 0.05, "wheels": 0.28 }
    },
    "wants_power": { "type": "noul", "noul": 0.87 }
  },
  "latency_ms": 190
}
```

## Question types

| Type | Output | Use for |
|---|---|---|
| `choice` | `choice` + `probabilities` per key | Pick one from N options |
| `noul` | `noul` ∈ [0, 1] | Yes/no probability |
| `score` | `score` (expected level) + `probabilities` | Ordinal rating |

## Benchmark

Same model (`Qwen3.8-27B`), same question, 20 products:

|  | jev-serve (logit readout) | LLM structured output |
|---|---|---|
| Mean / product | **0.23 s** | 7.80 s |
| 10 k products | **0.6 h** | 21.7 h |
| Speedup | **34×** | — |
| Probabilities | ✅ full distribution | ❌ point estimate |
| Output tokens | 0 | ~23 |

## How it works

Instead of generating text, `jev-serve` reads the model's **next-token logit distribution** at the boundary position (after the prompt), picks out each option's first token, and normalizes with softmax. One forward pass per question; no sampling, no JSON parsing, no hallucinated fields.

This is identical to what [openjev.com](https://openjev.com) does in the browser with wllama.

## Attribution

Portions derived from [kev](https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache 2.0.
