"""jev-server: typed probabilistic decisions via first-token logit readout.

Two backends — pick one:
  --mlx MODEL_PATH          direct MLX inference (local, no API)
  --api URL --api-model ID  OpenAI-compatible API (LM Studio, ollama, etc.)

Run:
  jev-serve lmstudio-community/Qwen3.8-27B-MLX-6bit
  jev-serve qwen/qwen3.8-27b --api http://localhost:1234/v1

Portions derived from kev (https://github.com/jaredpalmer/kev), Copyright 2026 Jared Palmer, Apache 2.0.
"""
import argparse, threading
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from jev.api import SystemOneRequest, to_record, to_answers, output_tokens

app = FastAPI(title="jev-server")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
STATE: dict = {"label": None, "predictor": None, "tok": None, "lock": threading.Lock()}


def _probs(rec):
    with STATE["lock"]:
        return STATE["predictor"].probs(rec)


@app.post("/v1/systemone")
def systemone(req: SystemOneRequest):
    rec, meta = to_record(req)
    ps, m = _probs(rec)
    answers = to_answers(ps, meta)
    tok = STATE["tok"]
    out_tokens = output_tokens(tok, answers) if tok else 0
    return {"model": req.model, "answers": answers, "usage": {"input_tokens": m["tokens"], "output_tokens": out_tokens}, "latency_ms": m["latency_ms"]}


@app.get("/v1/models")
def models():
    return {"models": [{"id": "jev-latest", "run": STATE["label"]}]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", metavar="MODEL_PATH", help="MLX model path or Hub id")
    ap.add_argument("--api", default=None, metavar="BASE_URL", help="OpenAI-compatible API base URL")
    ap.add_argument("--api-key", default="local")
    ap.add_argument("--port", type=int, default=8008)
    a = ap.parse_args()

    if a.api:
        from jev.api_backend import APIPredictor
        print(f"API backend: {a.api} model={a.model}")
        STATE.update(label=a.model, predictor=APIPredictor(a.api, a.model, a.api_key), tok=None)
    else:
        from jev.mlx_backend import ScratchPredictor
        print(f"MLX backend: loading {a.model}")
        p = ScratchPredictor(a.model)
        STATE.update(label=a.model, predictor=p, tok=p.tokenizer)

    print(f"serving on :{a.port}")
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=a.port)


if __name__ == "__main__":
    main()
