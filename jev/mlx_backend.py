"""Scratch inference: log-likelihood scoring over MLX models — no fine-tuning, no custom head.

For each option in a question, we run a forward pass over (state + instr + option) and sum the
log-probs of the option tokens given the prefix. Softmax across options gives the probability
distribution. This is the same approach openjev.com uses in the browser via wllama.

N forward passes per question (one per option). Prefix KV caching would collapse this to ~1 pass;
left as a future optimisation since for a 27B model the option tokens are short relative to state.
"""
import time
import mlx.core as mx
import mlx.nn as nn


class ScratchPredictor:
    def __init__(self, model_path: str):
        from mlx_lm import load
        self.model, _tok = load(model_path)
        self.model.eval()
        # TokenizerWrapper._tokenizer is the underlying HF tokenizer (callable, has .input_ids)
        self.tokenizer = _tok._tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def probs(self, rec: dict) -> tuple[list[list[float]], dict]:
        state = rec["state"]
        results = []
        total_tokens = 0
        t0 = time.time()

        for q in rec["questions"]:
            prefix_ids = self._encode(f"{state}\n\n{q['instr']}\n")
            option_ids_list = [self._encode(opt) for opt in q["options"]]
            scores = []

            # One forward pass on the prefix; read each option's first token logit at the boundary.
            # This matches openjev's single-position logit readout and avoids length bias.
            ids = mx.array(prefix_ids)[None]       # [1, L_p]
            logits = self.model(ids)               # [1, L_p, V]
            mx.eval(logits)
            boundary_logits = logits[0, -1, :]     # [V] — next-token distribution after prefix
            first_ids = [opt_ids[0] if opt_ids else 0 for opt_ids in option_ids_list]
            scores = [float(boundary_logits[t]) for t in first_ids]
            total_tokens += len(prefix_ids)

            probs = mx.softmax(mx.array(scores), axis=-1)
            mx.eval(probs)
            results.append(probs.tolist())

        dt = time.time() - t0
        meta = {"tokens": total_tokens, "state_tokens": len(self._encode(state)), "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}
        return results, meta
