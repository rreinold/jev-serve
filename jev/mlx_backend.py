"""Scratch inference: log-likelihood scoring over MLX models — no fine-tuning, no custom head.

Two prompt strategies depending on question type:

  noul   — options are "no"/"yes", natural continuations of a direct question. Read their
            first-token logits at the boundary after (state + instruction).

  choice/score — options are long strings whose first BPE token is arbitrary (e.g. "power" from
                 "powertrain:"). Instead, present options as a labeled MCQ list inside the prompt
                 (A/B/C/D) and read single-character letter logits at "Answer:". The model has
                 seen both the state AND all options in context, so it picks the right letter.
"""
import time
import mlx.core as mx

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


class ScratchPredictor:
    def __init__(self, model_path: str):
        from mlx_lm import load
        self.model, _tok = load(model_path)
        self.model.eval()
        # TokenizerWrapper._tokenizer is the underlying HF tokenizer (callable, has .input_ids)
        self.tokenizer = _tok._tokenizer

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def _forward(self, ids: list[int]):
        x = mx.array(ids)[None]
        logits = self.model(x)
        mx.eval(logits)
        return logits[0, -1, :]  # [V]

    def probs(self, rec: dict) -> tuple[list[list[float]], dict]:
        state = rec["state"]
        results = []
        total_tokens = 0
        t0 = time.time()

        for q in rec["questions"]:
            opts = q["options"]
            qtype = q.get("type", "noul")

            if qtype in ("choice", "score"):
                # MCQ format: options in context, read letter logit at "Answer:"
                labels = LETTERS[:len(opts)]
                lines = "\n".join(f"{lbl}: {opt}" for lbl, opt in zip(labels, opts))
                prompt = f"{state}\n\n{q['instr']}\n\n{lines}\n\nAnswer:"
                prefix_ids = self._encode(prompt)
                boundary = self._forward(prefix_ids)
                # Derive label token from context: what token appears after "Answer: " for each letter.
                # Avoids hardcoding tokenizer spacing assumptions (e.g. SentencePiece leading-space BPE).
                answer_prefix_len = len(self._encode("Answer:"))
                label_ids = [self._encode(f"Answer: {lbl}")[answer_prefix_len] for lbl in labels]
                scores = [float(boundary[t]) for t in label_ids]
            else:
                # noul: "no"/"yes" are natural next tokens after a direct question
                prefix_ids = self._encode(f"{state}\n\n{q['instr']}\n")
                boundary = self._forward(prefix_ids)
                first_ids = [self._encode(opt)[0] if self._encode(opt) else 0 for opt in opts]
                scores = [float(boundary[t]) for t in first_ids]

            total_tokens += len(prefix_ids)
            probs = mx.softmax(mx.array(scores), axis=-1)
            mx.eval(probs)
            results.append(probs.tolist())

        dt = time.time() - t0
        meta = {"tokens": total_tokens, "state_tokens": len(self._encode(state)), "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}
        return results, meta
