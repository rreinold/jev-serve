"""OpenAI-compatible API backend: logit readout via top_logprobs.

Works with LM Studio, ollama, any OpenAI-compatible server that supports logprobs.
Matches openjev's approach: one completion call per question with max_tokens=1,
reads top_logprobs at the boundary position, normalizes over supplied options.
"""
import time
import math


class APIPredictor:
    def __init__(self, base_url: str, model: str, api_key: str = "local"):
        from openai import OpenAI
        self.client = OpenAI(base_url=base_url, api_key=api_key)
        self.model = model

    def probs(self, rec: dict) -> tuple[list[list[float]], dict]:
        state = rec["state"]
        results = []
        total_tokens = 0
        t0 = time.time()

        for q in rec["questions"]:
            prompt = f"{state}\n\n{q['instr']}\n"
            options = q["options"]

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=20,
            )
            total_tokens += resp.usage.prompt_tokens if resp.usage else 0

            top = resp.choices[0].logprobs.content[0].top_logprobs if resp.choices[0].logprobs else []
            logprob_map = {entry.token: entry.logprob for entry in top}

            # Get first token of each option; fall back to -inf if not in top_logprobs
            from openai import OpenAI  # tokenizer not available; use raw string match
            scores = []
            for opt in options:
                first_token = opt.split()[0] if opt else opt
                # try exact match, then with leading space (common BPE artifact)
                lp = logprob_map.get(f" {first_token}", logprob_map.get(first_token, -100.0))
                scores.append(lp)

            # softmax over log-probs
            max_lp = max(scores)
            exps = [math.exp(lp - max_lp) for lp in scores]
            total = sum(exps)
            results.append([e / total for e in exps])

        dt = time.time() - t0
        meta = {"tokens": total_tokens, "state_tokens": 0, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}
        return results, meta
