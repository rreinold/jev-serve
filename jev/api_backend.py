"""OpenAI-compatible API backend: logit readout via top_logprobs.

Works with ollama or any OpenAI-compatible server that returns logprobs.
Note: LM Studio does not support logprobs (returns null) — use ollama or OpenAI.
Thinking models (Qwen3.x, DeepSeek-R1) suppress thinking via system prompt; if
that fails (model ignores it), the boundary token will be wrong and scores degrade.

Uses A/B/C/... letter labels in the prompt so each option maps to one guaranteed
single token — avoids BPE subword issues with option text like "Intake" → "Int".
"""
import time
import math

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

NO_THINK_SYSTEM = "/no_think\nRespond with only the letter of the correct option. No explanation."


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
            options = q["options"]
            labels = LETTERS[:len(options)]

            option_lines = "\n".join(f"{lbl}: {opt}" for lbl, opt in zip(labels, options))
            prompt = f"{state}\n\n{q['instr']}\n\n{option_lines}\n\nAnswer (letter only):"

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": NO_THINK_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1,
                temperature=0,
                logprobs=True,
                top_logprobs=5,
            )
            total_tokens += resp.usage.prompt_tokens if resp.usage else 0

            lp_content = resp.choices[0].logprobs.content if resp.choices[0].logprobs else []
            top = lp_content[0].top_logprobs if lp_content else []
            logprob_map = {entry.token: entry.logprob for entry in top}

            scores = []
            for lbl in labels:
                # Try "A", " A" (space-prefixed BPE variant)
                lp = logprob_map.get(lbl, logprob_map.get(f" {lbl}", -100.0))
                scores.append(lp)

            max_lp = max(scores)
            exps = [math.exp(lp - max_lp) for lp in scores]
            total = sum(exps)
            results.append([e / total for e in exps])

        dt = time.time() - t0
        meta = {"tokens": total_tokens, "state_tokens": 0, "latency_ms": round(dt * 1000, 1), "prefix_cache_hit": False}
        return results, meta
