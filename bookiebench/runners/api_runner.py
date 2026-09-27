"""OpenAI-compatible chat API baselines (DeepSeek V4.1 Flash on the local server), two variants.

    # one pass, letter logprobs (same letter prompt as logit_runner), no thinking
    python -m bookiebench.runners.api_runner results/release/_data/sub/test --variant logit --name deepseek-v41-flash-logit
    # reasoning, then one JSON of probabilities for all questions of a state (subset: --sample 100)
    python -m bookiebench.runners.api_runner results/release/_data/sub/test --variant cot --name deepseek-v41-flash-cot

logit: one request per question row, max_tokens=1, logprobs with top_logprobs=5, thinking disabled through
chat_template_kwargs. The top-5 tokens are normalised (strip "(", ")", spaces) and summed per option letter. A letter
that is not in the top-5 gets the leftover mass (1 - sum of the top-5) split evenly over the missing letters; the run
reports how often that happens (missing_letter_rows). The vector is then renormalised over the letters.
cot: one request per (state, variant): every question of that state is listed with an id, the model reasons, then
writes a final JSON {"<id>": [p_A, p_B, ...] | p_yes}. An unparseable reply or a malformed entry leaves those answers
missing (the metrics treat missing prices adversarially); parse failures are counted.
Concurrency is bounded (--concurrency, default 16, <= 32) with exponential backoff on errors.
"""
import argparse
import asyncio
import json
import math
import re
import time

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
SYSTEM_LOGIT = "Answer with the letter of exactly one option."


def _json_tail(text):
    """The last JSON object in a reply (after any reasoning)."""
    for m in reversed(list(re.finditer(r"\{", text))):
        frag = text[m.start():]
        end = frag.rfind("}")
        while end > 0:
            try:
                return json.loads(frag[:end + 1])
            except json.JSONDecodeError:
                end = frag.rfind("}", 0, end)
    return None


class APIScorer:
    wants_groups = True

    def __init__(self, base_url, model, name, variant="logit", concurrency=16, max_retries=6, timeout=600,
                 max_tokens_cot=8192):
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model, self.name, self.variant = model, name, variant
        self.conc, self.retries, self.timeout, self.max_tokens_cot = min(concurrency, 32), max_retries, timeout, max_tokens_cot
        self.stats = dict(requests=0, errors=0, retries=0, missing_letter_rows=0, rows=0, parse_failures=0, groups=0,
                          missing_answers=0, prompt_tokens=0, completion_tokens=0)

    def info(self):
        s = dict(self.stats)
        s["missing_letter_frac"] = round(s["missing_letter_rows"] / max(s["rows"], 1), 4)
        return dict(runner="api", variant=self.variant, model_id=self.model, url=self.url, **s)

    async def _post(self, session, body):
        import aiohttp
        delay = 2.0
        for attempt in range(self.retries + 1):
            try:
                async with session.post(self.url, json=body, timeout=aiohttp.ClientTimeout(total=self.timeout)) as r:
                    if r.status == 200:
                        d = await r.json()
                        self.stats["requests"] += 1
                        u = d.get("usage") or {}
                        self.stats["prompt_tokens"] += u.get("prompt_tokens", 0)
                        self.stats["completion_tokens"] += u.get("completion_tokens", 0)
                        return d
                    self.stats["errors"] += 1
            except Exception:  # noqa: BLE001 - network errors are retried
                self.stats["errors"] += 1
            if attempt < self.retries:
                self.stats["retries"] += 1
                await asyncio.sleep(delay); delay = min(delay * 2, 120)
        return None

    # ---- logit
    def _logit_body(self, row):
        from bookiebench.runners.logit_runner import prompt_for
        p = prompt_for(row)
        head, _ = p.rsplit("\nAnswer: (", 1)
        return {"model": self.model, "messages": [{"role": "system", "content": SYSTEM_LOGIT},
                                                  {"role": "user", "content": head + "\nAnswer:"}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 5,
                "chat_template_kwargs": {"thinking": False, "enable_thinking": False}}

    def _letters(self, d, n):
        try:
            top = d["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        except (TypeError, KeyError, IndexError):
            return None
        mass = [0.0] * n; tot = 0.0
        for t in top:
            p = math.exp(t["logprob"]); tot += p
            key = t["token"].strip().strip("()").strip()
            if len(key) == 1 and key in LETTERS[:n]:
                mass[LETTERS.index(key)] += p
        missing = [j for j in range(n) if mass[j] == 0.0]
        if missing:
            self.stats["missing_letter_rows"] += 1
            left = max(1.0 - tot, 0.0) / len(missing)
            for j in missing:
                mass[j] = left
        s = sum(mass)
        return [x / s for x in mass] if s > 0 else [1.0 / n] * n

    async def _score_logit(self, rows):
        import aiohttp
        sem = asyncio.Semaphore(self.conc); out = [None] * len(rows)

        async def one(session, k):
            async with sem:
                r = rows[k]; n = len(r.options) if r.kind == "choice" else 2
                d = await self._post(session, self._logit_body(r))
                p = self._letters(d, n) if d else None
                out[k] = (p[:1] if r.kind == "noul" else p) if p else None
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[one(session, k) for k in range(len(rows))])
        self.stats["rows"] += len(rows)
        return out

    # ---- cot
    def _cot_body(self, rows, ks):
        lines = []
        for k in ks:
            r = rows[k]
            if r.kind == "choice":
                opts = "; ".join(f"({LETTERS[j]}) {o}" for j, o in enumerate(r.options))
                lines.append(f'"r{k}": {r.question} Options: {opts}. Give a list of {len(r.options)} probabilities, one per '
                             f"option in the order shown, summing to 1.")
            else:
                lines.append(f'"r{k}": {r.question} Give the probability that the answer is yes, as one number.')
        user = (f"Context:\n{rows[ks[0]].state}\n\nAnswer every question below with calibrated probabilities. Think it "
                "through first; then end your reply with a single JSON object mapping each question id to its answer.\n\n"
                + "\n".join(lines))
        return {"model": self.model, "messages": [{"role": "user", "content": user}], "max_tokens": self.max_tokens_cot,
                "temperature": 0.6}

    async def _score_cot(self, rows, groups):
        import aiohttp
        sem = asyncio.Semaphore(self.conc); out = [None] * len(rows)
        seen, packs = set(), []
        for g in groups or [[k] for k in range(len(rows))]:
            by = {}
            for k in g:
                if k not in seen:
                    seen.add(k); by.setdefault(rows[k].state, []).append(k)
            packs += list(by.values())

        async def one(session, ks):
            async with sem:
                d = await self._post(session, self._cot_body(rows, ks))
                self.stats["groups"] += 1
                text = ""
                if d:
                    m = d["choices"][0]["message"]
                    text = (m.get("content") or "")
                obj = _json_tail(text) if text else None
                if not isinstance(obj, dict):
                    self.stats["parse_failures"] += 1; self.stats["missing_answers"] += len(ks); return
                for k in ks:
                    v = obj.get(f"r{k}"); r = rows[k]
                    try:
                        if r.kind == "choice":
                            v = [float(x) for x in v]
                            if len(v) != len(r.options) or min(v) < 0 or sum(v) <= 0:
                                raise ValueError
                            out[k] = [x / sum(v) for x in v]
                        else:
                            v = float(v[0] if isinstance(v, list) else v)
                            if not 0 <= v <= 1:
                                raise ValueError
                            out[k] = [v]
                    except (TypeError, ValueError):
                        self.stats["missing_answers"] += 1
        async with aiohttp.ClientSession() as session:
            await asyncio.gather(*[one(session, ks) for ks in packs])
        self.stats["rows"] += len(rows)
        return out

    def score(self, rows, groups=None):
        if self.variant == "logit":
            return asyncio.run(self._score_logit(rows))
        return asyncio.run(self._score_cot(rows, groups))


def main(argv=None):
    from bookiebench.runners import core
    ap = core.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--base-url", default="http://localhost:9103/v1")
    ap.add_argument("--model", default="deepseek-v41-flash")
    ap.add_argument("--variant", choices=["logit", "cot"], default="logit")
    ap.add_argument("--concurrency", type=int, default=16)
    args = ap.parse_args(argv)
    sc = APIScorer(args.base_url, args.model, args.name or f"deepseek-v41-flash-{args.variant}", args.variant, args.concurrency)
    return core.run_cli(sc, args)


if __name__ == "__main__":
    main()
