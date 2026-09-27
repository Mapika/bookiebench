"""decider runner: every query is a typed question over the rendered state (marginal -> Choice, noul/cond -> Noul).

    CUDA_VISIBLE_DEVICES=2 python -m bookiebench.runners.decider_runner data/test --model Mapika/decider-2b

Modes (--mode):
    isolated  (default) one prompt row per question: state + that question only, so no question can see another. Rows of
              all instances are batched together (sorted by length, token-budget chunks). Same layout as
              Decider.system_one(independent=True); it only skips the shared-prefix cache fork.
    auto      isolated rows, except that a state of >= --share-min-tokens tokens runs once and its cache is forked to
              every question row (the same rows and answers as isolated up to bf16 round-off; used for long contexts)
    shared    Decider.system_one(independent=True) per state: identical answers up to kernel round-off, the state is run once
              and its cache forked to every question (Engine.score_shared). One call per state, so less batching.
    packed    Decider.system_one(independent=False): all questions of one (instance, variant) behind one copy of the state in
              one row; later questions see earlier question texts.

Temperature: the package default (decider_config.json "temperature" and, if present, "temperature_by_type"; decider-ai
>= 1.4 semantics). --temperature T overrides it with one scalar (and switches the by-type map off), e.g. 1 for raw logits.

The decider package: by default the copy bundled in the model snapshot (<snapshot>/decider, the version the weights were
released with; 1.4.x for decider-2b v11 / decider-4b v2.1), else $DECIDER_SRC (a checkout of the public decider repo). --decider-src
picks a path explicitly.
"""
import argparse
import glob
import json
import os
import sys
import time
from collections import OrderedDict
from pathlib import Path

HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
PUBLIC_SRC = os.environ.get("DECIDER_SRC", "")  # optional local checkout of the public decider repo


def resolve_snapshot(model):
    """A local directory with weights: `model` itself if it is one, else the HF-cache snapshot of repo `model`
    (refs/main first, then the newest snapshot holding model.safetensors and decider_config.json)."""
    if os.path.isdir(model):
        return model
    repo = os.path.join(HF_HUB, "models--" + model.replace("/", "--"))
    cands = []
    ref = os.path.join(repo, "refs", "main")
    if os.path.exists(ref):
        cands.append(os.path.join(repo, "snapshots", open(ref).read().strip()))
    cands += sorted(glob.glob(os.path.join(repo, "snapshots", "*")), key=os.path.getmtime, reverse=True)
    for c in cands:
        if glob.glob(os.path.join(c, "*.safetensors")) and os.path.exists(os.path.join(c, "decider_config.json")):
            return c
    raise FileNotFoundError(f"no local snapshot with weights for {model} under {repo}")


def pick_src(snapshot, src=None):
    if src:
        return src
    if os.path.exists(os.path.join(snapshot, "decider", "infer.py")):
        return snapshot
    for c in sorted(glob.glob(os.path.join(HF_HUB, "models--Mapika--decider-*", "snapshots", "*", "decider", "temperature.py")),
                    key=os.path.getmtime, reverse=True):
        return os.path.dirname(os.path.dirname(c))      # newest bundled package (1.4.x) for snapshots that ship none
    return PUBLIC_SRC


def question_spec(row):
    if row.kind == "choice":
        return {"type": "choice", "instructions": row.question, "criteria": list(row.options)}
    return {"type": "noul", "instructions": row.question}


def to_answer(row, p):
    if row.kind == "noul":
        s = p[0] + p[1]
        return [p[1] / s]
    p = list(p[:len(row.options)]); s = sum(p) or 1.0
    return [x / s for x in p]


class DeciderScorer:
    wants_groups = True

    def __init__(self, model, name=None, mode="isolated", temperature=None, graphs=False, device="cuda", src=None,
                 max_batch=256, token_budget=131072, max_state_tokens=32768, share_min_tokens=1024, fork_rows=32):
        self.snapshot = resolve_snapshot(model)
        self.src = pick_src(self.snapshot, src)
        if self.src not in sys.path:
            sys.path.insert(0, self.src)
        import torch
        from decider.infer import Decider
        import decider.infer as DI
        self.torch = torch
        self.d = Decider(self.snapshot, device=device, temperature=temperature, use_graphs=graphs)
        self.cfg = json.load(open(os.path.join(self.snapshot, "decider_config.json")))
        self.name = name or ("decider-" + model.rstrip("/").split("decider-")[-1] if "decider-" in model else os.path.basename(model))
        self.mode, self.graphs = mode, graphs
        self.max_batch = min(max_batch, 64) if graphs else max_batch
        self.token_budget, self.max_state_tokens = token_budget, max_state_tokens
        self.share_min_tokens, self.fork_rows = share_min_tokens, fork_rows
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):   # decider.engine's policy (cuDNN SDPA is wrong on the cached path)
            torch.backends.cuda.enable_cudnn_sdp(False)
        self.pkg_file = DI.__file__
        self.stats = dict(forwards=0, tokens=0, padded_tokens=0)

    def info(self):
        d = self.d
        return dict(runner="decider", mode=self.mode, graphs=self.graphs, snapshot=self.snapshot, package=self.pkg_file,
                    version=self.cfg.get("version"), temperature=getattr(d, "T", None),
                    temperature_by_type=getattr(d, "T_by_type", None), **self.stats)

    # ---- scoring of prompt items (dicts from decider.prompt.build carrying "types")
    def _temps(self, items):
        try:
            from decider import temperature as TT
            return TT.for_items(self.d.T, getattr(self.d, "T_by_type", {}) or {}, items)
        except ImportError:                                   # decider-ai 1.1.x: one scalar
            return self.d.T

    def _score_items(self, items):
        """-> one probability list per question slot, flattened in item order."""
        torch = self.torch
        from decider.model import collate
        temps = self._temps(items)
        order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
        res = [None] * len(items); i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < self.max_batch and len(items[order[j]]["ids"]) * (j - i + 1) <= self.token_budget:
                j += 1
            idx = order[i:j]; chunk = [items[k] for k in idx]
            T = [temps[k] for k in idx] if isinstance(temps, list) else temps
            self.stats["forwards"] += 1; self.stats["tokens"] += sum(len(it["ids"]) for it in chunk)
            self.stats["padded_tokens"] += len(chunk) * max(len(it["ids"]) for it in chunk)
            with torch.no_grad():
                if self.d.eng is not None:
                    pr = self.d.eng.score_items(chunk, temperature=T)
                else:
                    bt = collate(chunk, self.d.m.tok.pad_token_id)
                    lg = self.d.m.slot_logits(*[bt[k].to(self.d.dev) for k in ("input_ids", "attention_mask", "slot_idx", "slot_batch", "nopts")])
                    if isinstance(T, list):
                        from decider import temperature as TT
                        p = TT.scaled_softmax(lg, TT.slot_temperatures(T, chunk)).cpu()
                    else:
                        p = torch.softmax(lg / T, -1).cpu()
                    pr, c = [], 0
                    for it in chunk:
                        pr.append(p[c:c + len(it["slots"])]); c += len(it["slots"])
            for k, pk in zip(idx, pr):
                res[k] = [x.tolist() for x in pk]
            i = j
        return [p for r in res for p in r]

    def _score_shared(self, items):
        """Engine.score_shared without an Engine: rows that start with the same state tokens run the prefix once and fork
        its cache (attention KV + delta-net states) to every question suffix. One probability list per item (1 slot)."""
        torch = self.torch
        from decider.engine import read_slots, fill_ids
        try:
            from decider.temperature import slot_temperatures
        except ImportError:
            slot_temperatures = lambda T, its: T
        core, W = self.d.m.lm.model, self.d.m.lm.lm_head.weight[self.d.m.letters]
        temps = self._temps(items)
        ids = [it["ids"] for it in items]
        lcp = 0; short = min(len(x) for x in ids) - 1
        while lcp < short and all(x[lcp] == ids[0][lcp] for x in ids): lcp += 1
        res = []
        for a in range(0, len(items), self.fork_rows):
            chunk = items[a:a + self.fork_rows]; n = len(chunk)
            T = temps[a:a + n] if isinstance(temps, list) else temps
            with torch.no_grad():
                cache = core(input_ids=torch.tensor(ids[0][:lcp], device=self.d.dev)[None], use_cache=True).past_key_values
                cache.reorder_cache(torch.zeros(n, dtype=torch.long, device=self.d.dev))
                suf = fill_ids([it["ids"][lcp:] for it in chunk], n, max(len(it["ids"]) - lcp for it in chunk), self.d.m.tok.pad_token_id)
                h = core(input_ids=suf.to(self.d.dev), past_key_values=cache, use_cache=True).last_hidden_state
                sl = [it["slots"][0] - lcp for it in chunk]
                hs = h[torch.arange(n, device=self.d.dev), torch.tensor(sl, device=self.d.dev)]
                pr = read_slots(torch.nn.functional.linear(hs, W).float()[:, None, :], list(range(n)), [0] * n,
                                [it["nopts"][0] for it in chunk], slot_temperatures(T, chunk), [1] * n)
            self.stats["forwards"] += 2; self.stats["tokens"] += lcp + sum(len(it["ids"]) - lcp for it in chunk)
            self.stats["shared_states"] = self.stats.get("shared_states", 0) + 1
            res += [p[0].tolist() for p in pr]
        return res

    def _items(self, state, rows, independent):
        qs = OrderedDict((f"r{n}", question_spec(r)) for n, r in enumerate(rows))
        if hasattr(self.d, "_system_one_items"):
            _, _, items = self.d._system_one_items(state, qs, independent=independent, max_state_tokens=self.max_state_tokens,
                                                   layout="state_first", isolated=False)
            return items
        # decider-ai 1.1.x: rebuild the same rows (render_question + prompt.build, options kept in order)
        from decider.systemone import render_question
        from decider.prompt import build, MAX_OPTIONS
        from decider.infer import Example, Q

        class _Keep:
            def shuffle(self, x): pass
            def sample(self, xs, k): return xs[:k]
        rq = [render_question(v) for v in qs.values()]
        groups = [[r] for r in rq] if independent else [rq]
        items = []
        for g in groups:
            it = build(Example(state, [Q(r["question"], list(r["options"]), 0) for r in g]), self.d.m.tok, _Keep(),
                       max_options=MAX_OPTIONS, max_ctx_tokens=self.max_state_tokens)
            it["types"] = [r["type"] for r in g]; items.append(it)
        return items

    def score(self, rows, groups=None):
        t = time.time()
        if self.mode == "shared":
            by_state = OrderedDict()
            for k, r in enumerate(rows):
                by_state.setdefault(r.state, []).append(k)
            out = [None] * len(rows)
            for state, ks in by_state.items():
                qs = OrderedDict((f"r{k}", question_spec(rows[k])) for k in ks)
                a = self.d.system_one(state, qs, independent=True, max_state_tokens=self.max_state_tokens, layout="state_first")["answers"]
                for k in ks:
                    x = a[f"r{k}"]
                    out[k] = [x["noul"]] if rows[k].kind == "noul" else [x["probabilities"][o] for o in rows[k].options]
                    out[k] = to_answer(rows[k], out[k] if rows[k].kind == "choice" else [1 - out[k][0], out[k][0]])
            return out
        if self.mode == "auto":                                # isolated rows; long states share one prefix pass
            out = [None] * len(rows); rest = []
            by_state = OrderedDict()
            for k, r in enumerate(rows):
                by_state.setdefault(r.state, []).append(k)
            for state, ks in by_state.items():
                if len(ks) > 1 and len(self.d.m.tok.encode(state, add_special_tokens=False)) >= self.share_min_tokens:
                    items = [self._items(state, [rows[k]], True)[0] for k in ks]
                    for k, p in zip(ks, self._score_shared(items)):
                        out[k] = to_answer(rows[k], p)
                else:
                    rest += ks
            if rest:
                items = [self._items(rows[k].state, [rows[k]], True)[0] for k in rest]
                for k, p in zip(rest, self._score_items(items)):
                    out[k] = to_answer(rows[k], p)
            return out
        if self.mode == "isolated":
            groups = [[k] for k in range(len(rows))]
            items = [it for k in range(len(rows)) for it in self._items(rows[k].state, [rows[k]], True)]
            flat = self._score_items(items)
            return [to_answer(rows[k], flat[k]) for k in range(len(rows))]
        if self.mode == "packed":
            groups = groups or [[k] for k in range(len(rows))]
            seen, packs = set(), []
            for g in groups:                                   # a row answered in the first pack that contains it
                by_state = OrderedDict()
                for k in g:
                    if k not in seen:
                        seen.add(k); by_state.setdefault(rows[k].state, []).append(k)
                packs += list(by_state.values())
            items = [self._items(rows[ks[0]].state, [rows[k] for k in ks], False)[0] for ks in packs]
            flat = self._score_items(items)
            out = [None] * len(rows); c = 0
            for ks in packs:
                for k in ks:
                    out[k] = to_answer(rows[k], flat[c]); c += 1
            return out
        raise ValueError(f"unknown mode {self.mode!r}")


def main(argv=None):
    from bookiebench.runners import core
    ap = core.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", required=True, help="HF repo (Mapika/decider-2b) or a local model dir")
    ap.add_argument("--mode", default="isolated", choices=["isolated", "shared", "packed", "auto"])
    ap.add_argument("--share-min-tokens", type=int, default=1024, help="--mode auto: states this long share one prefix pass")
    ap.add_argument("--temperature", type=float, default=None, help="override the package default temperature")
    ap.add_argument("--graphs", action="store_true", help="CUDA graphs + torch.compile engine (default: eager)")
    ap.add_argument("--decider-src", default=None)
    ap.add_argument("--max-batch", type=int, default=256)
    ap.add_argument("--token-budget", type=int, default=131072)
    args = ap.parse_args(argv)
    sc = DeciderScorer(args.model, name=args.name, mode=args.mode, temperature=args.temperature, graphs=args.graphs,
                       src=args.decider_src, max_batch=args.max_batch, token_budget=args.token_budget,
                       share_min_tokens=args.share_min_tokens)
    return core.run_cli(sc, args)


if __name__ == "__main__":
    main()
