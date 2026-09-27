"""Zero-training logit baseline (SemIf-style) on any HF causal LM.

    CUDA_VISIBLE_DEVICES=2 python -m bookiebench.runners.logit_runner data/test --model Qwen/Qwen3.5-4B-Base

Prompt per question (one row per question, no other question visible):

    Context:
    <state>

    Question: <text>
    Options:
    (A) <option 0>
    (B) <option 1>
    Answer: (

P(option j) = softmax over the next-token logits of the letter tokens A, B, ... divided by the temperature.
Yes/no questions (--yesno letters, default): options (A) yes (B) no, read the same way.  --yesno words: the prompt ends in
"Answer (yes or no):" and P(yes) = softmax over the " yes" / " no" token logits.
--temperature: 1 (default, no fitting); bookiebench.runners.temper fits one afterwards from the T=1 predictions.
"""
import argparse
import os
import time

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def prompt_parts(row, yesno="letters"):
    """(shared prefix, question suffix): prefix + suffix == prompt_for(row)."""
    p = prompt_for(row, yesno)
    k = p.index("\n\nQuestion: ") + 2
    return p[:k], p[k:]


def prompt_for(row, yesno="letters"):
    if row.kind == "noul" and yesno == "words":
        return f"Context:\n{row.state}\n\nQuestion: {row.question}\nAnswer (yes or no):"
    opts = row.options if row.kind == "choice" else ("yes", "no")
    lines = "".join(f"\n({LETTERS[j]}) {o}" for j, o in enumerate(opts))
    return f"Context:\n{row.state}\n\nQuestion: {row.question}\nOptions:{lines}\nAnswer: ("


class LogitScorer:
    def __init__(self, model, name=None, temperature=1.0, yesno="letters", device="cuda", dtype="bfloat16",
                 max_batch=128, token_budget=131072, share_min_tokens=1024, fork_rows=32):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):   # cuDNN SDPA is wrong for masked rectangular attention on
            torch.backends.cuda.enable_cudnn_sdp(False)        # Blackwell/torch 2.14 (see decider.engine); the cached path uses it
        self.torch = torch
        self.model_id = model
        self.tok = AutoTokenizer.from_pretrained(model)
        self.m = AutoModelForCausalLM.from_pretrained(model, dtype=getattr(torch, dtype)).to(device).eval()
        self.decoder = self.m.get_decoder()                    # the backbone: last_hidden_state is after the final norm
        cfg = getattr(self.m.config, "text_config", None) or self.m.config
        self.softcap = getattr(cfg, "final_logit_softcapping", None) or getattr(self.m.config, "final_logit_softcapping", None)
        self.dev, self.T, self.yesno = device, float(temperature), yesno
        self.max_batch, self.token_budget = max_batch, token_budget
        self.name = name or (os.path.basename(model.rstrip("/")) + ("" if self.T == 1 else f"-T{self.T:g}"))
        self.letter_ids = []
        for L in LETTERS:
            ids = self.tok.encode(L, add_special_tokens=False)
            assert len(ids) == 1, f"letter {L} is not one token: {ids}"
            self.letter_ids.append(ids[0])
        self.yn_ids = []
        for w in (" yes", " no"):
            ids = self.tok.encode(w, add_special_tokens=False)
            assert len(ids) == 1, f"{w!r} is not one token: {ids}"
            self.yn_ids.append(ids[0])
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.tok.eos_token_id or 0)
        self.bos = [self.tok.bos_token_id] if (self.tok.bos_token_id is not None and
                                               self.tok("x").input_ids[:1] == [self.tok.bos_token_id]) else []
        self.share_min_tokens, self.fork_rows = share_min_tokens, fork_rows
        self.stats = dict(forwards=0, tokens=0, padded_tokens=0, shared_states=0, shared_rows=0)

    def info(self):
        return dict(runner="logit", model_id=self.model_id, temperature=self.T, yesno=self.yesno, **self.stats)

    def _cands(self, row):
        if row.kind == "noul":
            return self.yn_ids if self.yesno == "words" else self.letter_ids[:2]
        assert len(row.options) <= len(LETTERS), "more than 26 options"
        return self.letter_ids[:len(row.options)]

    def _readout(self, hl, row):
        c = self._cands(row)
        lg = (hl[None] @ self.m.get_output_embeddings().weight[c].T).float()[0]
        if self.softcap:
            lg = self.torch.tanh(lg / self.softcap) * self.softcap
        p = self.torch.softmax(lg / self.T, -1).cpu().tolist()
        return [p[0]] if row.kind == "noul" else p

    def _fork(self, cache, n):
        try:                                                   # Qwen3.5 hybrid cache (attention KV + delta-net states)
            cache.reorder_cache(self.torch.zeros(n, dtype=self.torch.long, device=self.dev))
        except (AttributeError, NotImplementedError):
            cache.batch_repeat_interleave(n)
        return cache

    def _score_shared(self, rows, idxs, out):
        """Rows with one state: the prefix (context) runs once, its cache is forked to every question suffix."""
        torch = self.torch
        pre_s = prompt_parts(rows[idxs[0]], self.yesno)[0]
        pre = self.bos + self.tok.encode(pre_s, add_special_tokens=False)
        sufs = [self.tok.encode(prompt_parts(rows[k], self.yesno)[1], add_special_tokens=False) for k in idxs]
        self.stats["shared_states"] += 1; self.stats["shared_rows"] += len(idxs)
        for a in range(0, len(idxs), self.fork_rows):
            ks, ss = idxs[a:a + self.fork_rows], sufs[a:a + self.fork_rows]
            with torch.no_grad():
                cache = self.decoder(input_ids=torch.tensor([pre], device=self.dev), use_cache=True).past_key_values
                cache = self._fork(cache, len(ks))
                T = max(len(x) for x in ss)
                ids = torch.full((len(ks), T), self.pad, dtype=torch.long)
                for b, x in enumerate(ss):
                    ids[b, :len(x)] = torch.tensor(x)
                h = self.decoder(input_ids=ids.to(self.dev), past_key_values=cache, use_cache=True).last_hidden_state
                self.stats["forwards"] += 2; self.stats["tokens"] += len(pre) + sum(map(len, ss))
                for b, (k, x) in enumerate(zip(ks, ss)):
                    out[k] = self._readout(h[b, len(x) - 1], rows[k])

    def score(self, rows):
        torch = self.torch
        out = [None] * len(rows)
        if self.share_min_tokens:
            groups = {}
            for k, r in enumerate(rows):
                groups.setdefault(r.state, []).append(k)
            rest = []
            for st, ks in groups.items():
                if len(ks) > 1 and len(self.tok.encode(st, add_special_tokens=False)) >= self.share_min_tokens:
                    self._score_shared(rows, ks, out)
                else:
                    rest += ks
            if not rest:
                return out
        else:
            rest = list(range(len(rows)))
        enc = {k: self.bos + self.tok.encode(prompt_for(rows[k], self.yesno), add_special_tokens=False) for k in rest}
        order = sorted(rest, key=lambda i: len(enc[i]))
        i = 0
        while i < len(order):
            j = i + 1
            while j < len(order) and j - i < self.max_batch and len(enc[order[j]]) * (j - i + 1) <= self.token_budget:
                j += 1
            idx = order[i:j]; T = max(len(enc[k]) for k in idx)
            ids = torch.full((len(idx), T), self.pad, dtype=torch.long)
            att = torch.zeros((len(idx), T), dtype=torch.long)
            for b, k in enumerate(idx):                         # right padding: causal, so pads never reach real positions
                ids[b, :len(enc[k])] = torch.tensor(enc[k]); att[b, :len(enc[k])] = 1
            last = torch.tensor([len(enc[k]) - 1 for k in idx])
            self.stats["forwards"] += 1; self.stats["tokens"] += int(att.sum()); self.stats["padded_tokens"] += ids.numel()
            with torch.no_grad():
                h = self.decoder(input_ids=ids.to(self.dev), attention_mask=att.to(self.dev)).last_hidden_state
                hl = h[torch.arange(len(idx), device=self.dev), last.to(self.dev)]
                for b, k in enumerate(idx):
                    out[k] = self._readout(hl[b], rows[k])
            i = j
        return out


def main(argv=None):
    from bookiebench.runners import core
    ap = core.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--yesno", default="letters", choices=["letters", "words"])
    ap.add_argument("--max-batch", type=int, default=128)
    ap.add_argument("--token-budget", type=int, default=131072)
    ap.add_argument("--share-min-tokens", type=int, default=1024,
                    help="states of at least this many tokens run once with their cache forked to every question (0: off)")
    ap.add_argument("--fork-rows", type=int, default=32)
    args = ap.parse_args(argv)
    sc = LogitScorer(args.model, name=args.name, temperature=args.temperature, yesno=args.yesno, max_batch=args.max_batch,
                     token_budget=args.token_budget, share_min_tokens=args.share_min_tokens, fork_rows=args.fork_rows)
    return core.run_cli(sc, args)


if __name__ == "__main__":
    main()
