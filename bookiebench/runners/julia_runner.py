"""Julia-1 (SupersonicLabs, 144M mmBERT-small encoder decision model) runner.

    CUDA_VISIBLE_DEVICES=2 python -m bookiebench.runners.julia_runner data/test data/heldout

Input format (the package's julia.data.sequence, strict layout):
    [CLS] "<type> question: <text>" [SEP] ([MASK] " <option>")* [SEP] <state> [SEP]
one score per option read at its [MASK] marker. Marginal -> type "choice" with the option texts; noul/cond -> type "noul"
with options ["false", "true"] (the package's Boolean mode) and P(yes) = p(true).
Scores are raw logits: we softmax them (temperature 1), exactly as the package's named-question API (julia.typed) does.

Context limit: --max-length 1024 (the length the published benchmarks used). A state that does not fit is truncated from the
LEFT (the earliest tokens are dropped, so the latest evidence and the question/options are always kept); the package itself
would cut the right end. The fraction of truncated rows is reported in run_info (truncated_rows / truncated_frac).
"""
import argparse
import glob
import os
import sys

HF_HUB = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")


def snapshot(model="SupersonicLabs/Julia-1"):
    if os.path.isdir(model):
        return model
    repo = os.path.join(HF_HUB, "models--" + model.replace("/", "--"))
    ref = os.path.join(repo, "refs", "main")
    if os.path.exists(ref):
        return os.path.join(repo, "snapshots", open(ref).read().strip())
    return sorted(glob.glob(os.path.join(repo, "snapshots", "*")))[-1]


class JuliaScorer:
    def __init__(self, model="SupersonicLabs/Julia-1", name="Julia-1", device="cuda", max_length=1024, head_length=256,
                 batch_size=256):
        self.root = snapshot(model)
        if self.root not in sys.path:
            sys.path.insert(0, self.root)
        import torch
        from julia import load_model
        self.torch = torch
        self.eng = load_model(self.root, device=device, max_length=max_length, head_length=head_length,
                              batch_size=batch_size, strict_encoding=False)
        enc = self.eng.model.encoder                          # the package's fast encoder path calls a transformers-5.0
        if hasattr(enc, "_julia_original_forward"):           # private method missing in 5.17: use the stock ModernBERT
            enc.forward = enc._julia_original_forward         # forward (same model, it only skips unused outputs)
        self.name, self.max_length, self.head_length = name, max_length, head_length
        self.stats = dict(scored_rows=0, truncated_rows=0, state_tokens_dropped=0)

    def info(self):
        s = dict(self.stats)
        s["truncated_frac"] = round(s["truncated_rows"] / max(s["scored_rows"], 1), 4)
        return dict(runner="julia", snapshot=self.root, max_length=self.max_length, head_length=self.head_length,
                    temperature=1.0, scores="raw logits, softmaxed by the runner", **s)

    def _encode(self, row):
        from julia.data import sequence
        req = dict(state="", question=row.question, type="choice" if row.kind == "choice" else "noul",
                   options=list(row.options) if row.kind == "choice" else ["false", "true"])
        enc = sequence(self.eng._tokens, req, self.max_length, self.head_length)      # head + options, empty state
        ids = enc["ids"][:-1]                                  # drop the final SEP after the (empty) state
        sep = self.eng.tokenizer.sep_token_id
        mask = self.eng.tokenizer.mask_token
        st = self.eng._tokens(row.state.replace(mask, " "))["input_ids"]
        room = self.max_length - len(ids) - 1
        self.stats["scored_rows"] += 1
        if len(st) > room:
            self.stats["truncated_rows"] += 1; self.stats["state_tokens_dropped"] += len(st) - room
            st = st[len(st) - room:]                           # keep the end of the state
        enc["ids"] = ids + list(st) + [sep]
        return enc

    def score(self, rows):
        torch = self.torch
        encoded = [self._encode(r) for r in rows]
        out = [None] * len(rows)
        with torch.inference_mode():
            for idx, values in self.eng._batches(encoded):
                v = values.float()
                for local, i in enumerate(idx):
                    n = len(encoded[i]["markers"])
                    p = torch.softmax(v[local, :n], -1).cpu().tolist()
                    out[i] = [p[1]] if rows[i].kind == "noul" else p
        return out


def main(argv=None):
    from bookiebench.runners import core
    ap = core.add_common_args(argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter))
    ap.add_argument("--model", default="SupersonicLabs/Julia-1")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-length", type=int, default=1024)
    ap.add_argument("--head-length", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=256)
    args = ap.parse_args(argv)
    sc = JuliaScorer(args.model, name=args.name or "Julia-1", device=args.device, max_length=args.max_length,
                     head_length=args.head_length, batch_size=args.batch_size)
    return core.run_cli(sc, args)


if __name__ == "__main__":
    main()
