"""Format shift: render the same world as an email thread, a chat transcript, a log file, statement templates plus a
CSV of parameters, or a markdown report.

Only the surface changes. The prelude keeps every unit in its original order; evidence step k gets a wrapper that
depends on k only (never on the content), applied identically to every `next_evidence` alternative, so it carries no
information. JSON preludes are kept as JSON (compacted into one line for logs/chat, flattened to rows for CSV).
"""
from __future__ import annotations

import csv
import io
import json
import re

from bookiebench.sims.common import FIRST_NAMES, pick

from .common import is_json_prelude, map_evidence, start, transform_rng, units

FORMATS = ["email", "chat", "log", "csv", "markdown"]

NUM_TOKEN = re.compile(r"\d{1,2}:\d{2}|\d+(?:\.\d+)?(?:\s?%| percent\b)?")
DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _names(rng, text, k):
    cands = [n for n in FIRST_NAMES if n not in text]
    idx = rng.choice(len(cands), size=k, replace=False)
    return [cands[int(i)] for i in idx]


def _one_line(u):
    if u.get("json"):
        return u.get("prefix", "") + json.dumps(json.loads(u["text"]), ensure_ascii=False)
    return " ".join(u["text"].split("\n")) if not u["text"].lstrip().startswith("- ") else u["text"].replace("\n", " ")


def _alltext(inst):
    return inst["prelude"] + " ".join(st["evidence"] for st in inst["steps"])


# ----------------------------------------------------------------------------------------------------------------------

def _email(inst, rng):
    a, *others = _names(rng, _alltext(inst), 4)
    subj = pick(rng, ["Case background", "Notes for the team", "Situation summary", "Background figures"])
    day, mon, d = pick(rng, DAYS), pick(rng, MONTHS), int(rng.integers(1, 28))
    h0 = int(rng.integers(7, 11))
    dom = pick(rng, ["example.org", "example.net", "example.com"])
    pre = (f"From: {a} <{a.lower()}@{dom}>\nTo: case-team@{dom}\nDate: {day}, {d} {mon} 2026 {h0:02d}:00\n"
           f"Subject: {subj}\n\nHi all,\n\n{inst['prelude']}\n\nI will forward the observations as they come in.\n\n"
           f"Best,\n{a}")
    senders = [pick(rng, others) for _ in inst["steps"]]

    def wrap(t, k):
        return (f"-----\nFrom: {senders[k]} <{senders[k].lower()}@{dom}>\nTo: case-team@{dom}\n"
                f"Date: {day}, {d} {mon} 2026 {h0 + 1 + k // 6:02d}:{(k * 10) % 60:02d}\nSubject: Re: {subj}\n\n"
                f"New observation: {t}\n\n{senders[k]}")

    return pre, wrap


def _chat(inst, rng):
    a, b = _names(rng, _alltext(inst), 2)
    h, m = int(rng.integers(8, 20)), int(rng.integers(0, 30))
    us = units(inst["prelude"])
    lines = [f"--- #case-room, {pick(rng, MONTHS)} {int(rng.integers(1, 28))} ---"]
    for i, u in enumerate(us):
        lines.append(f"[{h:02d}:{(m + i) % 60:02d}] {a}: {_one_line(u)}")
    lines.append(f"[{h:02d}:{(m + len(us)) % 60:02d}] {b}: ok, I'll post the observations here as they arrive")

    def wrap(t, k):
        return f"[{(h + 1 + k // 12) % 24:02d}:{(5 * k) % 60:02d}] {b}: {' '.join(t.splitlines())}"

    return "\n".join(lines), wrap


def _log(inst, rng):
    date = f"2026-{int(rng.integers(1, 13)):02d}-{int(rng.integers(1, 28)):02d}"
    svc = pick(rng, ["casebook", "tracker", "field-notes", "intake"])
    h = int(rng.integers(0, 20))
    us = units(inst["prelude"])
    lines = [f"{date}T{h:02d}:00:00Z INFO  {svc}[{int(rng.integers(100, 9999))}]: session opened"]
    for i, u in enumerate(us):
        lines.append(f"{date}T{h:02d}:00:{i + 1:02d}Z INFO  {svc}: context: {_one_line(u)}")

    def wrap(t, k):
        return f"{date}T{h + 1:02d}:{k:02d}:00Z INFO  {svc}: observation #{k + 1}: {' '.join(t.splitlines())}"

    return "\n".join(lines), wrap


def _flatten(obj, path=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from _flatten(v, f"{path} > {k}" if path else str(k))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from _flatten(v, f"{path}[{i + 1}]")
    else:
        yield path, obj


def _csv_rows(rows):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    for r in rows:
        w.writerow(r)
    return buf.getvalue().rstrip("\n")


def _csv(inst, rng):
    us = units(inst["prelude"])
    stmts, tables, rows = [], [], []
    for i, u in enumerate(us):
        sid = f"S{i + 1}"
        if u.get("json"):
            kv = [("key", "value")] + [(k, str(v).lower() if isinstance(v, bool) else v)
                                       for k, v in _flatten(json.loads(u["text"]))]
            lab = (u.get("prefix") or "").strip().rstrip("=:").strip()
            stmts.append(f"{sid}: the parameters{' (' + lab + ')' if lab else ''} are listed in parameters.csv below "
                         f"(one row per key).")
            tables.append(_csv_rows(kv))
            continue

        def sub(m, sid=sid):
            pid = f"p{len(rows) + 1}"
            rows.append((pid, sid, m.group(0)))
            return f"[{pid}]"

        stmts.append(f"{sid}: " + NUM_TOKEN.sub(sub, u["text"].replace("\n", " ")))
    parts = ["The case is described by numbered statements (statements.txt). A bracketed id such as [p1] stands for "
             "the value in the row of parameters.csv with that id.", "", "statements.txt", "\n".join(stmts)]
    if rows:
        parts += ["", "parameters.csv", _csv_rows([("param", "statement", "value")] + rows)]
    for t in tables:
        parts += ["", "parameters.csv", t]
    parts += ["", "observations.csv (one row per observation, appended as they arrive)", "step,observation"]

    def wrap(t, k):
        return _csv_rows([(k + 1, t)])

    return "\n".join(parts), wrap


def _markdown(inst, rng):
    title = pick(rng, ["Case report", "Situation report", "Briefing note", "Analyst summary"])
    us = units(inst["prelude"])
    body = [f"# {title}", "", "## Background", ""]
    for i, u in enumerate(us):
        if u.get("json"):
            if u.get("prefix", "").strip():
                body.append(f"**{u['prefix'].strip()}**")
            body += ["```json", json.dumps(json.loads(u["text"]), indent=2, ensure_ascii=False), "```"]
        elif u["text"].lstrip().startswith("- ") or "\n- " in u["text"]:
            body.append(u["text"])
        elif re.search(r"\d", u["text"]):
            body.append(f"- {u['text']}")
        else:
            body.append(u["text"] if i == 0 else f"- {u['text']}")
    body += ["", "## Observations (in order of arrival)"]

    def wrap(t, k):
        return f"### Observation {k + 1}\n{t}"

    return "\n".join(body), wrap


_RENDER = {"email": _email, "chat": _chat, "log": _log, "csv": _csv, "markdown": _markdown}


def format_shift(inst, fmt: str, seed: int = 0):
    name = f"format_{fmt}"
    rng = transform_rng(inst, name, seed)
    out = start(inst, name, format=fmt, json_prelude=is_json_prelude(inst["prelude"]))
    pre, wrap = _RENDER[fmt](inst, rng)
    out["prelude"] = pre
    map_evidence(out, wrap)
    return out
