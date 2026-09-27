"""Naive-Bayes text classification (spam filter / ticket router) with word-presence evidence.

Word checks are conditionally independent given the class (exchangeable). Variables: class, whether a not-yet-checked
word is present, and a downstream action that depends on the class.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import get_tv, COMPANIES, Fmt, cap, join_list, json_block, json_line, person, pick, pct_row, rand_pct, sample
from .world import Var, World

THEMES = [
    dict(key="email", item="email", classes=["spam", "personal", "promotional"],
         words=["free", "winner", "invoice", "meeting", "unsubscribe", "lunch", "discount", "urgent", "family",
                "offer", "password", "tonight"],
         action=("opened", "not opened"), act_q="Will {who} open this email?",
         act_cl=["{who} will open this email", "{who} will not open this email"],
         act_desc="{who} opens {p} of {c} emails", intro="An email has just arrived in {who}'s inbox.",
         cls_cl="the email is {opt}", cls_frac="{p} of emails are {k}", cls_in="{k} emails"),
    dict(key="ticket", item="support ticket", classes=["billing", "technical", "account"],
         words=["refund", "error", "password", "crash", "charge", "login", "update", "invoice", "slow", "locked"],
         action=("escalated", "not escalated"), act_q="Will this ticket be escalated?",
         act_cl=["the ticket will be escalated", "the ticket will not be escalated"],
         act_desc="{p} of {c} tickets get escalated", intro="A new support ticket has arrived at the {firm} help desk.",
         cls_cl="the ticket is about a {opt} issue", cls_frac="{p} of tickets are about {k} issues",
         cls_in="{k} tickets"),
    dict(key="forum", item="forum post", classes=["spam", "on-topic", "off-topic"],
         words=["buy", "link", "question", "thanks", "crypto", "tutorial", "meme", "cheap", "help", "vacation"],
         action=("removed", "kept"), act_q="Will the moderators remove this post?",
         act_cl=["the moderators will remove this post", "the moderators will keep this post"],
         act_desc="moderators remove {p} of {c} posts", intro="A new post has appeared on the {firm} user forum.",
         cls_cl="the post is {opt}", cls_frac="{p} of posts are {k}", cls_in="{k} posts"),
]


def make_world(rng) -> World:
    fmt = Fmt(rng)
    th = pick(rng, THEMES)
    who = person(rng)
    firm = pick(rng, COMPANIES)
    K = int(rng.integers(2, 4))
    classes = th["classes"][:K]
    item = th["item"]
    T = int(rng.integers(3, 6))
    words = sample(rng, th["words"], T + 1)
    prior_c = pct_row(rng, K, lo=10, step=5, conc=2.0)
    pw = [[rand_pct(rng, 2, 90, 1) for _ in range(K)] for _ in range(T + 1)]  # P(word present | class) in %
    pact = [rand_pct(rng, 5, 95, 5) for _ in range(K)]
    PW = np.array(pw, float) / 100

    Z = np.array(list(itertools.product(range(K), range(2), range(2))))  # class, pending word (0=present), action
    c, w, a = Z[:, 0], Z[:, 1], Z[:, 2]
    pa = np.array(pact, float)[c] / 100
    prior = (np.array(prior_c, float)[c] / 100 * np.where(w == 0, PW[T][c], 1 - PW[T][c]) *
             np.where(a == 0, pa, 1 - pa))
    L = [np.stack([PW[j][c], 1 - PW[j][c]], 1) for j in range(T)]
    pend = words[T]
    intro = th["intro"].format(who=who, firm=firm)

    json_style = rng.random() < 0.3
    if json_style:
        prelude = json_block(rng, {
            "item": item, "note": intro, "P(class)": {k: fmt.p(p) for k, p in zip(classes, prior_c)},
            "P(word appears | class)": {f"'{words[j]}'": {k: fmt.p(pw[j][i]) for i, k in enumerate(classes)}
                                         for j in range(T + 1)},
            f"P({th['action'][0]} | class)": {k: fmt.p(p) for k, p in zip(classes, pact)},
            "model": "naive Bayes: word occurrences independent given the class",
            **({"action_model": f"'{th['action'][0]}' depends only on the class, not on the words"}
               if get_tv() >= 2 else {}),
        })
        render = lambda k, o: json_line({"word": words[o[0]], "present": o[1] == 0})  # noqa: E731
    else:
        s_pri = "Overall, " + join_list([th["cls_frac"].format(p=fmt.p(p), k=k) for k, p in zip(classes, prior_c)]) + "."
        s_w = " ".join(
            f"The word '{words[j]}' appears in " + join_list(
                [f"{fmt.p(pw[j][i])} of " + th["cls_in"].format(k=k) for i, k in enumerate(classes)]) + "."
            for j in range(T + 1))
        s_act = cap(join_list([th["act_desc"].format(who=who, p=fmt.p(p), c=k) for k, p in zip(classes, pact)])) + "."
        s_nb = "Given the class, whether each word appears is independent of the others."
        prelude = pick(rng, [
            f"{intro} {s_pri} {s_w} {s_nb} {s_act} A filter scans the {item} for words one at a time.",
            f"{intro} A naive Bayes filter uses these statistics. {s_pri} {s_w} {s_act} {s_nb}",
            f"{s_pri} {s_w} {s_nb} {s_act} {intro} The scan results come in word by word.",
        ])
        if get_tv() >= 2:
            prelude += (f" Whether the {item} is {th['action'][0]} depends only on its class, not on which words "
                        f"it contains.")
        ev = [lambda wd, pr: f"The {item} {'contains' if pr else 'does not contain'} the word '{wd}'.",
              lambda wd, pr: f"Scan: '{wd}' {'found' if pr else 'not found'}.",
              lambda wd, pr: f"The word '{wd}' is {'present' if pr else 'absent'}."]
        pos = [pick(rng, ev) for _ in range(T)]
        render = lambda k, o: pos[k](words[o[0]], o[1] == 0)  # noqa: E731

    variables = [
        Var("class", classes, [f"What kind of {item} is this?", f"Which class does this {item} belong to?"],
            th["cls_cl"]),
        Var("pending_word", ["present", "absent"], [f"Does the {item} contain the word '{pend}'?",
                                                    f"Will the scan find the word '{pend}' in the {item}?"],
            [f"the {item} contains the word '{pend}'", f"the {item} does not contain the word '{pend}'"]),
        Var("action", list(th["action"]), [th["act_q"].format(who=who)], [x.format(who=who) for x in th["act_cl"]]),
    ]
    meta = {"style": "json" if json_style else "prose", "theme": th["key"]}
    if get_tv() >= 2:  # F10: flag worlds whose stated numbers contradict the real-word labels
        reasons = prior_conflicts(th["key"], classes, words, pw, pact)
        meta["prior_conflict"] = bool(reasons)
        meta["prior_conflict_reasons"] = reasons
    return World(variables, prior, Z, T, alternatives=lambda k, past: [(k, 0), (k, 1)],
                 lik=lambda o, past: L[o[0]][:, o[1]], render=render, prelude=prelude, mart_var="class",
                 exchangeable=True, meta=meta)


# word -> the class a reader would associate it with; action -> the class a reader expects to maximise it
EXPECT = {
    "email": ({"free": "spam", "winner": "spam", "urgent": "spam", "password": "spam", "offer": "promotional",
               "discount": "promotional", "unsubscribe": "promotional", "family": "personal", "lunch": "personal",
               "tonight": "personal", "meeting": "personal"}, "personal"),
    "ticket": ({"refund": "billing", "charge": "billing", "invoice": "billing", "error": "technical",
                "crash": "technical", "slow": "technical", "update": "technical", "password": "account",
                "login": "account", "locked": "account"}, None),
    "forum": ({"buy": "spam", "link": "spam", "crypto": "spam", "cheap": "spam", "question": "on-topic",
               "tutorial": "on-topic", "help": "on-topic", "thanks": "on-topic", "meme": "off-topic",
               "vacation": "off-topic"}, "spam"),
}


def prior_conflicts(theme, classes, words, pw, pact):
    exp_w, exp_a = EXPECT[theme]
    out = []
    for w, row in zip(words, pw):
        c = exp_w.get(w)
        if c in classes and classes[int(max(range(len(row)), key=lambda i: row[i]))] != c:
            out.append(f"'{w}' is not most frequent in {c}")
    if exp_a in classes and classes[int(max(range(len(pact)), key=lambda i: pact[i]))] != exp_a:
        out.append(f"action is not most likely for {exp_a}")
    return out
