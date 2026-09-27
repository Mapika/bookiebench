"""LLM rewriting transforms (paraphrase, translation, gain/loss framing): prompts, strict programmatic checks, and
assembly. No model code lives here; `llm_run.py` drives a local model (vLLM, in-process) over these jobs.

Transforms
  paraphrase     English rewrite of prelude, every evidence text (realised and alternatives) and every query text.
  lang_<xx>      translation into de / es / hu / pt / zh. Options are translated once per instance (a glossary) and
                 the instance's `variables[*].options` / marginal query `options` are replaced by the translations, so
                 option indices, joints and gold are unchanged.
  framing        loss framing of the prelude: probability statements restated for the complementary outcome
                 (X% have A -> (100-X)% do not have A). Accepted only if a round-trip rewrite back to the positive
                 frame restores exactly the original numbers.

Checks (a segment failing any of them is retried, and the instance is rejected if it still fails):
  * the multiset of number tokens is identical (framing: every number maps to itself or its complement, and a
    round trip restores the original multiset); no new numbers, no extra spelled-out numerals;
  * every named entity (invented names, people, places, companies, labels) is preserved verbatim;
  * every option phrase that occurs in the source occurs in the output (translations: the glossary translation);
  * negation cues are preserved (English: same count; translations: present iff present in the source);
  * quoted terms are kept verbatim, questions stay questions, JSON stays JSON with the same structure;
  * translations must actually be in the target language; length ratio within bounds.
"""
from __future__ import annotations

import copy
import gzip
import json
import re
from pathlib import Path
from collections import Counter
from functools import lru_cache

from bookiebench.sims.common import COMPANIES, FIRST_NAMES, SURNAMES, TOWNS

from .common import evidence_texts, is_json_prelude, replace_evidence_texts, start

LANGS = {"en": "English", "de": "German", "es": "Spanish", "hu": "Hungarian", "pt": "Portuguese (European)",
         "zh": "Simplified Chinese"}
TRANSLATE = ["de", "es", "hu", "pt", "zh"]

WEEKDAYS_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAYS = {
    "de": ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag|Sonnabend", "Sonntag"],
    "es": ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"],
    "pt": ["segunda", "terça", "quarta", "quinta", "sexta", "sábado", "domingo"],
    "hu": ["hétf", "kedd", "szerd", "csütörtök", "péntek", "szombat", "vasárnap"],
    "zh": ["星期一|周一|礼拜一", "星期二|周二|礼拜二", "星期三|周三|礼拜三", "星期四|周四|礼拜四", "星期五|周五|礼拜五",
           "星期六|周六|礼拜六", "星期日|星期天|周日|礼拜天"],
}
NUMWORDS_EN = ["two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve", "twenty",
               "hundred", "thousand", "half", "twice", "thrice", "double", "triple", "dozen", "quarter", "third"]
NUMWORDS = {
    "de": ["zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht", "neun", "zehn", "elf", "zwölf", "zwanzig",
           "hundert", "tausend", "hälfte", "doppelt", "dreifach", "dutzend", "viertel", "drittel"],
    "es": ["dos", "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "diez", "once", "doce", "veinte",
           "cien", "mil", "mitad", "doble", "triple", "docena", "tercio"],
    "pt": ["dois", "duas", "três", "quatro", "cinco", "seis", "sete", "oito", "nove", "dez", "onze", "doze", "vinte",
           "cem", "mil", "metade", "dobro", "triplo", "dúzia", "terço"],
    "hu": ["kettő", "két", "három", "négy", "nyolc", "kilenc", "tíz", "tizenegy", "tizenkét", "húsz", "száz", "ezer",
           "dupla", "kétszer", "tucat", "negyed", "harmad"],
    "zh": [],
}
SPELLED = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
           "ten": 10, "eleven": 11, "twelve": 12}
NEG_EN = (r"\b(not|no|never|neither|nor|none|nobody|nothing|false|untrue|wrong|deny|incorrect|inaccurate|absent|"
          r"lacks?|lacking|without|fails?|failed|missing)\b|n't\b")
NEG = {
    "en": NEG_EN,
    "de": r"\b(nicht|kein\w*|nie|niemals|weder|falsch|unwahr|unzutreffend|unrichtig|ohne|nichts|niemand|fehl\w*|"
          r"abwesend)\b",
    "es": r"\b(no|ni|nunca|jamás|ningun\w*|nada|nadie|falso|falsa|incorrect[oa]|sin|ausente|falta\w*)\b",
    "pt": r"\b(não|nem|nunca|jamais|nenhum\w*|nada|ninguém|falso|falsa|incorret[oa]|sem|ausente|falta\w*)\b",
    "hu": r"\b(nem|sem|ne|nincs\w*|hamis|téves|soha|a?nélkül|senki|semmi\w*|hiány\w*|valótlan\w*)\b",
    "zh": r"不|没|非|无|未|假|错|缺",
}
TITLES = {"Dr", "Mr", "Mrs", "Ms", "Prof"}
_NUM = re.compile(r"\d+(?:[.,]\d+)*")


def number_tokens(text: str) -> list[str]:
    text = text.replace("％", "%")
    return sorted(t.replace(",", ".") for t in _NUM.findall(text))


# ----------------------------------------------------------------------------------------------------------------------
# vocabulary for telling invented names from ordinary capitalised words
# ----------------------------------------------------------------------------------------------------------------------

COMMON_WORDS = Path(__file__).with_name("common_words.txt.gz")


@lru_cache(maxsize=1)
def common_words() -> frozenset:
    """Lower-case words seen >= 3 times in lower case in 60k CNN/DailyMail training articles (ordinary English words;
    built once from the local HF copy and shipped as common_words.txt.gz)."""
    with gzip.open(COMMON_WORDS, "rt", encoding="utf-8") as f:
        return frozenset(w.strip() for w in f if w.strip())


def required_entities(text: str, exempt_words=()) -> list[str]:
    """Tokens that must survive verbatim: known names, and capitalised tokens that are not ordinary English words."""
    common = common_words()
    exempt = {w.lower() for w in exempt_words}
    out = set()
    for tok in re.findall(r"\b[A-Z][\w-]*", text):
        base = tok[:-2] if tok.endswith("'s") else tok
        low = base.lower()
        if base in TITLES or base in WEEKDAYS_EN or low in exempt:
            continue
        if base in FIRST_NAMES or base in SURNAMES or base in COMPANIES:
            out.add(base)
        elif low not in common and low.rstrip("s") not in common and not (len(base) > 1 and base.isupper()):
            out.add(base.split("-")[0] + "-" if "-" in base and len(base.split("-")[0]) <= 2 else base)
    for t in TOWNS + COMPANIES:
        if " " in t and t in text:
            out.add(t)
    return sorted(out)


def quoted_terms(text: str) -> list[str]:
    # a real quote opens after a non-letter and closes before a non-letter (so possessives like "Jin's" never count)
    return sorted(set(re.findall(r"(?<![A-Za-z])'([A-Za-z][\w -]{0,30}?)'(?![A-Za-z])", text)))


# ----------------------------------------------------------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------------------------------------------------------

def _has_word(text, phrase):
    return re.search(r"(?<![\w])" + re.escape(phrase.lower()) + r"(?![\w])", text.lower()) is not None


def _fuzzy_contains(text, phrase, lang):
    """Inflection-tolerant containment of a (translated) phrase: each word of >= 3 letters must match a word of the
    text on a common prefix (zh: plain substring)."""
    if lang == "zh" or not re.search(r"[A-Za-zÀ-ž]", phrase):
        return phrase.lower() in text.lower() or phrase.replace(" ", "") in text.replace(" ", "")
    words = re.findall(r"[\wÀ-ž'-]+", text.lower())
    for w in re.findall(r"[\wÀ-ž'-]+", phrase.lower()):
        if len(w) < 3:
            continue
        k = min(len(w), max(3, len(w) - 2))
        if not any(x.startswith(w[:k]) or w.startswith(x[:k]) and len(x) >= k for x in words):
            return False
    return True


_LONG = {"a": "á", "e": "é", "o": "ó", "ö": "ő", "u": "ú", "ü": "ű"}


def _entity_in(e, tgt, lang):
    if e in tgt:
        return True
    if lang == "hu" and e[-1:] in _LONG and (e[:-1] + _LONG[e[-1]]) in tgt:
        return True
    return lang == "zh" and e.replace(" ", "") in tgt.replace(" ", "")


def _count(pattern, text):
    return len(re.findall(pattern, text, flags=re.I))


def _lang_ok(tgt, lang):
    letters = re.findall(r"[^\W\d_]", tgt)
    if not letters:
        return True
    if lang == "zh":
        cjk = re.findall(r"[一-鿿]", tgt)
        return len(cjk) >= 0.3 * len(letters)
    words = [w.lower() for w in re.findall(r"[A-Za-zÀ-ž]+", tgt)]
    if len(words) < 6:
        return True
    eng = {"the", "and", "of", "that", "with", "have", "are", "this", "what", "how", "likely", "probability", "which",
           "there", "their", "been", "would", "could"}
    return sum(w in eng for w in words) <= 0.15 * len(words)


def _json_shape(o):
    if isinstance(o, dict):
        return ("d", len(o), tuple(_json_shape(v) for v in o.values()))
    if isinstance(o, list):
        return ("l", len(o), tuple(_json_shape(v) for v in o))
    return type(o).__name__


def _as_json(t):
    t = t.strip()
    if not t.startswith(("{", "[")):
        j = t.find("{")
        if j < 0:
            return None
        t = t[j:]
    try:
        return json.loads(t)
    except ValueError:
        return None


def check_text(src: str, tgt: str, lang: str, ctx: dict, is_question: bool = False) -> list[str]:
    """Reasons why `tgt` is not an acceptable rewrite/translation of `src` (empty list = accepted).

    ctx: options (all source option strings), opt_map (source option -> translated option; translations only),
    exempt (words exempt from the verbatim-entity rule, e.g. words of options that get translated)."""
    bad = []
    if not tgt or not tgt.strip():
        return ["empty"]
    ns_, nt_ = Counter(number_tokens(src)), Counter(number_tokens(tgt))
    spelled = Counter(str(v) for w, v in SPELLED.items() for _ in range(_count(rf"\b{w}\b", src)))
    if ns_ - nt_ or (nt_ - ns_) - spelled:  # every number kept; extra digits only for spelled-out source numerals
        bad.append("numbers")
    en_nw = sum(_count(rf"\b{w}\b", src) for w in NUMWORDS_EN + ["both", "pair", "couple"])
    tgt_nw = re.sub(r"\d+[-‐]\w+", " ", tgt)
    if lang == "en":
        if any(_count(rf"\b{w}\b", tgt) > _count(rf"\b{w}\b", src) for w in NUMWORDS_EN):
            bad.append("number_words")
    elif NUMWORDS.get(lang):
        tw = sum(_count(rf"\b{w}\b", tgt_nw) for w in NUMWORDS[lang])
        opt_nw = sum(_count(rf"\b{w}\b", " ".join(ctx.get("opt_map", {}).values())) for w in NUMWORDS[lang])
        if tw > en_nw + opt_nw:
            bad.append("number_words")
    exempt = set(ctx.get("exempt", ())) if lang != "en" else set()
    for e in required_entities(src, exempt):
        if not _entity_in(e, tgt, lang):
            bad.append(f"entity:{e}")
            break
    for qt in quoted_terms(src):
        if qt not in tgt:
            bad.append(f"quoted:{qt}")
            break
    for o in ctx.get("options", []):
        if len(o) <= 2 or not _has_word(src, o) or not required_entities(o):
            continue
        if lang == "en":
            if not _has_word(tgt, o):
                bad.append(f"option:{o}")
                break
        else:
            tr = ctx.get("opt_map", {}).get(o)
            if tr is None or not _fuzzy_contains(tgt, tr, lang):
                bad.append(f"option:{o}")
                break
    if lang != "en":
        for i, d in enumerate(WEEKDAYS_EN):
            if re.search(rf"\b{d}\b", src) and not re.search(WEEKDAYS[lang][i], tgt, flags=re.I):
                bad.append(f"weekday:{d}")
                break
    js = _as_json(src) if src.lstrip().startswith("{") or is_json_prelude(src) else None
    s_neg, t_neg = src, re.sub(r"是否|假设|假如|假定", "", tgt)
    if js is not None:  # JSON literals and (untranslated) keys carry no negation
        s_neg = re.sub(r"\b(true|false|null)\b|\"[^\"]*\"\s*:", " ", s_neg)
        t_neg = re.sub(r"\b(true|false|null)\b|\"[^\"]*\"\s*:", " ", t_neg)
    ns, nt = _count(NEG_EN, s_neg), _count(NEG[lang], t_neg)
    if lang == "de":
        nt += _count(r"\bweder\b", t_neg)  # weder ... noch
    if lang == "en":
        if ns != nt:
            bad.append("negation")
    elif (ns > 0) != (nt > 0) or (is_question and ns >= 2 and nt < 2):
        bad.append("negation")
    if js is not None:
        jt = _as_json(tgt)
        if jt is None or _json_shape(js) != _json_shape(jt):
            bad.append("json")
    if is_question and not tgt.strip().endswith(("?", "？")):
        bad.append("question")
    ls, lt = len(src), len(tgt)
    lo, hi = {"en": (0.5, 2.2), "zh": (0.12, 1.5)}.get(lang, (0.5, 3.0))
    if not lo * ls <= lt <= hi * ls + 20:
        bad.append("length")
    if lang != "en" and js is None and not _lang_ok(tgt, lang):
        bad.append("not_translated")
    if lang == "en" and ctx.get("need_change") and tgt.strip() == src.strip() and len(src) > 40:
        bad.append("unchanged")
    return bad


# ----------------------------------------------------------------------------------------------------------------------
# framing checks
# ----------------------------------------------------------------------------------------------------------------------

_PNUM = re.compile(r"(\d+(?:\.\d+)?)(\s?%|\s?percent\b)?")


def prob_numbers(text):
    """[(value, kind)] with kind 'pct' (N% / N percent), 'dec' (0.xx) or 'cnt' (anything else)."""
    out = []
    for m in _PNUM.finditer(text):
        v, unit = m.group(1), m.group(2)
        if unit:
            out.append((float(v), "pct"))
        elif v.startswith("0.") or v == "0" or v == "1":
            out.append((float(v), "dec"))
        else:
            out.append((float(v), "cnt"))
    return out


def complement_match(src: str, tgt: str):
    """(ok, n_complemented): can the numbers of tgt be matched one-to-one to those of src with each equal to the source
    number or (for probabilities) its complement?"""
    from scipy.optimize import linear_sum_assignment
    import numpy as np
    S, T = prob_numbers(src), prob_numbers(tgt)
    if len(S) != len(T):
        return False, 0
    if not S:
        return True, 0
    C = np.full((len(S), len(T)), 10.0)
    for i, (sv, sk) in enumerate(S):
        for j, (tv, tk) in enumerate(T):
            if tk != sk:
                continue
            if abs(tv - sv) < 1e-9:
                C[i, j] = 0.001 if (sk == "pct" and sv == 50) or (sk == "dec" and sv == 0.5) else 0.0
            elif sk == "pct" and abs(tv - (100 - sv)) < 1e-9:
                C[i, j] = 0.001
            elif sk == "dec" and abs(tv - (1 - sv)) < 1e-9:
                C[i, j] = 0.001
    r, c = linear_sum_assignment(C)
    cost = C[r, c]
    if np.any(cost >= 10):
        return False, 0
    return True, int(np.sum(np.isclose(cost, 0.001)))


def frame_eligible(inst) -> bool:
    pre = inst["prelude"]
    if is_json_prelude(pre):
        return False
    return sum(k in ("pct", "dec") for _, k in prob_numbers(pre)) >= 2


# ----------------------------------------------------------------------------------------------------------------------
# prompts
# ----------------------------------------------------------------------------------------------------------------------

SYSTEM = "You are a meticulous editor. You follow formatting and preservation rules exactly."

RULES_EN = """Rules:
- Keep the meaning exactly. Do not add, drop or change any fact.
- Keep every number exactly as written (same digits, same % sign or the word "percent"). Do not add any number, quantity or numeral word, and do not spell numbers out.
- Keep all names of people, places, companies, invented terms and labels (such as "Tin A") exactly as written.
- Keep these option phrases word for word wherever they occur: {options}
- Keep every negation (not, no, neither, false, ...) with the same logical effect; do not merge or cancel negations.
- Keep quoted words in quotes unchanged."""

RULES_TR = """Rules:
- Translate faithfully. Do not add, drop or change any fact.
- Keep every number exactly as written with ASCII digits (you may write % or the target-language word for percent). Keep the decimal point as a point. Do not add any number or numeral word.
- Keep all names of people, places, companies, invented terms and labels (such as "Tin A") exactly as written, in Latin script (never transliterate a name: write "Clara", not a transliteration).
- Translate the following option phrases exactly as given in this glossary, wherever they occur (inflect only if grammar requires it): {glossary}
- Keep every negation with the same logical effect; do not merge or cancel negations.
- Keep words in single quotes unchanged (do not translate them)."""


def _opt_list(options):
    return "; ".join(f'"{o}"' for o in options) if options else "(none)"


def prompt_text(text, lang, options, glossary=None, kind="prelude"):
    js = kind == "prelude" and is_json_prelude(text)
    if lang == "en":
        task = ("Paraphrase the text below: use substantially different wording and sentence structure." +
                (" It contains JSON: keep it valid JSON with the same structure, the same keys in the same order and "
                 "the same number of entries; rewrite only the prose in string values and any text before the JSON."
                 if js else ""))
        rules = RULES_EN.format(options=_opt_list(options))
    else:
        task = (f"Translate the text below into {LANGS[lang]}." +
                (" It contains JSON: keep it valid JSON with the same structure, the same keys (do not translate "
                 "keys) and the same number of entries; translate the prose in string values and any text before the "
                 "JSON; keep true/false/null and numbers as they are." if js else ""))
        rules = RULES_TR.format(glossary=json.dumps(glossary or {}, ensure_ascii=False))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task}\n\n{rules}\n\nOutput only the result, nothing else.\n\nText:\n{text}"}]


def prompt_list(texts, lang, options, glossary=None, what="sentences", questions=False):
    n = len(texts)
    if lang == "en":
        task = (f"Paraphrase each of the following {n} {what}: use different wording while keeping the meaning "
                f"exactly." + (" Each must remain a question ending with '?'." if questions else "") +
                " Items that are JSON objects must stay valid JSON with the same keys; paraphrase only their prose "
                "values.")
        rules = RULES_EN.format(options=_opt_list(options))
    else:
        task = (f"Translate each of the following {n} {what} into {LANGS[lang]}." +
                (" Each must remain a question ending with '?'." if questions else "") +
                " Items that are JSON objects must stay valid JSON with the same keys (do not translate keys) and the "
                "same value types; translate only prose string values, keep names, true/false and numbers.")
        rules = RULES_TR.format(glossary=json.dumps(glossary or {}, ensure_ascii=False))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{task}\n\n{rules}\n\nReturn a JSON array of exactly {n} strings, the i-th "
                                        f"being the result for the i-th input, and nothing else.\n\nInputs (JSON "
                                        f"array):\n{json.dumps(texts, ensure_ascii=False)}"}]


def prompt_options(options_by_var, lang, context):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                f"Below are the answer options of some questions about this situation:\n\n{context[:1500]}\n\n"
                f"Translate every option into {LANGS[lang]} as a short phrase that fits the situation. Keep names, "
                f"invented terms, labels such as \"Tin A\" and digits unchanged. Options of the same variable must stay "
                f"distinct. Return a JSON object with the same keys, each mapped to a list of the translated options "
                f"in the same order, and nothing else.\n\n{json.dumps(options_by_var, ensure_ascii=False)}"}]


def prompt_frame(text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "Rewrite the text below in a LOSS frame: every probability or percentage it states for a yes/no "
                "outcome must be restated as the probability of the complementary outcome (the outcome NOT "
                "happening), e.g. \"35% of patients have a cough\" -> \"65% of patients do not have a cough\", "
                "\"with probability 0.2 it rains\" -> \"with probability 0.8 it does not rain\", \"it is positive for "
                "95% of patients with A and 10% with B\" -> \"it is negative for 5% of patients with A and 90% with "
                "B\". Complement every probability in the text, not just some. If the text instead splits the "
                "probability among three or more mutually exclusive alternatives (e.g. \"10% have A, 75% have B and "
                "15% have neither\"), return it unchanged. Keep counts, times and all names exactly, and do not add "
                "anything.\n\nOutput only the rewritten text.\n\nText:\n" + text}]


def prompt_unframe(text):
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content":
                "The text below states some probabilities for outcomes NOT happening (e.g. \"65% do not have a "
                "cough\"). Rewrite it so that every such statement gives the probability of the outcome happening "
                "instead (\"35% have a cough\"). Leave statements that are already positive unchanged. Keep all other "
                "numbers and names exactly.\n\nOutput only the rewritten text.\n\nText:\n" + text}]


# ----------------------------------------------------------------------------------------------------------------------
# parsing
# ----------------------------------------------------------------------------------------------------------------------

def clean_output(s: str) -> str:
    s = re.sub(r"<think>.*?</think>", "", s, flags=re.S).strip()
    m = re.match(r"^```[a-zA-Z]*\n(.*)\n```$", s, flags=re.S)
    return (m.group(1) if m else s).strip()


def parse_json(s: str):
    s = clean_output(s)
    try:
        return json.loads(s)
    except ValueError:
        pass
    for o, c in (("[", "]"), ("{", "}")):
        i, j = s.find(o), s.rfind(c)
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except ValueError:
                continue
    return None


def parse_list(s: str, n: int):
    v = parse_json(s)
    if isinstance(v, list) and len(v) == n and all(isinstance(x, (str, dict)) for x in v):
        return [x if isinstance(x, str) else json.dumps(x, ensure_ascii=False) for x in v]
    return None


def check_options(src_opts: dict, out, lang) -> list[str]:
    if not isinstance(out, dict) or set(out) != set(src_opts):
        return ["options_format"]
    bad = []
    for k, so in src_opts.items():
        to = out[k]
        if not isinstance(to, list) or len(to) != len(so) or not all(isinstance(x, str) and x.strip() for x in to):
            return ["options_format"]
        if len({x.strip().lower() for x in to}) != len(to):
            bad.append("options_not_distinct")
        for a, b in zip(so, to):
            if number_tokens(a) != number_tokens(b):
                bad.append("options_numbers")
            for e in required_entities(a):
                if e not in b and not (len(e) > 1 and e.isupper()):
                    bad.append(f"options_entity:{e}")
    return sorted(set(bad))


# ----------------------------------------------------------------------------------------------------------------------
# segments / assembly
# ----------------------------------------------------------------------------------------------------------------------

def all_options(inst):
    return sorted({o for v in inst["variables"] for o in v["options"]}, key=len, reverse=True)


def options_by_var(inst):
    return {v["name"]: list(v["options"]) for v in inst["variables"]}


def glossary(inst, opt_tr: dict) -> dict:
    """{source option: translated option} (for prompts and checks)."""
    g = {}
    for v in inst["variables"]:
        for a, b in zip(v["options"], opt_tr[v["name"]]):
            g[a] = b
    return g


def option_exempt_words(inst):
    return {w for o in all_options(inst) for w in re.findall(r"[A-Za-z][\w-]*", o)}


def assemble(inst, transform, prelude, ev_map, qtexts, opt_tr=None, info=None):
    """Build the rewritten instance. ev_map: {source evidence text: new text}; qtexts: new query texts in order."""
    out = start(inst, transform, **(info or {}))
    out["prelude"] = prelude
    replace_evidence_texts(out, ev_map)
    for q, t in zip(out["queries"], qtexts):
        t = t.strip()
        if t.endswith("？"):
            t = t[:-1] + "?"
        q["text"] = t
    if opt_tr is not None:
        for v in out["variables"]:
            v["options"] = [o.strip() for o in opt_tr[v["name"]]]
        by = {v["name"]: v["options"] for v in out["variables"]}
        for q in out["queries"]:
            if q["kind"] == "marginal":
                q["options"] = list(by[q["var"]])
        out["meta"]["stress"]["option_map"] = {v["name"]: dict(zip(sv["options"], v["options"]))
                                               for v, sv in zip(out["variables"], inst["variables"])}
    return out


__all__ = ["LANGS", "TRANSLATE", "check_text", "check_options", "complement_match", "frame_eligible", "prompt_text",
           "prompt_list", "prompt_options", "prompt_frame", "prompt_unframe", "parse_list", "parse_json",
           "clean_output", "assemble", "glossary", "all_options", "options_by_var", "evidence_texts",
           "option_exempt_words", "copy"]


# --- zh double negation -------------------------------------------------------------------------------------------
# Observed in judge-clean zh items: "Is it untrue that X?" (X positive) rendered as "X不…，这是否不属实?" ("not-X, is that
# untrue?"), which flips the queried event. Neither the programmatic checks nor the LLM judge catch it reliably, so
# finalize drops zh items with such a query.
_NEG_Q = re.compile(r"^(Is it untrue that|Is it false that|Is it not the case that) (.*)\?$")
_EN_INNER_NEG = re.compile(r"\b(not|no|neither|nor|none|never|stayed silent|stayed quiet|absent)\b")
_ZH_NEG_TAIL = re.compile(r"[，,]?\s*(这|这一说法|这种说法|这说法|这种情况)?(是否|是)(不|错误|假)")
_ZH_NEG = re.compile(r"不|未(?!来)|没|无(?!论)|非")


def zh_double_negation(src_q: str, tgt_q: str) -> bool:
    m = _NEG_Q.match(src_q.strip())
    if not m or _EN_INNER_NEG.search(m.group(2)):
        return False
    t = tgt_q.strip()
    if t.startswith(("是否并非", "并非", "是否不是")):
        return False
    mt = _ZH_NEG_TAIL.search(t)
    return bool(mt and mt.start() > 0 and _ZH_NEG.search(t[:mt.start()]))
