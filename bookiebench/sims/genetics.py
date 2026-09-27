"""Validation family: Mendelian inheritance puzzle (one gene, dominant/recessive trait, invented species).

Parents' genotypes are latent (Hardy-Weinberg prior from a stated allele frequency, optionally conditioned on known
parent phenotypes); evidence = phenotypes of offspring (i.i.d. given the parents) and possibly an imperfect genetic
test on a parent. Exchangeable. Variables: sire genotype, dam genotype, next offspring's genotype or trait.
"""
from __future__ import annotations

import itertools

import numpy as np

from .common import Fmt, cap, json_block, json_line, person, pick, rand_pct
from .lexicon import pseudo
from .world import Var, World

DECIMALS = 8
ANIMALS = [("goat", "goats", "kid", "kids"), ("hound", "hounds", "pup", "pups"), ("pony", "ponies", "foal", "foals"),
           ("lizard", "lizards", "hatchling", "hatchlings"), ("rabbit", "rabbits", "kit", "kits"),
           ("finch", "finches", "chick", "chicks"), ("sheep", "sheep", "lamb", "lambs")]
TRAITS = ["velvet ears", "a blue tongue", "striped legs", "curled horns", "a silver mane", "webbed toes",
          "a split tail", "spotted eyelids", "a copper sheen"]
G_LABELS = ["AA", "Aa", "aa"]
# P(child genotype | father g, mother g): each parent passes each of its two copies with prob 1/2
PASS_A = np.array([1.0, 0.5, 0.0])  # P(pass A | genotype)


def child_dist(f, m):
    pa, ma = PASS_A[f], PASS_A[m]
    return np.array([pa * ma, pa * (1 - ma) + (1 - pa) * ma, (1 - pa) * (1 - ma)])


def make_world(rng) -> World:
    fmt = Fmt(rng)
    sp, sps, yo, yos = pick(rng, ANIMALS)
    breed = f"{pseudo(rng)} {sps}"
    breeder = person(rng)
    trait = pick(rng, TRAITS)
    recessive = rng.random() < 0.6
    q = rand_pct(rng, 10, 70, 5)  # frequency of allele a
    qa = q / 100
    hw = np.array([(1 - qa) ** 2, 2 * qa * (1 - qa), qa ** 2])
    shows = np.array([0, 0, 1]) if recessive else np.array([1, 1, 0])  # trait shown by genotype
    sire, dam = f"the {sp} {pseudo(rng, 2)}", f"the {sp} {pseudo(rng, 2)}"
    sire_n, dam_n = sire.split()[-1], dam.split()[-1]
    known = [int(rng.integers(3)) for _ in range(2)]  # 0 unknown, 1 shows trait, 2 does not show
    has_test = rng.random() < 0.4
    test_acc = rand_pct(rng, 80, 98, 1)
    T = int(rng.integers(3, 6))
    child_var = "genotype" if rng.random() < 0.5 else "trait"

    Z = np.array(list(itertools.product(range(3), range(3), range(3))))  # sire, dam, next offspring genotype
    f, m, c = Z[:, 0], Z[:, 1], Z[:, 2]
    prior = hw[f] * hw[m]
    for who, kn in zip((f, m), known):
        if kn == 1:
            prior = prior * shows[who]
        elif kn == 2:
            prior = prior * (1 - shows[who])
    if prior.sum() == 0:
        known = [0, 0]
        prior = hw[f] * hw[m]
    CD = np.array([[child_dist(a, b) for b in range(3)] for a in range(3)])  # f x m x child
    prior = prior * CD[f, m, c]
    p_show = (CD[f, m] * shows).sum(1)  # P(offspring shows trait | parents)
    carrier = (f == 1)  # the test detects a heterozygous carrier in the sire
    steps = ["kid"] * T
    if has_test:
        steps[int(rng.integers(T))] = "test"

    def lik(o, past):
        k, v = o
        if steps[k] == "test":
            a = test_acc / 100
            pos = np.where(carrier, a, 1 - a)
            return pos if v == 0 else 1 - pos
        return p_show if v == 0 else 1 - p_show

    dom_word = "recessive" if recessive else "dominant"
    kn_txt = []
    for nm, kn in ((sire_n, known[0]), (dam_n, known[1])):
        if kn == 1:
            kn_txt.append(f"{nm} shows {trait}.")
        elif kn == 2:
            kn_txt.append(f"{nm} does not show {trait}.")
    json_style = rng.random() < 0.3
    if json_style:
        st = {"breed": breed, "breeder": breeder, "gene": "one gene with alleles A and a",
              "trait": trait, "inheritance": f"{dom_word} ({'aa shows it' if recessive else 'AA and Aa show it'})",
              "P(a random copy is a)": fmt.p(q), "population genotypes": "Hardy-Weinberg (two independent copies)",
              "mendel": "each parent passes one of its two copies at random, independently",
              "sire": sire_n, "dam": dam_n,
              "known": {sire_n: ["unknown", "shows trait", "does not show trait"][known[0]],
                        dam_n: ["unknown", "shows trait", "does not show trait"][known[1]]}}
        if has_test:
            st["carrier test on " + sire_n] = f"reports correctly whether it is Aa with probability {fmt.p(test_acc)}"
        prelude = json_block(rng, st)
    else:
        s1 = (f"{breeder} breeds {breed}. Whether a {sp} has {trait} is controlled by one gene with a dominant allele A "
              f"and a recessive allele a. The trait is {dom_word}: " +
              ("only aa animals show it." if recessive else "AA and Aa animals show it; aa animals do not."))
        s2 = (f"In this breed each of an animal's two gene copies is a with probability {fmt.p(q)}, independently.")
        s3 = (f"Each parent passes one of its two copies to each {yo}, each copy with probability 1/2, independently "
              f"for every {yo}.")
        s4 = f"{cap(sire)} (the sire) and {dam} (the dam) have a litter. " + " ".join(kn_txt)
        s5 = (f" A carrier test on {sire_n} reports correctly whether it is Aa with probability {fmt.p(test_acc)}."
              if has_test else "")
        prelude = pick(rng, [" ".join([s1, s2, s3, s4]) + s5, " ".join([s4, s1, s3, s2]) + s5,
                             " ".join([s1, s4, s2, s3]) + s5])

    def render(k, o):
        _, v = o
        if steps[k] == "test":
            if json_style:
                return json_line({"carrier_test": sire_n, "result": "carrier" if v == 0 else "not carrier"})
            return f"The carrier test on {sire_n} comes back {'positive (Aa)' if v == 0 else 'negative (not Aa)'}."
        if json_style:
            return json_line({yo: "shows trait" if v == 0 else "no trait"})
        return pick_f[k].format(yo=yo, t=trait, neg="" if v == 0 else "not ")

    forms = ["A {yo} is born that does {neg}have {t}.", "Another {yo} is examined: it does {neg}show {t}.",
             "One {yo} in the litter does {neg}have {t}."]
    pick_f = [pick(rng, [forms[0], forms[2]]) for _ in range(T)]

    glab = ["AA", "Aa", "aa"]
    variables = [
        Var("sire", glab, [f"What is {sire_n}'s genotype?", f"Which genotype does the sire {sire_n} have?"],
            f"{sire_n} has genotype {{opt}}"),
        Var("dam", glab, [f"What is {dam_n}'s genotype?", f"Which genotype does the dam {dam_n} have?"],
            f"{dam_n} has genotype {{opt}}"),
    ]
    if child_var == "genotype":
        variables.append(Var("next_offspring", glab, [f"What genotype will the next {yo} have?"],
                             f"the next {yo} will have genotype {{opt}}"))
        proj = Z
    else:
        variables.append(Var("next_offspring", ["shows trait", "no trait"],
                             [f"Will the next {yo} have {trait}?"],
                             [f"the next {yo} will have {trait}", f"the next {yo} will not have {trait}"]))
        proj = np.stack([f, m, 1 - shows[c]], 1)
    return World(variables, prior, proj, T,
                 alternatives=lambda k, past: [(k, 0), (k, 1)], lik=lik, render=render, prelude=prelude,
                 mart_var="sire", exchangeable=True,
                 meta={"style": "json" if json_style else "prose", "recessive": bool(recessive)})
