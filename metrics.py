import editdistance
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu

from tokenizer import tokenize_latex

_DROP = {
    "\\,", "\\;", "\\:", "\\!", "\\ ", "~", "\\quad", "\\qquad",
    "\\displaystyle", "\\textstyle", "\\scriptstyle",
}
_MAP = {"\\dfrac": "\\frac", "\\tfrac": "\\frac", "\\le": "\\leq", "\\ge": "\\geq", "\\ne": "\\neq"}


def strip_wrappers(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        s = s.strip("`")
        if s.lower().startswith("latex"):
            s = s[5:]
    for a, b in (("$$", "$$"), ("\\[", "\\]"), ("$", "$"), ("\\(", "\\)")):
        if s.startswith(a) and s.endswith(b) and len(s) >= len(a) + len(b):
            s = s[len(a): len(s) - len(b)]
            break
    return s.strip()


def normalize(s: str) -> list[str]:
    toks = [_MAP.get(t, t) for t in tokenize_latex(strip_wrappers(s))]
    return [t for t in toks if t not in _DROP]


def compute_metrics(preds: list[str], refs: list[str]) -> dict:
    P = [normalize(p) for p in preds]
    R = [normalize(r) for r in refs]
    bleu = corpus_bleu([[r] for r in R], P, smoothing_function=SmoothingFunction().method1)
    ed = sum(editdistance.eval(p, r) for p, r in zip(P, R)) / max(1, sum(max(len(p), len(r)) for p, r in zip(P, R)))
    em = sum(p == r for p, r in zip(P, R)) / max(1, len(R))
    return {"bleu": bleu, "edit_score": 1 - ed, "exact_match": em, "n": len(R)}
