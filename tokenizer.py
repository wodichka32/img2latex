import json
import re
from collections import Counter

PAD, BOS, EOS, UNK = "<pad>", "<bos>", "<eos>", "<unk>"
SPECIALS = [PAD, BOS, EOS, UNK]

_LATEX_RE = re.compile(r"\\[a-zA-Z]+\*?|\\.|\S")


def tokenize_latex(s: str) -> list[str]:
    return _LATEX_RE.findall(s)


class Tokenizer:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {t: i for i, t in enumerate(itos)}
        self.pad_id, self.bos_id, self.eos_id, self.unk_id = (self.stoi[t] for t in SPECIALS)

    def __len__(self):
        return len(self.itos)

    @classmethod
    def build(cls, formulas, min_freq: int = 2):
        cnt = Counter(t for f in formulas for t in tokenize_latex(f))
        return cls(SPECIALS + sorted(t for t, c in cnt.items() if c >= min_freq))

    def encode(self, formula: str) -> list[int]:
        ids = [self.stoi.get(t, self.unk_id) for t in tokenize_latex(formula)]
        return [self.bos_id] + ids + [self.eos_id]

    def decode(self, ids) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id:
                break
            if i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i])
        return " ".join(out)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.itos, f, ensure_ascii=False, indent=0)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as f:
            return cls(json.load(f))
