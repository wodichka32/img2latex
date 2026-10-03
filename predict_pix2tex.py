import argparse
import json
import warnings
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from data import DATASET, MAX_TOKENS
from metrics import compute_metrics

warnings.filterwarnings("ignore")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    from pix2tex.cli import LatexOCR
    model = LatexOCR()

    ds = load_dataset(DATASET)["test"]
    keep = [i for i, f in enumerate(ds["formula"]) if len(f.split()) <= MAX_TOKENS]
    if a.limit:
        keep = keep[: a.limit]
    ds = ds.select(keep)

    out = Path("runs") / "pix2tex"
    out.mkdir(parents=True, exist_ok=True)
    path = out / "test_preds.jsonl"
    old = [json.loads(l) for l in open(path, encoding="utf-8")] if path.exists() else []
    preds, refs = [r["pred"] for r in old], [r["ref"] for r in old]
    with open(path, "a", encoding="utf-8") as f:
        for i, ex in enumerate(tqdm(ds, ncols=100)):
            if i < len(old):
                continue
            try:
                pred = model(ex["image"].convert("RGB"))
            except Exception as e:
                pred = ""
                print("fail", i, e)
            preds.append(pred)
            refs.append(ex["formula"])
            f.write(json.dumps({"i": i, "pred": pred, "ref": ex["formula"]}, ensure_ascii=False) + "\n")

    m = compute_metrics(preds, refs)
    print(json.dumps(m, indent=2))
    json.dump(m, open(out / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
