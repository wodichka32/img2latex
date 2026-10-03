import json
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from metrics import compute_metrics, normalize

RUNS, OUT = Path("runs"), Path("results")
BUCKETS = [(0, 30, "short (<=30)"), (31, 70, "medium (31-70)"), (71, 10**9, "long (>70)")]


def load_preds(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def main():
    OUT.mkdir(exist_ok=True)
    runs = {p.parent.name: load_preds(p) for p in sorted(RUNS.glob("*/test_preds.jsonl"))}
    runs.pop("overfit", None)
    if not runs:
        print("no runs/*/test_preds.jsonl yet")
        return

    rows, bucket_rows = [], []
    for name, recs in runs.items():
        m = compute_metrics([r["pred"] for r in recs], [r["ref"] for r in recs])
        rows.append({"model": name, **m})
        for lo, hi, label in BUCKETS:
            sub = [r for r in recs if lo <= len(normalize(r["ref"])) <= hi]
            if sub:
                bm = compute_metrics([r["pred"] for r in sub], [r["ref"] for r in sub])
                bucket_rows.append({"model": name, "length": label, **bm})

    table = pd.DataFrame(rows).set_index("model").round(4)
    buckets = pd.DataFrame(bucket_rows)
    print(table.to_markdown(), "\n")
    print(buckets.pivot(index="length", columns="model", values="bleu").round(3).to_markdown())
    table.to_csv(OUT / "metrics.csv")
    buckets.to_csv(OUT / "metrics_by_length.csv", index=False)
    (OUT / "metrics.md").write_text(
        "## Test metrics\n\n" + table.to_markdown() +
        "\n\n## BLEU by formula length\n\n" +
        buckets.pivot(index="length", columns="model", values="bleu").round(3).to_markdown() + "\n",
        encoding="utf-8")

    piv = buckets.pivot(index="length", columns="model", values="bleu").reindex([b[2] for b in BUCKETS])
    ax = piv.plot.bar(rot=0, figsize=(7, 4))
    ax.set_ylabel("BLEU"), ax.set_title("BLEU by formula length"), ax.set_ylim(0, 1)
    plt.tight_layout(), plt.savefig(OUT / "bleu_by_length.png", dpi=120), plt.close()

    logs = {p.parent.name: pd.read_csv(p) for p in RUNS.glob("*/log.csv") if p.parent.name in runs}
    if logs:
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
        for name, df in logs.items():
            a1.plot(df.epoch, df.train_loss, label=f"{name} train")
            a1.plot(df.epoch, df.val_loss, "--", label=f"{name} val")
            a2.plot(df.epoch, df.bleu, label=name)
        a1.set_title("loss"), a2.set_title("val BLEU (subset)"), a1.legend(), a2.legend()
        plt.tight_layout(), plt.savefig(OUT / "training_curves.png", dpi=120), plt.close()

    with open(OUT / "errors.md", "w", encoding="utf-8") as f:
        for name, recs in runs.items():
            wrong = [r for r in recs if normalize(r["pred"]) != normalize(r["ref"])]
            f.write(f"## {name}: {len(wrong)}/{len(recs)} wrong\n\n")
            for r in wrong[:15]:
                f.write(f"- #{r['i']}\n  - ref:  `{r['ref']}`\n  - pred: `{r['pred']}`\n")
            f.write("\n")
    print("saved to", OUT)


if __name__ == "__main__":
    main()
