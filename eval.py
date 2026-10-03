import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset, load_splits, make_collate
from metrics import compute_metrics
from model import Im2Latex
from tokenizer import Tokenizer


def load_model(ckpt_path, device="cuda"):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = Im2Latex(**ckpt["cfg"], pretrained=False).to(device).eval()
    model.load_state_dict(ckpt["model"])
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hybrid_tf")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--bs", type=int, default=64)
    a = p.parse_args()

    out = Path("runs") / a.run
    tok = Tokenizer.load("vocab.json")
    model = load_model(out / a.ckpt)
    ds = Im2LatexDataset(load_splits()["test"], tok, limit=a.limit)
    dl = DataLoader(ds, a.bs, num_workers=4, collate_fn=make_collate(tok.pad_id))

    preds, refs = [], []
    for x, _, formulas in tqdm(dl, ncols=100):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out_ids = model.generate(x.cuda())
        preds += [tok.decode(o) for o in out_ids]
        refs += formulas

    m = compute_metrics(preds, refs)
    print(json.dumps(m, indent=2))
    with open(out / "test_preds.jsonl", "w", encoding="utf-8") as f:
        for i, (p_, r) in enumerate(zip(preds, refs)):
            f.write(json.dumps({"i": i, "pred": p_, "ref": r}, ensure_ascii=False) + "\n")
    json.dump(m, open(out / "test_metrics.json", "w"), indent=2)


if __name__ == "__main__":
    main()
