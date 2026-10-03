import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import Im2LatexDataset, load_splits, make_collate
from metrics import compute_metrics
from model import Im2Latex
from tokenizer import Tokenizer


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hybrid_tf")
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--bs", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--warmup", type=int, default=1000)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="use only N train samples")
    p.add_argument("--overfit", action="store_true", help="sanity check: train and eval on the same small subset")
    p.add_argument("--val_gen", type=int, default=500, help="val samples decoded for BLEU")
    p.add_argument("--resume", action="store_true", help="continue from runs/<run>/last.pt")
    p.add_argument("--eval_every", type=int, default=1, help="run validation every N epochs")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, tok, device, max_gen):
    model.eval()
    loss_sum, n_tok, preds, refs = 0.0, 0, [], []
    for x, y, formulas in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x, y[:, :-1])
            loss = F.cross_entropy(logits.float().transpose(1, 2), y[:, 1:], ignore_index=tok.pad_id, reduction="sum")
            loss_sum += loss.item()
            n_tok += y[:, 1:].ne(tok.pad_id).sum().item()
            if len(refs) < max_gen:
                out = model.generate(x)
                preds += [tok.decode(o) for o in out]
                refs += formulas
    m = compute_metrics(preds, refs)
    m["loss"] = loss_sum / n_tok
    return m, list(zip(preds[:3], refs[:3]))


def main():
    a = get_args()
    torch.manual_seed(0)
    device = "cuda"
    out = Path("runs") / a.run
    out.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    tok_path = Path("vocab.json")
    if tok_path.exists():
        tok = Tokenizer.load(tok_path)
    else:
        tok = Tokenizer.build(splits["train"]["formula"])
        tok.save(tok_path)
    print("vocab:", len(tok))

    if a.overfit:
        a.limit = a.limit or 100
        train_ds = Im2LatexDataset(splits["train"], tok, train=False, limit=a.limit)
        val_ds = train_ds
    else:
        train_ds = Im2LatexDataset(splits["train"], tok, train=True, limit=a.limit)
        val_ds = Im2LatexDataset(splits["val"], tok)
    collate = make_collate(tok.pad_id)
    train_dl = DataLoader(train_ds, a.bs, shuffle=True, num_workers=a.workers, collate_fn=collate,
                          pin_memory=True, persistent_workers=a.workers > 0, drop_last=not a.overfit)
    val_dl = DataLoader(val_ds, 64, num_workers=a.workers, collate_fn=collate, persistent_workers=a.workers > 0)

    cfg = dict(vocab_size=len(tok), pad_id=tok.pad_id, bos_id=tok.bos_id, eos_id=tok.eos_id, d_model=a.d_model)
    model = Im2Latex(**cfg).to(device)
    print(f"params: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M, train={len(train_ds)} val={len(val_ds)}")

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
    total = a.epochs * len(train_dl)
    warmup = min(a.warmup, total // 10)

    start_ep, best, mins0 = 1, -1.0, 0.0
    if a.resume and (out / "last.pt").exists():
        ckpt = torch.load(out / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "opt" in ckpt:
            opt.load_state_dict(ckpt["opt"])
        start_ep, mins0 = ckpt["epoch"] + 1, ckpt.get("minutes", 0.0)
        best = torch.load(out / "best.pt", map_location="cpu", weights_only=False)["metrics"]["bleu"]
        print(f"resumed from epoch {ckpt['epoch']}, best bleu {best:.3f}")
    done = (start_ep - 1) * len(train_dl)

    def lr_at(s):
        s += done
        return (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, total - warmup)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    log_f = open(out / "log.csv", "a" if start_ep > 1 else "w", newline="")
    log = csv.writer(log_f)
    if start_ep == 1:
        log.writerow(["epoch", "train_loss", "val_loss", "bleu", "edit_score", "exact_match", "minutes"])
    t0 = time.time() - mins0 * 60
    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run_loss, pbar = 0.0, tqdm(train_dl, desc=f"ep {ep}", ncols=100)
        for i, (x, y, _) in enumerate(pbar, 1):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x, y[:, :-1])
            loss = F.cross_entropy(logits.float().transpose(1, 2), y[:, 1:], ignore_index=tok.pad_id,
                                   label_smoothing=0.0 if a.overfit else 0.1)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            run_loss += loss.item()
            pbar.set_postfix(loss=f"{run_loss / i:.3f}", lr=f"{sched.get_last_lr()[0]:.1e}")

        if ep % a.eval_every and ep != a.epochs:
            continue
        m, ex = evaluate(model, val_dl, tok, device, a.val_gen)
        mins = (time.time() - t0) / 60
        print(f"ep {ep}: val_loss={m['loss']:.3f} bleu={m['bleu']:.3f} edit={m['edit_score']:.3f} "
              f"em={m['exact_match']:.3f} ({mins:.0f} min)")
        for p, r in ex[:2]:
            print("  pred:", p[:120], "\n  ref: ", r[:120])
        log.writerow([ep, run_loss / len(train_dl), m["loss"], m["bleu"], m["edit_score"], m["exact_match"], round(mins, 1)])
        log_f.flush()

        ckpt = {"model": model.state_dict(), "opt": opt.state_dict(), "cfg": cfg, "epoch": ep, "metrics": m,
                "minutes": mins}
        torch.save(ckpt, out / "last.pt")
        if m["bleu"] > best:
            best = m["bleu"]
            torch.save(ckpt, out / "best.pt")
    json.dump(vars(a), open(out / "args.json", "w"), indent=2)
    print("best val bleu:", best)


if __name__ == "__main__":
    main()
