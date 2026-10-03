import argparse

import torch
from PIL import Image

from app import clean_photo
from data import preprocess
from eval import load_model
from tokenizer import Tokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("images", nargs="+")
    p.add_argument("--run", default="hybrid_tf")
    p.add_argument("--ckpt", default="best.pt")
    p.add_argument("--photo", action="store_true", help="clean grey/uneven background first")
    a = p.parse_args()

    tok = Tokenizer.load("vocab.json")
    model = load_model(f"runs/{a.run}/{a.ckpt}")
    for path in a.images:
        img = Image.open(path)
        if a.photo:
            img = clean_photo(img)
        x = preprocess(img)[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(x)[0]
        print(f"{path}\n{tok.decode(ids)}\n")


if __name__ == "__main__":
    main()
