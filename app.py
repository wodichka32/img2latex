import argparse

import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from data import preprocess
from eval import load_model
from tokenizer import Tokenizer


def clean_photo(img: Image.Image) -> Image.Image:
    g = ImageOps.autocontrast(img.convert("L"), cutoff=1)
    a = np.array(g)
    thr = min(200, int(a.mean() - a.std() * 0.5))
    a = np.where(a > thr, 255, a).astype(np.uint8)
    return Image.fromarray(a)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="hybrid_tf")
    p.add_argument("--ckpt", default="best.pt")
    a = p.parse_args()

    tok = Tokenizer.load("vocab.json")
    model = load_model(f"runs/{a.run}/{a.ckpt}")

    def predict(img, is_photo):
        if img is None:
            return "", ""
        if is_photo:
            img = clean_photo(img)
        x = preprocess(img)[None].cuda()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            ids = model.generate(x)[0]
        latex = tok.decode(ids)
        return latex, f"$$ {latex} $$"

    with gr.Blocks(title="img2latex") as demo:
        gr.Markdown("## Image → LaTeX")
        with gr.Row():
            with gr.Column():
                inp = gr.Image(type="pil", label="Formula image")
                photo = gr.Checkbox(label="Photo (clean background)", value=False)
                btn = gr.Button("Convert", variant="primary")
            with gr.Column():
                code = gr.Textbox(label="LaTeX", lines=4, buttons=["copy"])
                render = gr.Markdown(latex_delimiters=[{"left": "$$", "right": "$$", "display": True}])
        btn.click(predict, [inp, photo], [code, render])
    demo.launch()


if __name__ == "__main__":
    main()
