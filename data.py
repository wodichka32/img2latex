import torch
import torchvision.transforms.v2 as T
from datasets import load_dataset
from PIL import Image, ImageOps
from torch.utils.data import Dataset

DATASET = "yuntian-deng/im2latex-100k"
IMG_H, IMG_W = 128, 640
MAX_TOKENS = 150


def load_splits():
    return load_dataset(DATASET)


def preprocess(img: Image.Image) -> torch.Tensor:
    img = img.convert("L")
    bbox = ImageOps.invert(img).getbbox()
    if bbox:
        img = img.crop(bbox)
    pad = 8
    scale = min((IMG_H - 2 * pad) / img.height, (IMG_W - 2 * pad) / img.width, 2.0)
    img = img.resize((max(1, round(img.width * scale)), max(1, round(img.height * scale))), Image.BILINEAR)
    canvas = Image.new("L", (IMG_W, IMG_H), 255)
    canvas.paste(img, (pad, pad))
    x = T.functional.pil_to_tensor(canvas).float() / 255.0
    return (x - 0.5) / 0.5


class Im2LatexDataset(Dataset):
    def __init__(self, hf_split, tokenizer, train=False, limit=None):
        keep = [i for i, f in enumerate(hf_split["formula"]) if len(f.split()) <= MAX_TOKENS]
        if limit:
            keep = keep[:limit]
        self.ds = hf_split.select(keep)
        self.tok = tokenizer
        self.aug = T.RandomAffine(degrees=1, translate=(0.02, 0.02), scale=(0.9, 1.05), fill=1.0) if train else None

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        x = preprocess(ex["image"])
        if self.aug is not None:
            x = self.aug(x)
        return x, torch.tensor(self.tok.encode(ex["formula"])), ex["formula"]


class make_collate:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        xs, ys, fs = zip(*batch)
        y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=self.pad_id)
        return torch.stack(xs), y, list(fs)
