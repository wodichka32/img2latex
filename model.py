import math

import timm
import torch
import torch.nn as nn


def sincos_1d(n, d, device):
    pos = torch.arange(n, device=device, dtype=torch.float32)[:, None]
    div = torch.exp(torch.arange(0, d, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(n, d, device=device)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sincos_2d(h, w, d, device):
    ph = sincos_1d(h, d // 2, device)[:, None, :].expand(h, w, d // 2)
    pw = sincos_1d(w, d // 2, device)[None, :, :].expand(h, w, d // 2)
    return torch.cat([ph, pw], dim=-1).reshape(h * w, d)


class HybridEncoder(nn.Module):
    def __init__(self, d_model=256, layers=4, heads=8, backbone="resnet34", pretrained=True, dropout=0.1):
        super().__init__()
        self.cnn = timm.create_model(backbone, pretrained=pretrained, features_only=True, out_indices=(3,), in_chans=1)
        self.proj = nn.Conv2d(self.cnn.feature_info.channels()[-1], d_model, 1)
        layer = nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        f = self.proj(self.cnn(x)[-1])
        B, D, H, W = f.shape
        f = f.flatten(2).transpose(1, 2) + sincos_2d(H, W, D, f.device)
        return self.norm(self.tf(f))


class TransformerDecoder(nn.Module):
    def __init__(self, vocab_size, d_model=256, layers=4, heads=8, max_len=256, dropout=0.1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, std=d_model ** -0.5)
        self.register_buffer("pe", sincos_1d(max_len, d_model, "cpu"), persistent=False)
        layer = nn.TransformerDecoderLayer(d_model, heads, 4 * d_model, dropout, batch_first=True, norm_first=True)
        self.tf = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d_model)
        self.out = nn.Linear(d_model, vocab_size)
        self.out.weight = self.emb.weight
        self.scale = math.sqrt(d_model)

    def forward(self, y, memory, pad_mask=None):
        L = y.size(1)
        h = self.emb(y) * self.scale + self.pe[:L]
        causal = torch.ones(L, L, dtype=torch.bool, device=y.device).triu(1)
        h = self.tf(h, memory, tgt_mask=causal, tgt_key_padding_mask=pad_mask, tgt_is_causal=True)
        return self.out(self.norm(h))


class Im2Latex(nn.Module):
    def __init__(self, vocab_size, pad_id, bos_id, eos_id, d_model=256, enc_layers=4, dec_layers=4,
                 heads=8, max_len=256, pretrained=True):
        super().__init__()
        self.pad_id, self.bos_id, self.eos_id, self.max_len = pad_id, bos_id, eos_id, max_len
        self.encoder = HybridEncoder(d_model, enc_layers, heads, pretrained=pretrained)
        self.decoder = TransformerDecoder(vocab_size, d_model, dec_layers, heads, max_len)

    def forward(self, x, y_in):
        return self.decoder(y_in, self.encoder(x), pad_mask=y_in.eq(self.pad_id))

    @torch.no_grad()
    def generate(self, x, max_len=None):
        max_len = min(max_len or self.max_len, self.max_len)
        mem = self.encoder(x)
        y = torch.full((x.size(0), 1), self.bos_id, dtype=torch.long, device=x.device)
        done = torch.zeros(x.size(0), dtype=torch.bool, device=x.device)
        for _ in range(max_len - 1):
            nxt = self.decoder(y, mem)[:, -1].argmax(-1)
            nxt = nxt.masked_fill(done, self.pad_id)
            y = torch.cat([y, nxt[:, None]], 1)
            done |= nxt.eq(self.eos_id)
            if done.all():
                break
        return y[:, 1:]
