#!/usr/bin/env python3
"""Precompute GPT-2 (small) attention maps for the Transformer demos and slides.

Runs a minimal re-implementation of GPT-2 small in PyTorch on a few short
sentences and on a random token sequence repeated twice, and writes:

- html/data/gpt2_attention.json, used by html/components/GPT2Heads.vue and
  InductionHeads.vue: tokens, the causal attention weights of all 12x12 heads
  (lower triangle, row by row, in percent), three per-head scores (previous
  token, first token, induction) and the per-token loss on the repeated
  sequence;
- courses/nnds/assets/images/transformers/gpt2_heads.tex, the TikZ cells of
  three heads drawn on the slide "Attention in a trained model".

Weights: the public safetensors file of openai-community/gpt2, cached in
~/.cache/gpt2 (548 MB, read with a small header parser, no extra package).
Run with the repository venv plus tiktoken:

    .venv/bin/python -m pip install tiktoken
    .venv/bin/python scripts/make_gpt2_attention.py
"""
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
import torch

ROOT = Path(__file__).resolve().parents[1]
OUT_JSON = ROOT / 'html' / 'data' / 'gpt2_attention.json'
OUT_TEX = ROOT / 'courses' / 'nnds' / 'assets' / 'images' / 'transformers' / 'gpt2_heads.tex'
URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors'
CACHE = Path.home() / '.cache' / 'gpt2' / 'model.safetensors'
SENTENCES = [
    'The cat sat on the mat because it was tired.',
    'When Mary and John went to the store, John gave a drink to',
    'The keys to the cabinet are on the table.',
    'def add(a, b):\n    return a + b',
]
REPEAT_LEN, SEED = 15, 0
N_LAYERS, N_HEADS, D = 12, 12, 768
DTYPES = {'F32': np.float32}


def load_weights():
    if not CACHE.exists():
        CACHE.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, CACHE)
    raw = CACHE.read_bytes()
    n = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    w = {}
    for name, info in header.items():
        if name == '__metadata__' or name.endswith('.attn.bias') and 'c_attn' not in name:
            continue
        a, b = info['data_offsets']
        arr = np.frombuffer(raw[8 + n + a:8 + n + b], dtype=DTYPES[info['dtype']])
        w[name] = torch.from_numpy(arr.reshape(info['shape']).copy())
    return w


def layer_norm(x, w, prefix):
    return torch.nn.functional.layer_norm(x, (D,), w[prefix + '.weight'], w[prefix + '.bias'])


def forward(w, ids):
    """Returns logits (T, V) and attention weights (layers, heads, T, T)."""
    T = len(ids)
    x = w['wte.weight'][ids] + w['wpe.weight'][:T]
    mask = torch.full((T, T), float('-inf')).triu(1)
    attns = []
    for l in range(N_LAYERS):
        p = f'h.{l}'
        h = layer_norm(x, w, p + '.ln_1')
        qkv = h @ w[p + '.attn.c_attn.weight'] + w[p + '.attn.c_attn.bias']
        q, k, v = qkv.split(D, -1)
        q, k, v = (t.reshape(T, N_HEADS, D // N_HEADS).transpose(0, 1) for t in (q, k, v))
        a = (q @ k.transpose(-2, -1) / (D // N_HEADS) ** 0.5 + mask).softmax(-1)
        attns.append(a)
        o = (a @ v).transpose(0, 1).reshape(T, D)
        x = x + o @ w[p + '.attn.c_proj.weight'] + w[p + '.attn.c_proj.bias']
        h = layer_norm(x, w, p + '.ln_2')
        h = torch.nn.functional.gelu(h @ w[p + '.mlp.c_fc.weight'] + w[p + '.mlp.c_fc.bias'], approximate='tanh')
        x = x + h @ w[p + '.mlp.c_proj.weight'] + w[p + '.mlp.c_proj.bias']
    x = layer_norm(x, w, 'ln_f')
    return x @ w['wte.weight'].T, torch.stack(attns)


def lower_triangle(a):
    """(L, H, T, T) attention -> nested lists of percentages, causal part only."""
    T = a.shape[-1]
    rows = [(a[..., t, :t + 1] * 100).round().int() for t in range(T)]
    return [[sum((r[l, h].tolist() for r in rows), []) for h in range(N_HEADS)] for l in range(N_LAYERS)]


def label(enc, i):
    return '⟨bos⟩' if i == enc.eot_token else enc.decode([i])


def main():
    enc = tiktoken.get_encoding('gpt2')
    w = load_weights()
    bos = [enc.eot_token]
    with torch.no_grad():
        sentences = []
        for s in SENTENCES:
            ids = bos + enc.encode(s)
            _, a = forward(w, ids)
            sentences.append({'text': s, 'tokens': [label(enc, i) for i in ids], 'attn': lower_triangle(a)})

        # Random tokens repeated twice: an induction head at position t attends
        # to the token right after the previous occurrence of token t.
        g = torch.Generator().manual_seed(SEED)
        rand = torch.randint(1000, 10000, (REPEAT_LEN,), generator=g).tolist()
        ids = bos + rand + rand
        logits, a = forward(w, ids)
        loss = torch.nn.functional.cross_entropy(logits[:-1], torch.tensor(ids[1:]), reduction='none')
        second = torch.arange(REPEAT_LEN + 1, 2 * REPEAT_LEN + 1)
        induction = a[:, :, second, second - REPEAT_LEN + 1].mean(-1)
        T = a.shape[-1]
        prev = torch.stack([a[:, :, t, t - 1] for t in range(1, T)], -1).mean(-1)
        first = a[:, :, 1:, 0].mean(-1)
        repeat = {'tokens': [label(enc, i) for i in ids], 'length': REPEAT_LEN,
                  'attn': lower_triangle(a), 'loss': [round(v, 2) for v in loss.tolist()]}

    scores = {k: [[round(v, 3) for v in row] for row in s.tolist()]
              for k, s in (('previous', prev), ('first', first), ('induction', induction))}
    OUT_JSON.write_text(json.dumps({'sentences': sentences, 'repeat': repeat, 'scores': scores},
                                   ensure_ascii=False, separators=(',', ':')))

    for k, s in scores.items():
        top = torch.tensor(s).flatten().topk(4)
        print(k, [(divmod(i, N_HEADS), round(v, 2)) for v, i in zip(top.values.tolist(), top.indices.tolist())])
    print('loss first half %.2f, second half %.2f' % (loss[:REPEAT_LEN].mean(), loss[REPEAT_LEN:].mean()))

    # Slide figure: three heads on the first sentence, as TikZ cells.
    s = sentences[0]
    best = {k: max(((l, h) for l in range(N_LAYERS) for h in range(N_HEADS)), key=lambda p: v[p[0]][p[1]])
            for k, v in scores.items()}
    # Head 4.3 sends most of the attention of " it" to " cat" in this sentence
    # (87%); found by searching all heads, not a general coreference head.
    T = len(s['tokens'])
    on_first = lambda l, h: sum(s['attn'][l][h][t * (t + 1) // 2] for t in range(1, T))
    # On this sentence the strongest first-token head is 5.1, one of the
    # induction heads: with nothing to copy, it parks its attention on <bos>.
    first_head = max(((l, h) for l in range(N_LAYERS) for h in range(N_HEADS)), key=lambda p: on_first(*p))
    picks = [('previous', best['previous']), ('first', first_head), ('it', (4, 3))]
    lines = [f'% Written by scripts/make_gpt2_attention.py: GPT-2 small on "{s["text"]}".']
    for name, (l, h) in picks:
        flat, T, k = s['attn'][l][h], len(s['tokens']), 0
        lines.append(f'% {name}: layer {l}, head {h}')
        lines.append(f'\\def\\gpthead{name}{{{l}.{h}}}')
        lines.append(f'\\def\\gptcells{name}{{%')
        for t in range(T):
            for j in range(t + 1):
                lines.append(f'\\fill[black!{min(flat[k], 100)}] ({j},-{t}) rectangle ++(1,-1);%')
                k += 1
        lines.append('}')
    lines.append('\\def\\gpttokens{' + ','.join(tok.strip().replace('⟨bos⟩', r'$\langle$bos$\rangle$') for tok in s['tokens']) + '}')
    OUT_TEX.write_text('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
