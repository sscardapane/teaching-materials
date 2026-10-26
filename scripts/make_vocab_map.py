#!/usr/bin/env python3
"""Precompute the GPT-2 vocabulary map used by html/components/VocabExplorer.vue.

Reads only the token-embedding matrix of GPT-2 (HTTP range requests on the
public safetensors file), projects it to 2D with UMAP and writes a compact
JSON file to html/data/gpt2_vocab_map.json (copied to the site by
scripts/build_site.py).

Dependencies (kept out of the main requirements; use a scratch venv):
    pip install numpy umap-learn tiktoken
"""
import json
import struct
import urllib.request
from pathlib import Path

import numpy as np
import tiktoken
import umap

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'html' / 'data' / 'gpt2_vocab_map.json'
URL = 'https://huggingface.co/openai-community/gpt2/resolve/main/model.safetensors'
NEIGHBOURS = 8


def fetch(start, end):
    req = urllib.request.Request(URL, headers={'Range': f'bytes={start}-{end}'})
    with urllib.request.urlopen(req) as r:
        return r.read()


def token_embeddings():
    n = struct.unpack('<Q', fetch(0, 7))[0]
    header = json.loads(fetch(8, 8 + n - 1))
    info = header['wte.weight']
    assert info['dtype'] == 'F32'
    a, b = info['data_offsets']
    raw = fetch(8 + n + a, 8 + n + b - 1)
    return np.frombuffer(raw, dtype='<f4').reshape(info['shape'])


def main():
    enc = tiktoken.get_encoding('gpt2')
    emb = token_embeddings()
    vocab = emb.shape[0]
    tokens = []
    for i in range(vocab):
        raw = enc.decode_single_token_bytes(i)
        try:
            tokens.append(raw.decode('utf-8'))
        except UnicodeDecodeError:
            # A fragment of a multi-byte character: show its bytes.
            tokens.append(''.join(f'<{b:02X}>' for b in raw))

    # Under-trained tokens sit close to the mean embedding (Land and Bartolo, 2024).
    centred = emb - emb.mean(0)
    dist = np.linalg.norm(centred, axis=1)

    unit = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    nbrs = np.zeros((vocab, NEIGHBOURS), dtype=np.int32)
    for s in range(0, vocab, 2048):
        sim = unit[s:s + 2048] @ unit.T
        for r in range(sim.shape[0]):
            sim[r, s + r] = -2
        idx = np.argpartition(-sim, NEIGHBOURS, axis=1)[:, :NEIGHBOURS]
        order = np.take_along_axis(sim, idx, 1).argsort(1)[:, ::-1]
        nbrs[s:s + 2048] = np.take_along_axis(idx, order, 1)

    xy = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='cosine', random_state=0).fit_transform(emb)
    # Scale the bulk of the map to the canvas; a few far outliers are clipped to the border.
    lo, hi = np.percentile(xy, 0.5, axis=0), np.percentile(xy, 99.5, axis=0)
    xy = np.clip((xy - lo) / (hi - lo), -0.04, 1.04)
    xy = np.round(40 + xy * 920).astype(int)

    OUT.write_text(json.dumps({
        'source': 'GPT-2 (openai-community/gpt2) token embeddings, UMAP (cosine)',
        'tokens': tokens,
        'x': xy[:, 0].tolist(),
        'y': xy[:, 1].tolist(),
        'dist': np.round(dist, 3).tolist(),
        'nbrs': nbrs.ravel().tolist(),
        'k': NEIGHBOURS,
    }, ensure_ascii=False, separators=(',', ':')))
    print(OUT, OUT.stat().st_size // 1024, 'KiB')


if __name__ == '__main__':
    main()
