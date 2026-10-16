"""Torch-free inference for the listwise attention ensemble (numpy only).
Used automatically when `import torch` fails (e.g. Windows Application Control blocking torch DLLs).
Weights: listwise_weights.npz (exported from listwise_model.pt by export_weights())."""
import os, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); W = os.path.join(HERE, "listwise_weights.npz")
HEADS = 4

def export_weights():
    import torch
    states = torch.load(os.path.join(HERE, "listwise_model.pt"), map_location="cpu"); out = {"n": np.array(len(states))}
    for i, s in enumerate(states):
        for k, v in s["state"].items(): out[f"m{i}.{k}"] = v.numpy().astype(np.float32)
    np.savez(W, **out)

def _ln(x, w, b, eps=1e-5):
    m = x.mean(-1, keepdims=True); v = x.var(-1, keepdims=True); return (x - m) / np.sqrt(v + eps) * w + b

def _forward(P, p, x, mask):          # x (B,L,d) mask (B,L) True=valid
    g = lambda k: P[f"{p}.{k}"]
    h = np.maximum(x @ g("emb.0.weight").T + g("emb.0.bias"), 0)
    B, L, dm = h.shape; dh = dm // HEADS; neg = np.where(mask, 0.0, -1e9)[:, None, None, :]
    for l in range(2):
        q = f"enc.layers.{l}"
        qkv = h @ g(f"{q}.self_attn.in_proj_weight").T + g(f"{q}.self_attn.in_proj_bias")
        Q, K, V = np.split(qkv, 3, -1); sp = lambda t: t.reshape(B, L, HEADS, dh).transpose(0, 2, 1, 3)
        Q, K, V = sp(Q), sp(K), sp(V)
        a = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(dh) + neg; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
        o = (a @ V).transpose(0, 2, 1, 3).reshape(B, L, dm) @ g(f"{q}.self_attn.out_proj.weight").T + g(f"{q}.self_attn.out_proj.bias")
        h = _ln(h + o, g(f"{q}.norm1.weight"), g(f"{q}.norm1.bias"))
        f = np.maximum(h @ g(f"{q}.linear1.weight").T + g(f"{q}.linear1.bias"), 0) @ g(f"{q}.linear2.weight").T + g(f"{q}.linear2.bias")
        h = _ln(h + f, g(f"{q}.norm2.weight"), g(f"{q}.norm2.bias"))
    z = np.maximum(h @ g("out.0.weight").T + g("out.0.bias"), 0) @ g("out.2.weight").T + g("out.2.bias")
    return (z + x @ g("skip.weight").T + g("skip.bias"))[..., 0]

_C = {}
def predict(Z, rows, L=40, bs=256):
    if "P" not in _C: _C["P"] = dict(np.load(W))
    P = _C["P"]; n = int(P["n"]); out = np.zeros(len(Z), np.float32); acc = np.zeros(len(Z), np.float64)
    for i0 in range(0, len(rows), bs):
        R = rows[i0:i0 + bs]; X = np.zeros((len(R), L, Z.shape[1]), np.float32); M = np.zeros((len(R), L), bool)
        for j, r in enumerate(R): X[j, :len(r)] = Z[r]; M[j, :len(r)] = True
        s = np.mean([1 / (1 + np.exp(-_forward(P, f"m{k}", X, M))) for k in range(n)], 0)
        for j, r in enumerate(R): out[r] = s[j, :len(r)]
    return out
