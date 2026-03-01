"""Universal Grid-to-Grid Domain for URM Training

Auto-detects grid dimensions and block structure from .npy data.
Provides all domain callbacks for train_URM.py.

Usage:
    python train_URM_Universal.py --data_path data/sudoku --lr_from 1e-4 --lr_to 4e-5
"""
import math, io
import numpy as np
import torch
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image
from URM_Core import StructuralEncoding, IGNORE_LABEL_ID
from train_URM import train, parse_args

# ── Grid config (set by setup_globals) ────────────────────
GH, GW, NT = 9, 9, 10
BH, BW = 0, 0

PALETTE = ['#000000','#0074D9','#FF4136','#2ECC40','#FFDC00',
           '#AAAAAA','#F012BE','#FF851B','#7FDBFF','#B10DC9',
           '#808080','#01FF70','#85144b','#3D9970','#111111','#DDDDDD']

# ══════════════════════════════════════════════════════════
# AUTO-DETECTION
# ══════════════════════════════════════════════════════════

def setup_globals(data_path):
    """Auto-detect grid config from .npy files. No config files needed."""
    global GH, GW, NT, BH, BW
    import os
    for split in ["train", "test"]:
        p = os.path.join(data_path, f"{split}_inputs.npy")
        if os.path.exists(p):
            arr = np.load(p, mmap_mode='r')
            lab = np.load(os.path.join(data_path, f"{split}_labels.npy"), mmap_mode='r')
            GH, GW = arr.shape[1], arr.shape[2]
            NT = max(int(arr.max()), int(lab.max())) + 1
            BH, BW = 0, 0
            if GH == GW:
                s = int(math.isqrt(GH))
                if s * s == GH and s > 1:
                    BH, BW = s, s
            print(f"Auto-detected: {GH}×{GW} grid, {NT} tokens, "
                  f"blocks={'%d×%d' % (BH, BW) if BH > 0 else 'none'}")
            return
    raise FileNotFoundError(f"No *_inputs.npy found in {data_path}")

# ══════════════════════════════════════════════════════════
# DOMAIN CALLBACKS
# ══════════════════════════════════════════════════════════

def create_structural_encoding(hs, dtype):
    """2D positional encoding. Adds block group if blocks detected."""
    seq = GH * GW
    rows = torch.arange(seq) // GW
    cols = torch.arange(seq) % GW
    groups, num_g = [rows, cols], [GH, GW]
    if BH > 0 and BW > 0:
        blocks = (rows // BH) * (GW // BW) + cols // BW
        groups.append(blocks)
        num_g.append((GH // BH) * (GW // BW))
    return StructuralEncoding(groups, num_g, hs, dtype)


def random_transform():
    """Random symmetry: token perm + flip + transpose + band/row/col perm."""
    t = dict(grid_h=GH, grid_w=GW, num_tokens=NT, block_h=BH, block_w=BW)
    # Token permutation: only non-blank tokens (1..NT-1), blank=0 stays
    digit_perm = np.random.permutation(NT - 1) + 1
    full_perm = np.zeros(NT, dtype=np.int64)
    full_perm[1:] = digit_perm
    t["token_perm"] = full_perm
    inv_perm = np.zeros(NT, dtype=np.int64)
    for i in range(NT): inv_perm[full_perm[i]] = i
    t["token_inv"] = inv_perm
    # Spatial
    t["flip_h"] = bool(np.random.random() < .5)
    t["flip_v"] = bool(np.random.random() < .5)
    t["transpose"] = bool(GH == GW and np.random.random() < .5)
    # Band/row/col permutations (only with blocks)
    if BH > 0 and BW > 0:
        nb_r, nb_c = GH // BH, GW // BW
        t["band_perm"] = np.random.permutation(nb_r)
        t["stack_perm"] = np.random.permutation(nb_c)
        t["row_in_band"] = [np.random.permutation(BH) for _ in range(nb_r)]
        t["col_in_stack"] = [np.random.permutation(BW) for _ in range(nb_c)]
    return t


def apply_transform(tok, t, dev, inverse=False):
    """Apply or invert a universal 2D transform on token sequences."""
    sq = tok.dim() == 1
    if sq: tok = tok.unsqueeze(0)
    H, W, nt = t["grid_h"], t["grid_w"], t["num_tokens"]
    S = H * W

    idx = torch.arange(S, device=dev).view(H, W)

    if not inverse:
        if t.get("transpose", False): idx = idx.t().contiguous()
        if t["flip_h"]: idx = idx.flip(1)
        if t["flip_v"]: idx = idx.flip(0)
        if "band_perm" in t:
            bh, bw = t["block_h"], t["block_w"]
            cur_h, cur_w = idx.shape
            rp = np.concatenate([t["band_perm"][b] * bh + t["row_in_band"][b]
                                 for b in range(cur_h // bh)])
            idx = idx[torch.tensor(rp, device=dev)]
            cp = np.concatenate([t["stack_perm"][s] * bw + t["col_in_stack"][s]
                                 for s in range(cur_w // bw)])
            idx = idx[:, torch.tensor(cp, device=dev)]
    else:
        if "band_perm" in t:
            bh, bw = t["block_h"], t["block_w"]
            cur_h, cur_w = idx.shape
            rp = np.concatenate([t["band_perm"][b] * bh + t["row_in_band"][b]
                                 for b in range(cur_h // bh)])
            idx = idx[torch.tensor(np.argsort(rp), device=dev)]
            cp = np.concatenate([t["stack_perm"][s] * bw + t["col_in_stack"][s]
                                 for s in range(cur_w // bw)])
            idx = idx[:, torch.tensor(np.argsort(cp), device=dev)]
        if t["flip_v"]: idx = idx.flip(0)
        if t["flip_h"]: idx = idx.flip(1)
        if t.get("transpose", False): idx = idx.t().contiguous()

    fp = idx.reshape(-1)

    # Token remapping (offset space: 0=pad, k+1 → perm[k]+1)
    tp_orig, ip_orig = t["token_perm"], t["token_inv"]
    fm = torch.arange(nt + 1, device=dev, dtype=torch.long)
    im = torch.arange(nt + 1, device=dev, dtype=torch.long)
    for k in range(nt):
        fm[k + 1] = int(tp_orig[k]) + 1
        im[int(tp_orig[k]) + 1] = k + 1

    if not inverse:
        r = fm[tok[:, fp].long()].to(tok.dtype)
    else:
        r = im[tok.long()].to(tok.dtype)[:, fp]
    return r.squeeze(0) if sq else r


def decode_output(tids):
    """Token IDs → 2D grid (subtract 1 to undo offset)."""
    g = np.zeros((GH, GW), dtype=int)
    for i, v in enumerate(tids.reshape(-1)[:GH * GW]):
        g[i // GW][i % GW] = max(0, int(v) - 1)
    return g


def draw_output(grid, title=""):
    """Draw colored grid as PIL Image."""
    h, w = grid.shape
    fig, ax = plt.subplots(figsize=(max(2, w * .45), max(2, h * .45)))
    n = max(NT, int(grid.max()) + 1)
    cm = ListedColormap([PALETTE[i % len(PALETTE)] for i in range(n)])
    ax.imshow(grid, cmap=cm, vmin=0, vmax=n - 1, aspect='equal')
    ax.set_xticks(np.arange(-.5, w, 1), minor=True)
    ax.set_yticks(np.arange(-.5, h, 1), minor=True)
    ax.grid(which='minor', color='#333', linewidth=.5)
    ax.tick_params(which='both', size=0, labelbottom=False, labelleft=False)
    if BH > 0 and BW > 0:
        for r in range(0, h + 1, BH):
            ax.axhline(r - .5, color='white', linewidth=2)
        for c in range(0, w + 1, BW):
            ax.axvline(c - .5, color='white', linewidth=2)
    ax.set_title(title, fontsize=9)
    if h <= 16 and w <= 16:
        fs = max(5, 14 - max(h, w) // 3)
        for r in range(h):
            for c in range(w):
                if grid[r, c] > 0:
                    ax.text(c, r, str(grid[r, c]), ha='center', va='center',
                            fontsize=fs, color='white' if grid[r, c] in (0, 6, 10, 14) else 'black')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=100); plt.close(fig); buf.seek(0)
    return Image.open(buf)


def count_violations(grid):
    """Count constraint violations. Block-aware when blocks detected."""
    v = 0
    h, w = grid.shape
    for i in range(h):
        for d in range(1, NT):
            rc = np.sum(grid[i, :] == d)
            if rc > 1: v += rc - 1
    for j in range(w):
        for d in range(1, NT):
            cc = np.sum(grid[:, j] == d)
            if cc > 1: v += cc - 1
    if BH > 0 and BW > 0:
        for br in range(h // BH):
            for bc in range(w // BW):
                blk = grid[br*BH:(br+1)*BH, bc*BW:(bc+1)*BW].flatten()
                for d in range(1, NT):
                    cnt = np.sum(blk == d)
                    if cnt > 1: v += cnt - 1
    return v


def mix_difficulty(inp, lab, prob=0.15):
    """Curriculum: reveal random unknown cells."""
    diff = inp != lab
    rv = (torch.rand_like(inp.float()) < prob) & diff
    return torch.where(rv, lab, inp), torch.where(rv, torch.full_like(lab, IGNORE_LABEL_ID), lab)

# ══════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    args = parse_args()
    setup_globals(args.data_path)

    domain = {
        "GH": GH, "GW": GW, "NT": NT, "BH": BH, "BW": BW,
        "create_structural_encoding": create_structural_encoding,
        "random_transform": random_transform,
        "apply_transform": apply_transform,
        "decode_output": decode_output,
        "draw_output": draw_output,
        "count_violations": count_violations,
        "mix_difficulty": mix_difficulty,
    }
    train(domain, args)