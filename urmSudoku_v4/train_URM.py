"""Universal Grid-to-Grid URM Training Engine

Domain-agnostic training loop. Called by a domain script that provides:
    - create_structural_encoding(hidden_size, dtype) -> nn.Module | None
    - random_transform() -> dict
    - apply_transform(tokens, t, device, inverse) -> tokens
    - decode_output(token_ids_np) -> grid
    - draw_output(grid, title) -> PIL.Image
    - count_violations(grid) -> int
    - mix_difficulty(inputs, labels, prob) -> (inputs, labels)
    - setup_globals(data_path) -> (GH, GW, NT, BH, BW)

Usage from domain script:
    from train_URM import train, parse_args
    train(domain_callbacks, parse_args())
"""
import os, math, glob, re, argparse, io
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from URM_Core import (URMConfig, URM, StructuralEncoding, TTAEnsemble,
                      EMA, stablemax_cross_entropy, IGNORE_LABEL_ID)

# ── Config ────────────────────────────────────────────────

@dataclass
class TrainConfig(URMConfig):
    data_path: str = "data/sudoku"
    epochs: int = 120_000
    lr: float = 1e-4
    lr_from: Optional[float] = None
    lr_to: Optional[float] = None
    weight_decay: float = 0.1
    use_ema: bool = True
    ema_decay: float = 0.999
    warmup_steps: int = 1000
    eval_interval: int = 4000
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    graphs_dir: str = "graphs"
    use_bf16: bool = True
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    grad_clip: float = 1.0
    difficulty_mix_prob: float = 0.15
    difficulty_mix_after_step: int = 5000
    tta_num_augments: int = 8
    augment_train: bool = True
    puzzle_emb_ndim: int = 0
    puzzle_emb_lr: float = 1e-4

# ── Dataset ───────────────────────────────────────────────

class GridDataset(IterableDataset):
    """Loads (N, H, W) grids from .npy files. On-the-fly offset +1."""
    def __init__(self, data_path, split, batch_size, seed=0):
        self.inputs = np.load(os.path.join(data_path, f"{split}_inputs.npy"), mmap_mode='r')
        self.labels = np.load(os.path.join(data_path, f"{split}_labels.npy"), mmap_mode='r')
        self.N, self.bs, self.seed = len(self.inputs), batch_size, seed
        self._ep = 0

    def __iter__(self):
        self._ep += 1
        rng = np.random.default_rng(self.seed * 10000 + self._ep)
        order = rng.permutation(self.N)
        for s in range(0, self.N - self.bs + 1, self.bs):
            idx = order[s:s + self.bs]
            inp = self.inputs[idx].reshape(self.bs, -1).astype(np.int32) + 1
            lab = self.labels[idx].reshape(self.bs, -1).astype(np.int32) + 1
            yield {"inputs": torch.from_numpy(inp.copy()),
                   "labels": torch.from_numpy(lab.copy()),
                   "puzzle_identifiers": torch.zeros(self.bs, dtype=torch.int32)}

# ── Evaluation ────────────────────────────────────────────

def evaluate_model(model, config, step, domain, ema=None):
    GH, GW = domain["GH"], domain["GW"]
    if ema: ema.apply_shadow()
    model.eval()
    try:
        ld = DataLoader(GridDataset(config.data_path, "test", config.global_batch_size, config.seed),
                        batch_size=None, num_workers=0)
        batch = next(iter(ld))
    except Exception:
        ld = DataLoader(GridDataset(config.data_path, "train", config.global_batch_size, config.seed),
                        batch_size=None, num_workers=0)
        batch = next(iter(ld))
    batch = {k: v.to(config.device) for k, v in batch.items()}

    # Visualization GIF
    vb = {k: v[0:1] for k, v in batch.items()}
    target = domain["decode_output"](vb["labels"][0].cpu().numpy())
    frames = [domain["draw_output"](domain["decode_output"](vb["inputs"][0].cpu().numpy()), "Input")]
    carry = model.initial_carry(vb)
    with torch.no_grad():
        for i in range(config.eval_loops):
            carry, out = model(carry, vb, compute_target_q=False)
            g = domain["decode_output"](torch.argmax(out["logits"], -1)[0].cpu().numpy())
            pct = (g == target).sum() / (GH * GW)
            frames.append(domain["draw_output"](g, f"Loop {i+1}/{config.eval_loops} ({pct:.0%})"))
    try:
        path = f"results/step_{step}.gif"
        frames[0].save(path, save_all=True, append_images=frames[1:], duration=300, loop=0)
    except Exception: path = None

    acc = TTAEnsemble.evaluate_batch(
        model, batch, config.eval_loops, config.tta_num_augments, config.vocab_size,
        transform_fn=domain["random_transform"], apply_fn=domain["apply_transform"],
        decode_fn=domain["decode_output"], violation_fn=domain["count_violations"],
        device=config.device)

    model.train()
    if ema: ema.restore()
    if path:
        try: os.rename(path, f"results/step_{step}_tta{config.tta_num_augments}_{acc*100:.1f}.gif")
        except Exception: pass
    return acc

# ── Checkpointing ─────────────────────────────────────────

def save_checkpoint(model, optimizer, adamw_opt, ema, step, config,
                    loss_hist=None, acc_hist=None, tag=None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    m = model._orig_mod if hasattr(model, '_orig_mod') else model
    ckpt = {"step": step, "model_state_dict": m.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss_history": loss_hist or [], "accuracy_history": acc_hist or []}
    if adamw_opt: ckpt["adamw_state"] = adamw_opt.state_dict()
    if ema: ckpt["ema_state"] = ema.state_dict()
    name = "checkpoint_best.pt" if tag == "best" else "checkpoint_latest.pt"
    torch.save(ckpt, os.path.join(config.checkpoint_dir, name))
    print(f"Checkpoint saved: {name}")

def load_checkpoint(config):
    d = config.checkpoint_dir
    if not os.path.exists(d): return None, 0
    for f in ["checkpoint_latest.pt", "checkpoint_best.pt"]:
        p = os.path.join(d, f)
        if os.path.exists(p):
            try:
                ckpt = torch.load(p, map_location=config.device, weights_only=False)
                ckpt["model_state_dict"] = {k.replace("_orig_mod.", ""): v
                                            for k, v in ckpt["model_state_dict"].items()}
                print(f"Loaded {p} (step {ckpt.get('step', 0)})")
                return ckpt, ckpt.get("step", 0)
            except Exception as e: print(f"Warning: {p} corrupted ({e})")
    files = sorted(glob.glob(os.path.join(d, "checkpoint_step_*.pt")),
                   key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group(1)), reverse=True)
    for fp in files:
        try:
            ckpt = torch.load(fp, map_location=config.device, weights_only=False)
            ckpt["model_state_dict"] = {k.replace("_orig_mod.", ""): v
                                        for k, v in ckpt["model_state_dict"].items()}
            return ckpt, int(re.search(r'(\d+)', os.path.basename(fp)).group(1))
        except Exception: pass
    return None, 0

def save_graphs(loss_hist, acc_hist, step, config):
    os.makedirs(config.graphs_dir, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    if loss_hist:
        s, l = zip(*loss_hist)
        a1.plot(s, l, 'b-'); a1.set_xlabel('Step'); a1.set_ylabel('Loss')
        a1.set_title('Loss'); a1.grid(True, alpha=.3)
    if acc_hist:
        s, a = zip(*acc_hist)
        a2.plot(s, [x*100 for x in a], 'b-o', markersize=4)
        a2.set_xlabel('Step'); a2.set_ylabel('%'); a2.set_title('Accuracy')
        a2.grid(True, alpha=.3); a2.set_ylim(0, 100)
    plt.tight_layout()
    plt.savefig(os.path.join(config.graphs_dir, f"graphs_{step}.png"), dpi=150); plt.close(fig)

# ── Utilities ─────────────────────────────────────────────

def get_lr(step, config):
    if config.lr_from is not None and config.lr_to is not None:
        if step < config.warmup_steps: return config.lr_from * step / max(1, config.warmup_steps)
        return config.lr_from + (config.lr_to - config.lr_from) * min(step / max(1, config.epochs), 1.0)
    if step < config.warmup_steps: return config.lr * step / config.warmup_steps
    p = min((step - config.warmup_steps) / max(1, config.epochs - config.warmup_steps), 1.0)
    return config.lr * 0.5 * (1 + math.cos(math.pi * p))

def get_param_groups(model):
    muon, adamw = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (muon if p.ndim == 2 and 'embed' not in name.lower() else adamw).append(p)
    return muon, adamw

# ── Args ──────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Universal Grid-to-Grid URM Training")
    p.add_argument("--data_path", type=str, default="data/sudoku")
    p.add_argument("--steps", type=int, default=120_000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr_from", type=float, default=None)
    p.add_argument("--lr_to", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--eval_interval", type=int, default=4000)
    p.add_argument("--eval_loops", type=int, default=8)
    p.add_argument("--tta_augments", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--no_compile", action="store_true")
    p.add_argument("--no_ema", action="store_true")
    p.add_argument("--no_augment", action="store_true")
    return p.parse_args()

# ── Main Training ─────────────────────────────────────────

def train(domain, args):
    """Run training with the provided domain callbacks.

    domain is a dict with keys:
        GH, GW, NT, BH, BW               — grid dimensions (set by setup_globals)
        create_structural_encoding        — fn(hidden_size, dtype) -> nn.Module
        random_transform                  — fn() -> dict
        apply_transform                   — fn(tokens, t, device, inverse) -> tokens
        decode_output                     — fn(token_ids_np) -> grid
        draw_output                       — fn(grid, title) -> PIL.Image
        count_violations                  — fn(grid) -> int
        mix_difficulty                    — fn(inputs, labels, prob) -> (inputs, labels)
    """
    torch.set_float32_matmul_precision('high')
    config = TrainConfig()
    config.data_path, config.epochs, config.lr = args.data_path, args.steps, args.lr
    config.lr_from, config.lr_to = args.lr_from, args.lr_to
    config.global_batch_size = args.batch_size
    config.eval_interval, config.eval_loops = args.eval_interval, args.eval_loops
    config.tta_num_augments, config.seed = args.tta_augments, args.seed
    config.checkpoint_dir = args.checkpoint_dir
    config.use_ema = not args.no_ema
    config.augment_train = not args.no_augment

    GH, GW, NT = domain["GH"], domain["GW"], domain["NT"]
    BH, BW = domain["BH"], domain["BW"]
    config.vocab_size = NT + 1
    config.seq_len = GH * GW
    config.num_puzzle_identifiers = 1
    config.puzzle_emb_ndim = 0

    lr_desc = f"Linear {config.lr_from}→{config.lr_to}" if config.lr_from else f"Cosine {config.lr}"
    blk_desc = f"{BH}×{BW} blocks" if BH > 0 else "no blocks"
    print(f"{'='*60}\nUniversal Grid-to-Grid URM Training\n{'='*60}")
    print(f"Grid: {GH}×{GW}, Tokens: {NT}, Vocab: {config.vocab_size}, Blocks: {blk_desc}")
    print(f"Steps: {config.epochs}, LR: {lr_desc}, BS: {config.global_batch_size}")
    print(f"EMA: {config.use_ema}, Augment: {config.augment_train}, Loops: {config.loops}")
    print(f"TTA: {config.tta_num_augments} augments @ {config.eval_loops} loops")
    print(f"Violations: {'row+col+block' if BH > 0 else 'disabled (uniform TTA weights)'}")
    print(f"{'='*60}")
    for d in ["results", config.checkpoint_dir, config.graphs_dir]: os.makedirs(d, exist_ok=True)
    torch.manual_seed(config.seed)

    # ── Model ─────────────────────────────────────────────
    struct_enc = domain["create_structural_encoding"](config.hidden_size, getattr(torch, config.forward_dtype))
    model = URM(config, structural_encoding=struct_enc).to(config.device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    ckpt, start_step = load_checkpoint(config)
    if ckpt:
        ms = model.state_dict(); loaded = ckpt["model_state_dict"]; matched = 0
        for k, v in loaded.items():
            if k in ms and ms[k].shape == v.shape: ms[k] = v; matched += 1
        model.load_state_dict(ms)
        print(f"Loaded {matched} params from step {start_step}")
    else: start_step = 0

    ema = EMA(model, config.ema_decay) if config.use_ema else None
    if ckpt and ema and "ema_state" in ckpt:
        try: ema.load_state_dict(ckpt["ema_state"])
        except Exception: pass

    if not args.no_compile: model = torch.compile(model)
    m_raw = model._orig_mod if hasattr(model, '_orig_mod') else model
    muon_p, adamw_p = get_param_groups(m_raw)
    print(f"Muon: {sum(p.numel() for p in muon_p):,}, AdamW: {sum(p.numel() for p in adamw_p):,}")

    opt = torch.optim.Muon(muon_p, lr=config.lr, weight_decay=config.weight_decay,
                           momentum=config.muon_momentum, nesterov=True,
                           ns_steps=config.muon_ns_steps, adjust_lr_fn="match_rms_adamw")
    adamw = torch.optim.AdamW(adamw_p, lr=config.lr, weight_decay=config.weight_decay,
                              betas=(0.9, 0.95)) if adamw_p else None

    if ckpt:
        try: opt.load_state_dict(ckpt["optimizer_state_dict"])
        except Exception: pass
        if adamw and "adamw_state" in ckpt:
            try: adamw.load_state_dict(ckpt["adamw_state"])
            except Exception: pass

    ctx = torch.autocast('cuda', torch.bfloat16) if config.use_bf16 and 'cuda' in config.device else nullcontext()
    model.train(); step = start_step
    loss_hist = ckpt.get("loss_history", []) if ckpt else []
    acc_hist = ckpt.get("accuracy_history", []) if ckpt else []
    best_acc = max([h[1] for h in acc_hist], default=0.0)
    if best_acc > 0: print(f"Best accuracy: {best_acc*100:.1f}%")

    dataset = GridDataset(config.data_path, "train", config.global_batch_size, config.seed)
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    train_iter = iter(dataloader)
    carry, smooth_loss = None, 0.0
    pbar = tqdm(total=config.epochs, initial=start_step, desc="Training")

    try:
        while step < config.epochs:
            try: batch = next(train_iter)
            except StopIteration: train_iter = iter(dataloader); batch = next(train_iter)
            batch = {k: v.to(config.device) for k, v in batch.items()}

            # On-the-fly augmentation
            if config.augment_train:
                t = domain["random_transform"]()
                batch["inputs"] = domain["apply_transform"](batch["inputs"], t, config.device)
                batch["labels"] = domain["apply_transform"](batch["labels"], t, config.device)

            # Difficulty mixing curriculum
            if (config.difficulty_mix_prob > 0 and step >= config.difficulty_mix_after_step
                    and np.random.random() < 0.3):
                batch["inputs"], batch["labels"] = domain["mix_difficulty"](
                    batch["inputs"], batch["labels"], config.difficulty_mix_prob)

            if carry is None: carry = model.initial_carry(batch)
            lr = get_lr(step, config)
            for g in opt.param_groups: g['lr'] = lr
            if adamw:
                for g in adamw.param_groups: g['lr'] = lr

            with ctx:
                carry, outputs = model(carry, batch, compute_target_q=False)
                labels = carry.current_data["labels"]
                mask = labels != IGNORE_LABEL_ID
                div = mask.sum(-1).clamp_min(1).unsqueeze(-1)
                lm_loss = (stablemax_cross_entropy(outputs["logits"], labels) / div).sum()
                preds = torch.argmax(outputs["logits"], -1)
                seq_ok = (mask & (preds == labels)).sum(-1) == mask.sum(-1)
                q_loss = F.binary_cross_entropy_with_logits(
                    outputs["q_halt_logits"], seq_ok.to(outputs["q_halt_logits"].dtype), reduction="sum")
                mid_loss = torch.tensor(0.0, device=config.device)
                if outputs["mid_logits"] is not None and config.mid_loop_loss_weight > 0:
                    mid_loss = (stablemax_cross_entropy(outputs["mid_logits"], labels) / div).sum()
                total = lm_loss + 0.5 * q_loss + config.mid_loop_loss_weight * mid_loss

            (total / config.global_batch_size).backward()
            carry.current_hidden = carry.current_hidden.detach()

            if config.grad_clip > 0:
                all_p = list(muon_p) + list(adamw_p)
                torch.nn.utils.clip_grad_norm_(all_p, config.grad_clip)

            opt.step(); opt.zero_grad()
            if adamw: adamw.step(); adamw.zero_grad()
            if ema: ema.update()

            step += 1; pbar.update(1)
            cl = total.item() / config.global_batch_size
            smooth_loss = 0.95 * smooth_loss + 0.05 * cl if smooth_loss > 0 else cl
            pbar.set_postfix(loss=f"{smooth_loss:.4f}", lr=f"{lr:.2e}")

            if step % config.eval_interval == 0:
                loss_hist.append((step, smooth_loss))
                acc = evaluate_model(model, config, step, domain, ema)
                acc_hist.append((step, acc))
                is_best = acc > best_acc
                if is_best: best_acc = acc
                tqdm.write(f"Step {step}: Loss={smooth_loss:.4f}, "
                           f"TTA@{config.eval_loops}={acc*100:.1f}%, "
                           f"Best={best_acc*100:.1f}%, LR={lr:.2e}"
                           + (" ★" if is_best else ""))
                if is_best:
                    save_checkpoint(model, opt, adamw, ema, step, config, loss_hist, acc_hist, "best")
                save_checkpoint(model, opt, adamw, ema, step, config, loss_hist, acc_hist)
                save_graphs(loss_hist, acc_hist, step, config)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    print(f"Done! Best accuracy: {best_acc*100:.1f}%")