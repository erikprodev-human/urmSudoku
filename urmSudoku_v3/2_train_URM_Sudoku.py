"""URM Sudoku Training Script — v3 with World Model Constraint Loss + Regularization"""
import os, math, json, glob, re, argparse, io
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional, List
import numpy as np
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from URM_Core import (
    URMConfig, URM, CastedSparseEmbeddingSignSGD,
    stablemax_cross_entropy, sudoku_constraint_loss, IGNORE_LABEL_ID,
    random_digit_perm, apply_token_perm, is_valid_sudoku_grid
)

SUDOKU_DATA_PATH = "data/sudoku-extreme-1k-aug-1000"


@dataclass
class SudokuTrainConfig(URMConfig):
    data_path: str = SUDOKU_DATA_PATH
    epochs: int = 50_000
    lr: float = 1e-4
    lr_from: Optional[float] = None
    lr_to: Optional[float] = None
    weight_decay: float = 0.1
    puzzle_emb_lr: float = 1e-4
    puzzle_emb_weight_decay: float = 0.5
    use_ema: bool = True
    ema_decay: float = 0.999
    warmup_steps: int = 1000
    lr_schedule: str = "cosine"
    lr_min_ratio: float = 0.0
    eval_interval: int = 2000
    checkpoint_interval: int = 2000
    seed: int = 0
    checkpoint_dir: str = "checkpoints"
    graphs_dir: str = "graphs"
    resume_step: int = 0
    gradient_accumulation_steps: int = 2
    use_bf16: bool = True
    muon_momentum: float = 0.95
    muon_ns_steps: int = 5
    grad_clip: float = 1.0


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.model, self.decay, self.shadow, self.backup = model, decay, {}, {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}

    def state_dict(self):
        return {name: tensor.clone() for name, tensor in self.shadow.items()}

    def load_state_dict(self, state_dict):
        self.shadow = {name: tensor.clone() for name, tensor in state_dict.items()}


@dataclass
class PuzzleDatasetMetadata:
    pad_id: int
    ignore_label_id: Optional[int]
    blank_identifier_id: int
    vocab_size: int
    seq_len: int
    num_puzzle_identifiers: int
    total_groups: int
    mean_puzzle_examples: float
    sets: List[str]


class SudokuDataset(IterableDataset):
    def __init__(self, config: SudokuTrainConfig, split: str = "train"):
        super().__init__()
        self.config, self.split = config, split
        with open(os.path.join(config.data_path, split, "dataset.json"), "r") as f:
            self.metadata = PuzzleDatasetMetadata(**json.load(f))
        self.local_batch_size = config.global_batch_size
        self._data, self._iters = None, 0

    def _lazy_load_dataset(self):
        if self._data is not None:
            return
        field_mmap_modes = {"inputs": "r", "labels": "r", "puzzle_identifiers": None, "puzzle_indices": None, "group_indices": None}
        self._data = {}
        for set_name in self.metadata.sets:
            self._data[set_name] = {fn: np.load(os.path.join(self.config.data_path, self.split, f"{set_name}__{fn}.npy"), mmap_mode=mm) for fn, mm in field_mmap_modes.items()}

    def _collate_batch(self, batch):
        batch = {k: v.astype(np.int32) for k, v in batch.items()}
        if self.metadata.ignore_label_id is not None:
            batch["labels"][batch["labels"] == self.metadata.ignore_label_id] = IGNORE_LABEL_ID
        if batch["puzzle_identifiers"].size < self.local_batch_size:
            pad_size = self.local_batch_size - batch["puzzle_identifiers"].size
            pad_values = {"inputs": self.metadata.pad_id, "labels": IGNORE_LABEL_ID, "puzzle_identifiers": self.metadata.blank_identifier_id}
            batch = {k: np.pad(v, ((0, pad_size),) + ((0, 0),) * (v.ndim - 1), constant_values=pad_values[k]) for k, v in batch.items()}
        return {k: torch.from_numpy(v) for k, v in batch.items()}

    def _sample_batch(self, rng, group_order, puzzle_indices, group_indices, start_index):
        batch, batch_puzzle_indices, current_size = [], [], 0
        while (start_index < group_order.size) and (current_size < self.config.global_batch_size):
            group_id = group_order[start_index]
            puzzle_id = rng.integers(group_indices[group_id], group_indices[group_id + 1])
            start_index += 1
            puzzle_start = puzzle_indices[puzzle_id]
            puzzle_size = int(puzzle_indices[puzzle_id + 1] - puzzle_start)
            append_size = min(puzzle_size, self.config.global_batch_size - current_size)
            batch_puzzle_indices.append(np.full(append_size, puzzle_id, dtype=np.int32))
            batch.append(puzzle_start + rng.choice(puzzle_size, append_size, replace=False))
            current_size += append_size
        return start_index, np.concatenate(batch), np.concatenate(batch_puzzle_indices)

    def __iter__(self):
        self._lazy_load_dataset()
        for set_name, dataset in self._data.items():
            self._iters += 1
            rng = np.random.Generator(np.random.Philox(seed=self.config.seed + self._iters))
            group_order = rng.permutation(dataset["group_indices"].size - 1)
            start_index = 0
            while start_index < group_order.size:
                start_index, batch_indices, batch_puzzle_indices = self._sample_batch(rng, group_order, dataset["puzzle_indices"], dataset["group_indices"], start_index)
                if batch_puzzle_indices.size < self.config.global_batch_size:
                    break
                yield self._collate_batch({"inputs": dataset["inputs"][batch_indices], "labels": dataset["labels"][batch_indices], "puzzle_identifiers": dataset["puzzle_identifiers"][batch_puzzle_indices]})


def draw_sudoku(grid, title="Sudoku"):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)
    for i in range(10):
        lw = 2 if i % 3 == 0 else 0.5
        ax.plot([i, i], [0, 9], color='black', linewidth=lw)
        ax.plot([0, 9], [i, i], color='black', linewidth=lw)
    for r in range(9):
        for c in range(9):
            if grid[r][c] != 0:
                ax.text(c + 0.5, r + 0.5, str(grid[r][c]), ha='center', va='center', fontsize=16)
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight'); plt.close(fig); buf.seek(0)
    return Image.open(buf)


def decode_grid(token_ids):
    grid = np.zeros((9, 9), dtype=int)
    for i, tid in enumerate(token_ids.reshape(81)):
        if tid >= 2:
            grid[i // 9][i % 9] = tid - 1
    return grid


# ──────────────────────────────────────────────────────────
# TTA with Majority Voting
# ──────────────────────────────────────────────────────────

def run_single_inference(model, single_batch, num_loops):
    carry = model.initial_carry(single_batch)
    for _ in range(num_loops):
        carry, outputs = model(carry, single_batch, compute_target_q=False)
    return torch.argmax(outputs["logits"][0], dim=-1)


def evaluate_puzzle_with_tta(model, single_batch, config):
    num_augments = config.tta_num_augments
    num_loops = config.loops
    labels = single_batch["labels"][0]
    mask = labels != IGNORE_LABEL_ID
    device = labels.device

    if mask.sum() == 0:
        return True, decode_grid(labels.cpu().numpy())

    all_preds = []
    pred = run_single_inference(model, single_batch, num_loops)
    all_preds.append(pred)

    for _ in range(num_augments):
        perm, inv_perm = random_digit_perm()
        aug_inputs = apply_token_perm(single_batch["inputs"], perm)
        aug_batch = {
            "inputs": aug_inputs,
            "labels": single_batch["labels"],
            "puzzle_identifiers": single_batch["puzzle_identifiers"]
        }
        aug_pred = run_single_inference(model, aug_batch, num_loops)
        orig_pred = apply_token_perm(aug_pred.unsqueeze(0), inv_perm).squeeze(0)
        all_preds.append(orig_pred)

    weights = []
    for p in all_preds:
        grid = decode_grid(p.cpu().numpy())
        w = 3.0 if is_valid_sudoku_grid(grid) else 1.0
        weights.append(w)

    final_pred = torch.zeros(81, dtype=torch.long, device=device)
    for pos in range(81):
        vote_counts = torch.zeros(11, device=device)
        for k, p in enumerate(all_preds):
            vote_counts[p[pos].long()] += weights[k]
        final_pred[pos] = vote_counts.argmax()

    is_correct = ((final_pred == labels) | ~mask).all().item()
    return is_correct, decode_grid(final_pred.cpu().numpy())


def evaluate_puzzle_pass_at_k(model, batch, config):
    batch_size = batch["inputs"].shape[0]
    correct_count = 0
    max_eval = min(batch_size, 64)

    with torch.no_grad():
        for b in range(max_eval):
            labels = batch["labels"][b]
            mask = labels != IGNORE_LABEL_ID
            if mask.sum() == 0:
                continue
            single_batch = {
                "inputs": batch["inputs"][b:b+1],
                "labels": batch["labels"][b:b+1],
                "puzzle_identifiers": batch["puzzle_identifiers"][b:b+1]
            }
            is_correct, _ = evaluate_puzzle_with_tta(model, single_batch, config)
            if is_correct:
                correct_count += 1

    return correct_count / max_eval if max_eval > 0 else 0.0


def evaluate_model(model, config, step, ema=None):
    if ema is not None:
        ema.apply_shadow()
    model.eval()

    try:
        val_dataset = SudokuDataset(config, split="test")
    except Exception:
        val_dataset = SudokuDataset(config, split="train")
    val_loader = DataLoader(val_dataset, batch_size=None, num_workers=0)

    try:
        batch = next(iter(val_loader))
    except StopIteration:
        if ema is not None:
            ema.restore()
        return 0.0

    batch = {k: v.to(config.device) for k, v in batch.items()}

    # Visualization GIF
    viz_batch = {k: v[0:1] for k, v in batch.items()}
    viz_frames = []
    carry = model.initial_carry(viz_batch)
    labels_viz = viz_batch["labels"][0].cpu().numpy()
    input_ids = viz_batch["inputs"][0].cpu().numpy()
    input_grid = decode_grid(input_ids)
    target_grid_decoded = decode_grid(labels_viz)
    viz_frames.append(draw_sudoku(input_grid, title=f"Step 0 (Input) {(input_grid == target_grid_decoded).sum() / 81.0:.0%}"))

    with torch.no_grad():
        for i in range(config.loops):
            carry, outputs = model(carry, viz_batch, compute_target_q=False)
            current_grid = decode_grid(torch.argmax(outputs["logits"], dim=-1)[0].cpu().numpy())
            viz_frames.append(draw_sudoku(current_grid, title=f"Loop {i+1}/{config.loops} ({(current_grid == target_grid_decoded).sum() / 81.0:.0%})"))

    save_path = f"results/step_{step}.gif"
    viz_frames[0].save(save_path, save_all=True, append_images=viz_frames[1:], duration=300, loop=0)

    # TTA evaluation
    pass_at_k = evaluate_puzzle_pass_at_k(model, batch, config)

    model.train()
    if ema is not None:
        ema.restore()

    new_save_path = f"results/step_{step}_tta{config.tta_num_augments}_loops{config.loops}_{pass_at_k*100:.1f}.gif"
    os.rename(save_path, new_save_path)
    return pass_at_k


# ──────────────────────────────────────────────────────────
# Checkpoint utilities
# ──────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, adamw_optimizer, puzzle_optimizer, ema, step, config,
                    loss_history=None, accuracy_history=None, tag=None):
    os.makedirs(config.checkpoint_dir, exist_ok=True)
    model_to_save = model._orig_mod if hasattr(model, '_orig_mod') else model
    checkpoint = {
        "step": step,
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss_history": loss_history or [],
        "accuracy_history": accuracy_history or []
    }
    if adamw_optimizer is not None:
        checkpoint["adamw_optimizer_state_dict"] = adamw_optimizer.state_dict()
    if puzzle_optimizer is not None:
        checkpoint["puzzle_optimizer_state_dict"] = puzzle_optimizer.state_dict()
    if ema is not None:
        checkpoint["ema_state_dict"] = ema.state_dict()
    filename = f"checkpoint_best.pt" if tag == "best" else f"checkpoint_step_{step}.pt"
    torch.save(checkpoint, os.path.join(config.checkpoint_dir, filename))
    print(f"Checkpoint saved: {filename}")


def load_latest_checkpoint(config):
    if not os.path.exists(config.checkpoint_dir):
        return None, 0
    checkpoint_files = glob.glob(os.path.join(config.checkpoint_dir, "checkpoint_step_*.pt"))
    if not checkpoint_files:
        return None, 0
    def get_step(f):
        match = re.search(r'checkpoint_step_(\d+)\.pt', f)
        return int(match.group(1)) if match else 0
    for checkpoint_path in sorted(checkpoint_files, key=get_step, reverse=True):
        step = get_step(checkpoint_path)
        try:
            print(f"Loading checkpoint: {checkpoint_path} (step {step})")
            checkpoint = torch.load(checkpoint_path, map_location=config.device, weights_only=False)
            checkpoint["model_state_dict"] = {k.replace("_orig_mod.", ""): v for k, v in checkpoint["model_state_dict"].items()}
            return checkpoint, step
        except Exception as e:
            print(f"Warning: Checkpoint {checkpoint_path} corrupted, skipping... ({e})")
            try:
                os.remove(checkpoint_path)
            except:
                pass
    return None, 0


def save_training_graphs(loss_history, accuracy_history, step, config):
    os.makedirs(config.graphs_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    if loss_history:
        steps_loss, losses = zip(*loss_history)
        ax1.plot(steps_loss, losses, 'b-', linewidth=1)
        ax1.set_xlabel('Step'); ax1.set_ylabel('Training Loss'); ax1.set_title('Training Loss over Time'); ax1.grid(True, alpha=0.3)
    if accuracy_history:
        steps_acc = [h[0] for h in accuracy_history]
        pass_k_accs = [h[1] * 100 for h in accuracy_history]
        ax2.plot(steps_acc, pass_k_accs, 'b-', linewidth=2, marker='o', markersize=4, label=f'TTA ({config.tta_num_augments} aug) @ {config.loops} loops')
        ax2.set_xlabel('Step'); ax2.set_ylabel('Accuracy (%)'); ax2.set_title('Accuracy over Time (TTA)')
        ax2.grid(True, alpha=0.3); ax2.set_ylim(0, 100); ax2.legend(loc='lower right')
    plt.tight_layout(); plt.savefig(os.path.join(config.graphs_dir, f"training_graphs_step_{step}.png"), dpi=150); plt.close(fig)


def get_lr(step, config):
    if config.lr_from is not None and config.lr_to is not None:
        if step < config.warmup_steps:
            return config.lr_from * step / max(1, config.warmup_steps)
        progress = min(step / max(1, config.epochs), 1.0)
        return config.lr_from + (config.lr_to - config.lr_from) * progress
    if step < config.warmup_steps:
        return config.lr * step / config.warmup_steps
    progress = min((step - config.warmup_steps) / max(1, config.epochs - config.warmup_steps), 1.0)
    if config.lr_schedule == "cosine":
        return config.lr * (config.lr_min_ratio + (1 - config.lr_min_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))
    return config.lr * (1 - progress * (1 - config.lr_min_ratio))


def get_parameter_groups(model, config):
    muon_params, adamw_params = [], []
    puzzle_local_weights = None
    if config.puzzle_emb_ndim > 0:
        puzzle_local_weights = model.inner.puzzle_emb.local_weights
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if puzzle_local_weights is not None and param is puzzle_local_weights:
            continue
        if param.ndim == 2 and 'embed' not in name.lower():
            muon_params.append(param)
        else:
            adamw_params.append(param)
    return muon_params, adamw_params


def parse_args():
    parser = argparse.ArgumentParser(description="Train URM Sudoku Model (v3 - World Model)")
    parser.add_argument("--epochs", "--steps", type=int, default=100_000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_from", type=float, default=None)
    parser.add_argument("--lr_to", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--checkpoint_interval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data_path", type=str, default=SUDOKU_DATA_PATH)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--no_ema", action="store_true")
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--no_compile", action="store_true")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--no_bf16", action="store_true")
    parser.add_argument("--muon_momentum", type=float, default=0.95)
    parser.add_argument("--muon_ns_steps", type=int, default=5)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    # TTA
    parser.add_argument("--tta_augments", type=int, default=8)
    # Architecture
    parser.add_argument("--no_sudoku_struct", action="store_true")
    parser.add_argument("--no_loop_gate", action="store_true")
    parser.add_argument("--no_constraint_bias", action="store_true")
    # Regularization
    parser.add_argument("--attn_dropout", type=float, default=0.1)
    parser.add_argument("--mlp_dropout", type=float, default=0.1)
    parser.add_argument("--loop_noise_std", type=float, default=0.02)
    parser.add_argument("--constraint_loss_weight", type=float, default=0.0)
    return parser.parse_args()


def main():
    torch.set_float32_matmul_precision('high')

    args = parse_args()
    config = SudokuTrainConfig()
    config.epochs, config.lr, config.lr_from, config.lr_to = args.epochs, args.lr, args.lr_from, args.lr_to
    config.weight_decay, config.puzzle_emb_weight_decay = args.weight_decay, args.weight_decay
    config.warmup_steps, config.global_batch_size = args.warmup_steps, args.batch_size
    config.eval_interval, config.checkpoint_interval = args.eval_interval, args.checkpoint_interval
    config.seed, config.data_path, config.checkpoint_dir = args.seed, args.data_path, args.checkpoint_dir
    config.use_ema, config.ema_decay = not args.no_ema, args.ema_decay
    config.gradient_accumulation_steps = args.gradient_accumulation_steps
    config.use_bf16 = not args.no_bf16
    config.muon_momentum = args.muon_momentum
    config.muon_ns_steps = args.muon_ns_steps
    config.grad_clip = args.grad_clip
    config.tta_num_augments = args.tta_augments
    config.use_sudoku_struct = not args.no_sudoku_struct
    config.use_loop_gate = not args.no_loop_gate
    config.use_constraint_bias = not args.no_constraint_bias
    # Regularization
    config.attn_dropout = args.attn_dropout
    config.mlp_dropout = args.mlp_dropout
    config.loop_noise_std = args.loop_noise_std
    config.constraint_loss_weight = args.constraint_loss_weight

    print("=" * 60 + f"\nURM Sudoku Training (v3 - World Model)\n" + "=" * 60)
    print(f"Steps: {config.epochs}")
    print(f"LR Schedule: {'Linear ' + str(config.lr_from) + ' -> ' + str(config.lr_to) if config.lr_from and config.lr_to else 'Cosine with base LR=' + str(config.lr)}")
    print(f"Weight Decay: {config.weight_decay}, Batch Size: {config.global_batch_size}, EMA: {config.use_ema} (decay={config.ema_decay})")
    print(f"H_cycles: {config.H_cycles}, L_cycles: {config.L_cycles}, Loops: {config.loops}")
    print(f"Gradient Accumulation: {config.gradient_accumulation_steps}, Grad Clip: {config.grad_clip}")
    print(f"BF16: {config.use_bf16}")
    print(f"--- Architecture ---")
    print(f"Sudoku Structural Encoding: {config.use_sudoku_struct}")
    print(f"Loop Gate: {config.use_loop_gate}")
    print(f"Constraint Attention Bias: {config.use_constraint_bias}")
    print(f"--- Regularization (Anti-Overfitting) ---")
    print(f"Attention Dropout: {config.attn_dropout}")
    print(f"MLP Dropout: {config.mlp_dropout}")
    print(f"Loop Noise Std: {config.loop_noise_std}")
    print(f"Constraint Loss Weight: {config.constraint_loss_weight} (World Model)")
    print(f"--- Evaluation ---")
    print(f"TTA Augments: {config.tta_num_augments}, Loops: {config.loops}")
    print("=" * 60)

    for d in ["results", config.checkpoint_dir, config.graphs_dir]:
        os.makedirs(d, exist_ok=True)
    torch.manual_seed(config.seed)

    print("Loading dataset...")
    dataset = SudokuDataset(config, split="train")
    dataloader = DataLoader(dataset, batch_size=None, num_workers=0)
    config.vocab_size = dataset.metadata.vocab_size
    config.seq_len = dataset.metadata.seq_len
    config.num_puzzle_identifiers = dataset.metadata.num_puzzle_identifiers

    print("Creating model...")
    model = URM(config).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} (trainable: {trainable_params:,})")

    checkpoint, start_step = load_latest_checkpoint(config)
    if checkpoint is not None:
        print(f"Resuming from step {start_step}...")
        model_state = model.state_dict()
        loaded_state = checkpoint["model_state_dict"]
        matched, skipped = 0, 0
        for k, v in loaded_state.items():
            if k in model_state and model_state[k].shape == v.shape:
                model_state[k] = v
                matched += 1
            else:
                skipped += 1
        model.load_state_dict(model_state)
        if skipped > 0:
            print(f"Loaded {matched} params, skipped {skipped} (new components initialized fresh)")
        else:
            print(f"Loaded all {matched} params successfully")
    else:
        start_step = 0
        print("Starting fresh training...")

    ema = EMA(model, decay=config.ema_decay) if config.use_ema else None
    if config.use_ema:
        print(f"EMA enabled with decay={config.ema_decay}")
    if checkpoint is not None and ema and "ema_state_dict" in checkpoint:
        try:
            ema.load_state_dict(checkpoint["ema_state_dict"])
        except Exception:
            print("EMA state mismatch, reinitializing EMA")

    if not args.no_compile:
        print("Compiling model...")
        model = torch.compile(model)

    model_for_params = model._orig_mod if hasattr(model, '_orig_mod') else model
    muon_params, adamw_params = get_parameter_groups(model_for_params, config)

    print(f"Muon parameters: {sum(p.numel() for p in muon_params):,} ({len(muon_params)} tensors)")
    print(f"AdamW parameters: {sum(p.numel() for p in adamw_params):,} ({len(adamw_params)} tensors)")

    optimizer = torch.optim.Muon(
        muon_params, lr=config.lr, weight_decay=config.weight_decay,
        momentum=config.muon_momentum, nesterov=True, ns_steps=config.muon_ns_steps,
        adjust_lr_fn="match_rms_adamw"
    )
    adamw_optimizer = torch.optim.AdamW(
        adamw_params, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.95)
    ) if adamw_params else None

    puzzle_emb_params = model_for_params.get_puzzle_emb_params()
    puzzle_optimizer = CastedSparseEmbeddingSignSGD(puzzle_emb_params, lr=config.puzzle_emb_lr, weight_decay=config.puzzle_emb_weight_decay) if puzzle_emb_params else None

    if checkpoint is not None:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        except Exception:
            print("Optimizer state mismatch, reinitializing")
        if adamw_optimizer and "adamw_optimizer_state_dict" in checkpoint:
            try:
                adamw_optimizer.load_state_dict(checkpoint["adamw_optimizer_state_dict"])
            except Exception:
                print("AdamW optimizer state mismatch, reinitializing")
        if puzzle_optimizer and "puzzle_optimizer_state_dict" in checkpoint:
            try:
                puzzle_optimizer.load_state_dict(checkpoint["puzzle_optimizer_state_dict"])
            except Exception:
                print("Puzzle optimizer state mismatch, reinitializing")

    device_type = 'cuda' if 'cuda' in config.device else 'cpu'
    use_amp = config.use_bf16 and device_type == 'cuda'
    autocast_ctx = torch.autocast(device_type=device_type, dtype=torch.bfloat16) if use_amp else nullcontext()
    if use_amp:
        print("Using BF16 mixed precision training")

    print("Starting training...")
    model.train()
    step = start_step
    loss_history = checkpoint.get("loss_history", []) if checkpoint else []
    accuracy_history = checkpoint.get("accuracy_history", []) if checkpoint else []
    best_accuracy = max([h[1] for h in accuracy_history], default=0.0)
    if best_accuracy > 0:
        print(f"Best accuracy so far: {best_accuracy*100:.1f}%")

    carry, pbar, train_iter = None, tqdm(total=config.epochs, initial=start_step, desc="Training"), iter(dataloader)
    accumulation_counter = 0
    accumulated_loss = 0.0
    accumulated_constraint_loss = 0.0

    try:
        while step < config.epochs:
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(dataloader)
                batch = next(train_iter)
            batch = {k: v.to(config.device) for k, v in batch.items()}
            if carry is None:
                carry = model.initial_carry(batch)

            lr = get_lr(step, config)
            for g in optimizer.param_groups:
                g['lr'] = lr
            if adamw_optimizer:
                for g in adamw_optimizer.param_groups:
                    g['lr'] = lr

            with autocast_ctx:
                carry, outputs = model(carry, batch, compute_target_q=False)
                labels = carry.current_data["labels"]
                mask = carry.current_data["labels"] != IGNORE_LABEL_ID
                loss_divisor = mask.sum(-1).clamp_min(1).unsqueeze(-1)

                # Main loss: cross-entropy on predictions
                lm_loss = (stablemax_cross_entropy(outputs["logits"], labels, ignore_index=IGNORE_LABEL_ID) / loss_divisor).sum()

                # Q-halt loss
                preds = torch.argmax(outputs["logits"], dim=-1)
                seq_is_correct = (mask & (preds == labels)).sum(-1) == mask.sum(-1)
                q_halt_loss = F.binary_cross_entropy_with_logits(outputs["q_halt_logits"], seq_is_correct.to(outputs["q_halt_logits"].dtype), reduction="sum")

                # World Model: Constraint loss (teaches Sudoku rules)
                c_loss = sudoku_constraint_loss(outputs["logits"]) if config.constraint_loss_weight > 0 else torch.tensor(0.0, device=config.device)

                total_loss = lm_loss + 0.5 * q_halt_loss + config.constraint_loss_weight * c_loss

            scaled_loss = total_loss / (config.global_batch_size * config.gradient_accumulation_steps)
            scaled_loss.backward()

            accumulated_loss += total_loss.item() / config.global_batch_size
            accumulated_constraint_loss += c_loss.item()
            accumulation_counter += 1
            carry.current_hidden = carry.current_hidden.detach()

            if accumulation_counter >= config.gradient_accumulation_steps:
                if config.grad_clip > 0:
                    all_params = list(muon_params) + list(adamw_params)
                    if puzzle_emb_params:
                        all_params += [p for p in puzzle_emb_params if p.requires_grad]
                    torch.nn.utils.clip_grad_norm_(all_params, config.grad_clip)

                optimizer.step()
                optimizer.zero_grad()
                if adamw_optimizer:
                    adamw_optimizer.step()
                    adamw_optimizer.zero_grad()
                if puzzle_optimizer:
                    puzzle_optimizer.step()
                if ema:
                    ema.update()

                step += 1
                pbar.update(1)
                current_loss = accumulated_loss / config.gradient_accumulation_steps
                current_c_loss = accumulated_constraint_loss / config.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{current_loss:.4f}", "c_loss": f"{current_c_loss:.2f}", "lr": f"{lr:.2e}"})

                if step % config.eval_interval == 0:
                    loss_history.append((step, current_loss))
                    pass_at_k = evaluate_model(model, config, step, ema=ema)
                    accuracy_history.append((step, pass_at_k))
                    is_best = pass_at_k > best_accuracy
                    if is_best:
                        best_accuracy = pass_at_k
                    tqdm.write(
                        f"Step {step}: Loss={current_loss:.4f}, C_Loss={current_c_loss:.2f}, "
                        f"TTA({config.tta_num_augments})@{config.loops}={pass_at_k*100:.1f}%, "
                        f"Best={best_accuracy*100:.1f}%, LR={lr:.2e}"
                        + (" ★ NEW BEST" if is_best else "")
                    )
                    # Save best model
                    if is_best:
                        save_checkpoint(model, optimizer, adamw_optimizer, puzzle_optimizer,
                                        ema, step, config, loss_history, accuracy_history, tag="best")

                if step % config.checkpoint_interval == 0:
                    save_checkpoint(model, optimizer, adamw_optimizer, puzzle_optimizer,
                                    ema, step, config, loss_history, accuracy_history)
                    save_training_graphs(loss_history, accuracy_history, step, config)

                accumulation_counter = 0
                accumulated_loss = 0.0
                accumulated_constraint_loss = 0.0

    except KeyboardInterrupt:
        print("\nTraining interrupted.")

    print(f"Training complete! Best accuracy: {best_accuracy*100:.1f}%")


if __name__ == "__main__":
    main()