"""Universal Reasoning Model (URM) — Core Architecture
Domain-agnostic recursive latent-space reasoning model.
Includes generic StructuralEncoding and TTA framework.
"""
import math
from dataclasses import dataclass, replace
from typing import Dict, Optional, List, Callable
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, ParamsT

IGNORE_LABEL_ID = -100

# ── Config ────────────────────────────────────────────────

@dataclass
class URMConfig:
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4
    num_layers: int = 4
    loops: int = 8
    eval_loops: int = 8
    H_cycles: int = 2
    L_cycles: int = 6
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    loop_noise_std: float = 0.02
    mid_loop_loss_weight: float = 0.1
    forward_dtype: str = "bfloat16"
    vocab_size: int = 11
    seq_len: int = 81
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 512
    global_batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ── Utilities ─────────────────────────────────────────────

def trunc_normal_init_(t: torch.Tensor, std: float = 1.0):
    with torch.no_grad():
        if std == 0: return t.zero_()
        s2 = math.sqrt(2)
        a, b = math.erf(-2 / s2), math.erf(2 / s2)
        z = (b - a) / 2
        c = (2 * math.pi) ** -0.5
        pu = pl = c * math.exp(-2)
        cs = std / math.sqrt(1 - (2*pu - (-2)*pl)/z - ((pu-pl)/z)**2)
        t.uniform_(a, b).erfinv_().mul_(s2 * cs).clip_(-2*cs, 2*cs)
    return t

def _ceil_multiple(a, b):
    return (-(a // -b)) * b

def rms_norm(x, eps):
    xf = x.float()
    return (xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)).to(x.dtype)

def _s(x):
    return torch.where(x < 0, 1 / (1 - x + 1e-30), x + 1)

def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    sx = _s(logits.to(torch.float64))
    lp = torch.log(sx / sx.sum(-1, keepdim=True))
    mask = labels != ignore_index
    safe = torch.where(mask, labels, 0)
    return -torch.where(mask, lp.gather(-1, safe.long().unsqueeze(-1)).squeeze(-1), 0)

def _rotate_half(x):
    return torch.cat((-x[..., x.shape[-1]//2:], x[..., :x.shape[-1]//2]), -1)

def _apply_rotary(q, k, cos, sin):
    d = q.dtype; q, k = q.to(cos.dtype), k.to(cos.dtype)
    qe = q * cos.unsqueeze(-2) + _rotate_half(q) * sin.unsqueeze(-2)
    ke = k * cos.unsqueeze(-2) + _rotate_half(k) * sin.unsqueeze(-2)
    return qe.to(d), ke.to(d)

# ── Structural Encoding (Generic) ────────────────────────

class StructuralEncoding(nn.Module):
    """Learned positional encoding from arbitrary group memberships."""
    def __init__(self, group_indices: List[torch.Tensor], num_groups: List[int],
                 hidden_size: int, dtype):
        super().__init__()
        assert len(group_indices) == len(num_groups)
        self.dtype = dtype
        emb_dim = min(64, hidden_size // 4)
        self.group_embs = nn.ModuleList([nn.Embedding(n, emb_dim) for n in num_groups])
        self.proj = nn.Linear(emb_dim * len(num_groups), hidden_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))
        for i, idx in enumerate(group_indices):
            self.register_buffer(f'gidx_{i}', idx)

    def forward(self):
        parts = [emb(getattr(self, f'gidx_{i}')) for i, emb in enumerate(self.group_embs)]
        return (self.scale * self.proj(torch.cat(parts, -1))).to(self.dtype)

# ── TTA Framework (Generic) ──────────────────────────────

class TTAEnsemble:
    """Generic test-time augmentation with weighted majority voting."""
    @staticmethod
    def _run_inference(model, batch, num_loops):
        carry = model.initial_carry(batch)
        for _ in range(num_loops):
            carry, out = model(carry, batch, compute_target_q=False)
        return torch.argmax(out["logits"][0], dim=-1)

    @staticmethod
    def _weight_from_violations(v: int) -> float:
        return 5.0 if v == 0 else max(0.5, 3.0 - 0.15 * v)

    @staticmethod
    @torch.no_grad()
    def predict(model, single_batch: dict, num_loops: int, num_augments: int,
                vocab_size: int, transform_fn: Callable, apply_fn: Callable,
                decode_fn: Callable, violation_fn: Callable,
                device: torch.device) -> torch.Tensor:
        seq_len = single_batch["inputs"].shape[1]
        all_preds, weights = [], []
        pred = TTAEnsemble._run_inference(model, single_batch, num_loops)
        all_preds.append(pred)
        w = TTAEnsemble._weight_from_violations(violation_fn(decode_fn(pred.cpu().numpy())))
        weights.append(w)
        for _ in range(num_augments):
            t = transform_fn()
            aug_inp = apply_fn(single_batch["inputs"][0], t, device, inverse=False).unsqueeze(0)
            aug_batch = {**single_batch, "inputs": aug_inp}
            aug_pred = TTAEnsemble._run_inference(model, aug_batch, num_loops)
            orig_pred = apply_fn(aug_pred, t, device, inverse=True)
            all_preds.append(orig_pred)
            w = TTAEnsemble._weight_from_violations(violation_fn(decode_fn(orig_pred.cpu().numpy())))
            weights.append(w)
        final = torch.zeros(seq_len, dtype=torch.long, device=device)
        for pos in range(seq_len):
            votes = torch.zeros(vocab_size, device=device)
            for k, p in enumerate(all_preds):
                votes[p[pos].long()] += weights[k]
            final[pos] = votes.argmax()
        return final

    @staticmethod
    @torch.no_grad()
    def evaluate_batch(model, batch: dict, eval_loops: int, num_augments: int,
                       vocab_size: int, transform_fn: Callable, apply_fn: Callable,
                       decode_fn: Callable, violation_fn: Callable,
                       device: torch.device, max_eval: int = 64) -> float:
        B = min(batch["inputs"].shape[0], max_eval)
        correct = 0
        for b in range(B):
            labels = batch["labels"][b]
            mask = labels != IGNORE_LABEL_ID
            if mask.sum() == 0: continue
            sb = {k: v[b:b+1] for k, v in batch.items()}
            pred = TTAEnsemble.predict(model, sb, eval_loops, num_augments,
                                       vocab_size, transform_fn, apply_fn,
                                       decode_fn, violation_fn, device)
            if ((pred == labels) | ~mask).all():
                correct += 1
        return correct / B if B > 0 else 0.0

# ── Core Modules ──────────────────────────────────────────

class CastedLinear(nn.Module):
    def __init__(self, in_f, out_f, bias=False):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(out_f, in_f), std=in_f**-0.5))
        self.bias = nn.Parameter(torch.zeros(out_f)) if bias else None
    def forward(self, x):
        return F.linear(x, self.weight.to(x.dtype), self.bias.to(x.dtype) if self.bias is not None else None)

class CastedEmbedding(nn.Module):
    def __init__(self, num, dim, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty(num, dim), std=init_std))
    def forward(self, x):
        return F.embedding(x, self.weight.to(self.cast_to))

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_pos, base):
        super().__init__()
        inv = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        freqs = torch.outer(torch.arange(max_pos, dtype=torch.float32), inv)
        emb = torch.cat((freqs, freqs), -1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)
    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, num_heads, dropout=0.0):
        super().__init__()
        self.num_heads, self.head_dim = num_heads, hidden_size // num_heads
        self.qkv = CastedLinear(hidden_size, 3 * hidden_size)
        self.o_proj = CastedLinear(hidden_size, hidden_size)
        self.dropout = dropout
    def forward(self, cos_sin, x):
        B, S, _ = x.shape
        qkv = self.qkv(x).view(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(2)
        if cos_sin is not None:
            q, k = _apply_rotary(q, k, *cos_sin)
        o = F.scaled_dot_product_attention(q.transpose(1,2), k.transpose(1,2), v.transpose(1,2),
                                           dropout_p=self.dropout if self.training else 0.0)
        return self.o_proj(o.transpose(1,2).reshape(B, S, -1))

class ConvSwiGLU(nn.Module):
    def __init__(self, hidden_size, expansion, conv_kernel=3):
        super().__init__()
        inter = _ceil_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up = CastedLinear(hidden_size, inter * 2)
        self.dwconv = nn.Conv1d(inter, inter, conv_kernel, padding=conv_kernel//2,
                                groups=inter, bias=True).to(dtype=torch.bfloat16)
        self.act = nn.SiLU()
        self.down = CastedLinear(inter, hidden_size)
    def forward(self, x):
        gate, up = self.gate_up(x).chunk(2, dim=-1)
        h = F.silu(gate) * up
        h = self.act(self.dwconv(h.transpose(1,2).to(self.dwconv.weight.dtype))[..., :up.size(1)])
        return self.down(h.transpose(1,2).contiguous())

class LoopGate(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.proj = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        with torch.no_grad():
            self.proj.bias.fill_(3.0); self.proj.weight.mul_(0.01)
    def forward(self, old_h, new_h):
        g = torch.sigmoid(self.proj(torch.cat([old_h, new_h], -1).to(self.proj.weight.dtype)))
        return g.to(new_h.dtype) * new_h + (1 - g.to(old_h.dtype)) * old_h

class LoopIterEmbedding(nn.Module):
    def __init__(self, hidden_size, max_loops=64, dtype=torch.bfloat16):
        super().__init__()
        self.dtype = dtype
        self.emb = nn.Embedding(max_loops, hidden_size)
        self.scale = nn.Parameter(torch.tensor(0.02))
        nn.init.normal_(self.emb.weight, std=0.02)
    def forward(self, idx: int):
        i = torch.tensor([min(idx, self.emb.num_embeddings-1)], device=self.emb.weight.device)
        return (self.scale * self.emb(i)).to(self.dtype)

# ── Model ─────────────────────────────────────────────────

@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None
    loop_idx: int = 0

class URMBlock(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        self.attn = Attention(config.hidden_size, config.num_heads, config.attn_dropout)
        self.mlp = ConvSwiGLU(config.hidden_size, config.expansion)
        self.eps = config.rms_norm_eps
        self.mlp_drop = nn.Dropout(config.mlp_dropout)
    def forward(self, cos_sin, x):
        x = rms_norm(x + self.attn(cos_sin, x), self.eps)
        return rms_norm(x + self.mlp_drop(self.mlp(x)), self.eps)

class URMInner(nn.Module):
    def __init__(self, config: URMConfig, structural_encoding: Optional[nn.Module] = None):
        super().__init__()
        self.config, self.structural_encoding = config, structural_encoding
        self.fwd_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size,
                                            1.0/self.embed_scale, self.fwd_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)
        with torch.no_grad(): self.q_head.weight.zero_(); self.q_head.bias.fill_(-5)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim,
                                                     config.global_batch_size, 0, self.fwd_dtype)
        total_seq = config.seq_len + self.puzzle_emb_len
        self.rotary = RotaryEmbedding(config.hidden_size // config.num_heads, total_seq, config.rope_theta)
        self.layers = nn.ModuleList([URMBlock(config) for _ in range(config.num_layers)])
        self.register_buffer("init_hidden", trunc_normal_init_(
            torch.empty(config.hidden_size, dtype=self.fwd_dtype), std=1), persistent=True)
        self.loop_gate = LoopGate(config.hidden_size)
        self.loop_iter_emb = LoopIterEmbedding(config.hidden_size, dtype=self.fwd_dtype)

    def _input_embeddings(self, inp, puzzle_ids):
        emb = self.embed_tokens(inp.to(torch.int32))
        if self.structural_encoding is not None:
            emb = emb + self.structural_encoding().unsqueeze(0)
        if self.config.puzzle_emb_ndim > 0:
            pe = self.puzzle_emb(puzzle_ids)
            pad = self.puzzle_emb_len * self.config.hidden_size - pe.shape[-1]
            if pad > 0: pe = F.pad(pe, (0, pad))
            emb = torch.cat((pe.view(-1, self.puzzle_emb_len, self.config.hidden_size), emb), dim=-2)
        return self.embed_scale * emb

    def empty_carry(self, B):
        S = self.config.seq_len + self.puzzle_emb_len
        return URMCarry(current_hidden=torch.empty(B, S, self.config.hidden_size,
                        dtype=self.fwd_dtype, device=self.init_hidden.device))

    def reset_carry(self, flag, carry):
        return replace(carry, current_hidden=torch.where(flag.view(-1,1,1), self.init_hidden, carry.current_hidden))

    def _run_layers(self, h, inp_emb, cos_sin):
        h = h + inp_emb
        for layer in self.layers: h = layer(cos_sin, h)
        return h

    def forward(self, carry, batch):
        cos_sin = self.rotary()
        inp_emb = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        inp_emb = inp_emb + self.loop_iter_emb(carry.loop_idx)
        h, old_h, mid_logits = carry.current_hidden, carry.current_hidden, None

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range((self.config.H_cycles - 1) * self.config.L_cycles):
                    h = self._run_layers(h, inp_emb, cos_sin)

        total_l, mid = self.config.L_cycles, self.config.L_cycles // 2
        for lc in range(total_l):
            h = self._run_layers(h, inp_emb, cos_sin)
            if lc == mid - 1 and self.training and self.config.mid_loop_loss_weight > 0:
                mid_logits = self.lm_head(h)[:, self.puzzle_emb_len:]

        h = self.loop_gate(old_h, h)
        if self.training and self.config.loop_noise_std > 0:
            h = h + torch.randn_like(h) * self.config.loop_noise_std

        q = self.q_head(h[:, 0]).to(torch.float32)
        logits = self.lm_head(h)[:, self.puzzle_emb_len:]
        return replace(carry, current_hidden=h.detach()), logits, (q[...,0], q[...,1]), mid_logits

class URM(nn.Module):
    def __init__(self, config: URMConfig, structural_encoding: Optional[nn.Module] = None):
        super().__init__()
        self.config = config
        self.inner = URMInner(config, structural_encoding)

    def initial_carry(self, batch):
        B, dev = batch["inputs"].shape[0], batch["inputs"].device
        base = self.inner.empty_carry(B)
        return URMCarry(base.current_hidden, torch.zeros(B, dtype=torch.int32, device=dev),
                        torch.ones(B, dtype=torch.bool, device=dev),
                        {k: torch.empty_like(v) for k, v in batch.items()}, 0)

    def forward(self, carry, batch, compute_target_q=False):
        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_lidx = 0 if carry.halted.all() else carry.loop_idx
        new_data = {k: torch.where(carry.halted.view((-1,)+(1,)*(batch[k].ndim-1)), batch[k], v)
                    for k, v in carry.current_data.items()}
        new_carry = replace(new_carry, loop_idx=new_lidx)
        new_carry, logits, (qh, qc), mid = self.inner(new_carry, new_data)
        with torch.no_grad():
            new_steps = new_steps + 1
            halted = new_steps >= self.config.loops
            if self.training and self.config.loops > 1:
                halted = halted | (qh > 0)
                mh = (torch.rand_like(qh) < 0.1) * torch.randint_like(new_steps, low=2, high=self.config.loops+1)
                halted = halted & (new_steps >= mh)
        return (URMCarry(new_carry.current_hidden, new_steps, halted, new_data, new_lidx+1),
                {"logits": logits, "q_halt_logits": qh, "q_continue_logits": qc, "mid_logits": mid})

    def get_puzzle_emb_params(self):
        if self.config.puzzle_emb_ndim > 0:
            e = self.inner.puzzle_emb
            return [e.weights, e.local_ids, e.local_weights]
        return []

# ── Sparse Embedding + Optimizer ──────────────────────────

class CastedSparseEmbedding(nn.Module):
    def __init__(self, num, dim, batch_size, init_std, cast_to):
        super().__init__()
        self.cast_to = cast_to
        self.register_buffer("weights", trunc_normal_init_(torch.empty(num, dim), std=init_std), persistent=True)
        self.local_weights = nn.Parameter(torch.zeros(batch_size, dim))
        self.register_buffer("local_ids", torch.zeros(batch_size, dtype=torch.int64), persistent=False)
    def forward(self, x):
        if not self.training: return self.weights[x].to(self.cast_to)
        with torch.no_grad(): self.local_weights.copy_(self.weights[x]); self.local_ids.copy_(x)
        return self.local_weights.to(self.cast_to)

class SparseSignSGD(Optimizer):
    def __init__(self, params: ParamsT, lr=1e-3, weight_decay=1e-2):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))
    @torch.no_grad()
    def step(self, closure=None):
        for g in self.param_groups:
            lw_grad, ids, weights = None, None, None
            for p in g["params"]:
                if p.requires_grad: lw_grad = p.grad
                elif p.ndim == 1: ids = p
                elif p.ndim == 2: weights = p
            if lw_grad is not None:
                uid, inv = ids.unique(return_inverse=True)
                grad = torch.zeros(uid.shape[0], lw_grad.shape[1], dtype=lw_grad.dtype, device=lw_grad.device)
                grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, lw_grad.shape[1]), lw_grad)
                w = weights[uid]
                w.mul_(1 - g["lr"] * g["weight_decay"]).add_(grad.sign(), alpha=-g["lr"])
                weights[uid] = w

# ── EMA ───────────────────────────────────────────────────

class EMA:
    def __init__(self, model: nn.Module, decay=0.999):
        self.model, self.decay = model, decay
        self.shadow = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        self.backup = {}
    @torch.no_grad()
    def update(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.shadow[n].mul_(self.decay).add_(p.data, alpha=1-self.decay)
    def apply_shadow(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad and n in self.shadow:
                self.backup[n] = p.data.clone(); p.data.copy_(self.shadow[n])
    def restore(self):
        for n, p in self.model.named_parameters():
            if n in self.backup: p.data.copy_(self.backup[n])
        self.backup = {}
    def state_dict(self):
        return {n: t.clone() for n, t in self.shadow.items()}
    def load_state_dict(self, sd):
        self.shadow = {n: t.clone() for n, t in sd.items()}