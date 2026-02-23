"""Universal Reasoning Model (URM) - Enhanced Core Architecture for Sudoku
v3 Changes:
1. Sudoku Constraint Loss (World Model) - differentiable penalty for rule violations
2. Dropout in attention + MLP (recurrent regularization)
3. Loop Noise - Gaussian noise on hidden state between segments
4. SudokuStructuralEncoding, Loop Gate, Constraint Attention Bias (from v2)
"""
import math
from dataclasses import dataclass, replace
from typing import Tuple, Dict, Optional, List
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.optimizer import Optimizer, ParamsT

IGNORE_LABEL_ID = -100

@dataclass
class URMConfig:
    hidden_size: int = 512
    num_heads: int = 8
    expansion: float = 4
    num_layers: int = 4
    loops: int = 8
    H_cycles: int = 2
    L_cycles: int = 6
    rms_norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    attn_dropout: float = 0.1
    mlp_dropout: float = 0.1
    pos_encodings: str = "rope"
    forward_dtype: str = "bfloat16"
    vocab_size: int = 11
    seq_len: int = 81
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 512
    global_batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # --- Architecture enhancements ---
    use_sudoku_struct: bool = True
    use_loop_gate: bool = True
    use_constraint_bias: bool = True
    # --- Regularization ---
    loop_noise_std: float = 0.02
    constraint_loss_weight: float = 0.0
    # --- TTA ---
    tta_num_augments: int = 8


def trunc_normal_init_(tensor: torch.Tensor, std: float = 1.0, lower: float = -2.0, upper: float = 2.0):
    with torch.no_grad():
        if std == 0:
            tensor.zero_()
        else:
            sqrt2 = math.sqrt(2)
            a, b = math.erf(lower / sqrt2), math.erf(upper / sqrt2)
            z = (b - a) / 2
            c = (2 * math.pi) ** -0.5
            pdf_u, pdf_l = c * math.exp(-0.5 * lower ** 2), c * math.exp(-0.5 * upper ** 2)
            comp_std = std / math.sqrt(1 - (upper * pdf_u - lower * pdf_l) / z - ((pdf_u - pdf_l) / z) ** 2)
            tensor.uniform_(a, b).erfinv_().mul_(sqrt2 * comp_std).clip_(lower * comp_std, upper * comp_std)
    return tensor


def _find_multiple(a, b):
    return (-(a // -b)) * b


try:
    from flash_attn import flash_attn_func as _flash_attn_func
    HAS_FLASH = True
except ImportError:
    HAS_FLASH = False

CosSin = Tuple[torch.Tensor, torch.Tensor]


def rotate_half(x: torch.Tensor):
    return torch.cat((-x[..., x.shape[-1] // 2:], x[..., :x.shape[-1] // 2]), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    orig_dtype = q.dtype
    q, k = q.to(cos.dtype), k.to(cos.dtype)
    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))
    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)
    variance = hidden_states.square().mean(-1, keepdim=True)
    return (hidden_states * torch.rsqrt(variance + variance_epsilon)).to(input_dtype)


# ──────────────────────────────────────────────────────────
# World Model: Differentiable Sudoku Constraint Loss
# ──────────────────────────────────────────────────────────

def sudoku_constraint_loss(logits):
    """Differentiable Sudoku constraint violation loss (World Model).

    Instead of just learning input→output mapping, this teaches the model
    the RULES of Sudoku: each digit 1-9 must appear at most once per
    row, column, and 3x3 block.

    Uses soft-counts from logit probabilities so gradients flow through.
    Penalty = Σ ReLU(soft_count(digit, unit) - 1) for all units.

    Args:
        logits: [B, 81, vocab_size] model output logits
    Returns:
        scalar loss (mean over batch)
    """
    B = logits.shape[0]

    # Soft probabilities for digits 1-9 (tokens 2-10)
    probs = F.softmax(logits.float(), dim=-1)[:, :, 2:11]  # [B, 81, 9]
    probs = probs.view(B, 9, 9, 9)  # [B, row, col, digit]

    penalty = torch.tensor(0.0, device=logits.device, dtype=torch.float32)

    # Row constraint: each digit appears at most once per row
    # Sum probabilities across columns for each (row, digit)
    row_counts = probs.sum(dim=2)  # [B, 9_rows, 9_digits]
    penalty = penalty + F.relu(row_counts - 1.0).sum()

    # Column constraint: each digit appears at most once per column
    col_counts = probs.sum(dim=1)  # [B, 9_cols, 9_digits]
    penalty = penalty + F.relu(col_counts - 1.0).sum()

    # Block constraint: each digit appears at most once per 3x3 block
    for br in range(3):
        for bc in range(3):
            block = probs[:, br*3:(br+1)*3, bc*3:(bc+1)*3, :]  # [B, 3, 3, 9]
            block_counts = block.sum(dim=(1, 2))  # [B, 9_digits]
            penalty = penalty + F.relu(block_counts - 1.0).sum()

    return penalty / B


# ──────────────────────────────────────────────────────────
# Sudoku structural encoding
# ──────────────────────────────────────────────────────────

def build_sudoku_constraint_mask(seq_len=81):
    """Build boolean mask: mask[i,j]=True if cells i,j share a row, col, or block."""
    mask = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    for i in range(seq_len):
        ri, ci = i // 9, i % 9
        bi = (ri // 3) * 3 + ci // 3
        for j in range(seq_len):
            if i == j:
                continue
            rj, cj = j // 9, j % 9
            bj = (rj // 3) * 3 + cj // 3
            if ri == rj or ci == cj or bi == bj:
                mask[i, j] = True
    return mask


class SudokuStructuralEncoding(nn.Module):
    """Adds Sudoku grid structure info (row, column, block) as learned embeddings."""
    def __init__(self, hidden_size: int, forward_dtype: torch.dtype):
        super().__init__()
        self.forward_dtype = forward_dtype
        emb_dim = min(64, hidden_size // 4)
        self.row_emb = nn.Embedding(9, emb_dim)
        self.col_emb = nn.Embedding(9, emb_dim)
        self.block_emb = nn.Embedding(9, emb_dim)
        self.proj = nn.Linear(emb_dim * 3, hidden_size, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.1))

        rows = torch.arange(81) // 9
        cols = torch.arange(81) % 9
        blocks = (rows // 3) * 3 + cols // 3
        self.register_buffer('row_idx', rows)
        self.register_buffer('col_idx', cols)
        self.register_buffer('block_idx', blocks)

    def forward(self) -> torch.Tensor:
        r = self.row_emb(self.row_idx)
        c = self.col_emb(self.col_idx)
        b = self.block_emb(self.block_idx)
        out = self.proj(torch.cat([r, c, b], dim=-1))
        return (self.scale * out).to(self.forward_dtype)


# ──────────────────────────────────────────────────────────
# Core modules
# ──────────────────────────────────────────────────────────

class CastedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool):
        super().__init__()
        self.weight = nn.Parameter(trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5)))
        self.bias = nn.Parameter(torch.zeros((out_features,))) if bias else None

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to
        self.embedding_weight = nn.Parameter(trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings, base, device=None):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        emb = torch.cat((freqs := torch.outer(t, inv_freq), freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached


class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads,
                 causal=False, attn_dropout=0.0, use_constraint_bias=False, total_seq_len=82):
        super().__init__()
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.output_size = head_dim * num_heads
        self.num_key_value_heads, self.causal = num_key_value_heads, causal
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)
        self.use_constraint_bias = use_constraint_bias
        self.scale = 1.0 / math.sqrt(head_dim)

        if use_constraint_bias:
            self.constraint_bias = nn.Parameter(torch.zeros(num_heads))
            sudoku_mask = build_sudoku_constraint_mask(81)
            self.register_buffer('_sudoku_mask_81', sudoku_mask.float())
            self._padded_mask = None
            self._padded_len = 0

    def _get_constraint_bias(self, S: int) -> torch.Tensor:
        if not self.use_constraint_bias:
            return None
        prefix = S - 81
        if self._padded_mask is None or self._padded_len != S:
            full = torch.zeros(S, S, device=self._sudoku_mask_81.device)
            full[prefix:, prefix:] = self._sudoku_mask_81
            self._padded_mask = full
            self._padded_len = S
        bias = self.constraint_bias.view(-1, 1, 1) * self._padded_mask.unsqueeze(0)
        return bias.unsqueeze(0)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        q = qkv[:, :, :self.num_heads]
        k = qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads]
        v = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, *cos_sin)

        if self.use_constraint_bias:
            q_t = q.permute(0, 2, 1, 3)
            k_t = k.permute(0, 2, 1, 3)
            v_t = v.permute(0, 2, 1, 3)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * self.scale
            c_bias = self._get_constraint_bias(S)
            if c_bias is not None:
                scores = scores + c_bias.to(scores.dtype)
            attn = F.softmax(scores, dim=-1)
            attn_output = torch.matmul(attn, v_t).permute(0, 2, 1, 3)
        elif HAS_FLASH:
            attn_output = _flash_attn_func(q=q, k=k, v=v, causal=self.causal)
            if isinstance(attn_output, tuple):
                attn_output = attn_output[0]
        else:
            q_t, k_t, v_t = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            attn_output = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=self.causal).transpose(1, 2)

        return self.o_proj(attn_output.reshape(B, S, self.output_size))


class ConvSwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float, conv_kernel: int = 3, intermediate_size: Optional[int] = None):
        super().__init__()
        self.inter = intermediate_size if intermediate_size is not None else _find_multiple(round(expansion * hidden_size * 2 / 3), 256)
        self.gate_up_proj = CastedLinear(hidden_size, self.inter * 2, bias=False)
        self.dwconv = nn.Conv1d(self.inter, self.inter, conv_kernel, padding=conv_kernel // 2, groups=self.inter, bias=True).to(dtype=torch.bfloat16)
        self.act = nn.SiLU()
        self.down_proj = CastedLinear(self.inter, hidden_size, bias=False)

    def forward(self, x: torch.Tensor):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        x_ffn = F.silu(gate) * up
        x_conv = self.act(self.dwconv(x_ffn.transpose(1, 2).to(self.dwconv.weight.dtype))[..., :up.size(1)]).transpose(1, 2).contiguous()
        return self.down_proj(x_conv)


class CastedSparseEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, batch_size: int, init_std: float, cast_to: torch.dtype):
        super().__init__()
        self.cast_to, self.num_embeddings = cast_to, num_embeddings
        self.register_buffer("weights", trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std), persistent=True)
        self.local_weights = nn.Parameter(torch.zeros(batch_size, embedding_dim), requires_grad=True)
        self.register_buffer("local_ids", torch.zeros(batch_size, dtype=torch.int64), persistent=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return self.weights[inputs].to(self.cast_to)
        with torch.no_grad():
            self.local_weights.copy_(self.weights[inputs])
            self.local_ids.copy_(inputs)
        return self.local_weights.to(self.cast_to)


class CastedSparseEmbeddingSignSGD(Optimizer):
    def __init__(self, params: ParamsT, lr: float = 1e-3, weight_decay: float = 1e-2):
        super().__init__(params, dict(lr=lr, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            local_weights_grad, local_ids, weights = None, None, None
            for p in group["params"]:
                if p.requires_grad:
                    local_weights_grad = p.grad
                elif p.ndim == 1:
                    local_ids = p
                elif p.ndim == 2:
                    weights = p
            if local_weights_grad is not None:
                lr, wd = group["lr"], group["weight_decay"]
                grad_ids, inv = local_ids.unique(return_inverse=True)
                grad = torch.zeros((grad_ids.shape[0], local_weights_grad.shape[1]), dtype=local_weights_grad.dtype, device=local_weights_grad.device)
                grad.scatter_add_(0, inv.unsqueeze(-1).expand(-1, local_weights_grad.shape[1]), local_weights_grad)
                p_weights = weights[grad_ids]
                p_weights.mul_(1.0 - lr * wd).add_(torch.sign(grad), alpha=-lr)
                weights[grad_ids] = p_weights


def s(x, epsilon=1e-30):
    return torch.where(x < 0, 1 / (1 - x + epsilon), x + 1)


def log_stablemax(x, dim=-1):
    s_x = s(x)
    return torch.log(s_x / torch.sum(s_x, dim=dim, keepdim=True))


def stablemax_cross_entropy(logits, labels, ignore_index=-100):
    logprobs = log_stablemax(logits.to(torch.float64), dim=-1)
    valid_mask = labels != ignore_index
    transformed_labels = torch.where(valid_mask, labels, 0)
    prediction_logprobs = torch.gather(logprobs, index=transformed_labels.to(torch.long).unsqueeze(-1), dim=-1).squeeze(-1)
    return -torch.where(valid_mask, prediction_logprobs, 0)


# ──────────────────────────────────────────────────────────
# Loop Gate
# ──────────────────────────────────────────────────────────

class LoopGate(nn.Module):
    """Per-position gate controlling hidden state update between segments."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = CastedLinear(hidden_size * 2, hidden_size, bias=True)
        with torch.no_grad():
            self.proj.bias.fill_(3.0)
            self.proj.weight.mul_(0.01)

    def forward(self, old_hidden: torch.Tensor, new_hidden: torch.Tensor) -> torch.Tensor:
        gate = torch.sigmoid(self.proj(torch.cat([old_hidden, new_hidden], dim=-1).to(self.proj.weight.dtype)))
        return gate.to(new_hidden.dtype) * new_hidden + (1 - gate.to(old_hidden.dtype)) * old_hidden


# ──────────────────────────────────────────────────────────
# Model components
# ──────────────────────────────────────────────────────────

@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None


class URMBlock(nn.Module):
    def __init__(self, config: URMConfig, total_seq_len: int) -> None:
        super().__init__()
        self.self_attn = Attention(
            config.hidden_size, config.hidden_size // config.num_heads,
            config.num_heads, config.num_heads, causal=False,
            use_constraint_bias=config.use_constraint_bias,
            total_seq_len=total_seq_len
        )
        self.mlp = ConvSwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps
        # Dropout for regularization within the recurrent loop
        self.attn_drop = nn.Dropout(config.attn_dropout)
        self.mlp_drop = nn.Dropout(config.mlp_dropout)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.attn_drop(self.self_attn(cos_sin, hidden_states)), self.norm_eps)
        return rms_norm(hidden_states + self.mlp_drop(self.mlp(hidden_states)), self.norm_eps)


class URMInner(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.config = config
        self.forward_dtype = getattr(torch, config.forward_dtype)
        self.embed_scale = math.sqrt(config.hidden_size)
        self.embed_tokens = CastedEmbedding(config.vocab_size, config.hidden_size, 1.0 / self.embed_scale, self.forward_dtype)
        self.lm_head = CastedLinear(config.hidden_size, config.vocab_size, bias=False)
        self.q_head = CastedLinear(config.hidden_size, 2, bias=True)

        self.puzzle_emb_len = -(config.puzzle_emb_ndim // -config.hidden_size)
        if config.puzzle_emb_ndim > 0:
            self.puzzle_emb = CastedSparseEmbedding(config.num_puzzle_identifiers, config.puzzle_emb_ndim, config.global_batch_size, 0, self.forward_dtype)

        total_seq_len = config.seq_len + self.puzzle_emb_len
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, total_seq_len, config.rope_theta)
        self.layers = nn.ModuleList([URMBlock(config, total_seq_len) for _ in range(config.num_layers)])
        self.register_buffer("init_hidden", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)

        self.sudoku_struct = SudokuStructuralEncoding(config.hidden_size, self.forward_dtype) if config.use_sudoku_struct else None
        self.loop_gate = LoopGate(config.hidden_size) if config.use_loop_gate else None

        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
        if self.sudoku_struct is not None:
            struct_enc = self.sudoku_struct()
            embedding = embedding + struct_enc.unsqueeze(0)
        if self.config.puzzle_emb_ndim > 0:
            puzzle_embedding = self.puzzle_emb(puzzle_identifiers)
            pad_count = self.puzzle_emb_len * self.config.hidden_size - puzzle_embedding.shape[-1]
            if pad_count > 0:
                puzzle_embedding = F.pad(puzzle_embedding, (0, pad_count))
            embedding = torch.cat((puzzle_embedding.view(-1, self.puzzle_emb_len, self.config.hidden_size), embedding), dim=-2)
        return self.embed_scale * embedding

    def empty_carry(self, batch_size: int) -> URMCarry:
        return URMCarry(current_hidden=torch.empty(batch_size, self.config.seq_len + self.puzzle_emb_len, self.config.hidden_size, dtype=self.forward_dtype, device=self.init_hidden.device))

    def reset_carry(self, reset_flag: torch.Tensor, carry: URMCarry) -> URMCarry:
        return replace(carry, current_hidden=torch.where(reset_flag.view(-1, 1, 1), self.init_hidden, carry.current_hidden))

    def forward(self, carry: URMCarry, batch: Dict[str, torch.Tensor]) -> Tuple[URMCarry, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        cos_sin = self.rotary_emb()
        input_embeddings = self._input_embeddings(batch["inputs"], batch["puzzle_identifiers"])
        hidden_states = carry.current_hidden
        old_hidden = hidden_states

        if self.config.H_cycles > 1:
            with torch.no_grad():
                for _ in range((self.config.H_cycles - 1) * self.config.L_cycles):
                    hidden_states = hidden_states + input_embeddings
                    for layer in self.layers:
                        hidden_states = layer(cos_sin, hidden_states)

        for _ in range(self.config.L_cycles):
            hidden_states = hidden_states + input_embeddings
            for layer in self.layers:
                hidden_states = layer(cos_sin, hidden_states)

        # Loop gate: blend old and new hidden states
        if self.loop_gate is not None:
            hidden_states = self.loop_gate(old_hidden, hidden_states)

        # Loop noise: Gaussian noise between segments (training only)
        if self.training and self.config.loop_noise_std > 0:
            hidden_states = hidden_states + torch.randn_like(hidden_states) * self.config.loop_noise_std

        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return (
            replace(carry, current_hidden=hidden_states.detach()),
            self.lm_head(hidden_states)[:, self.puzzle_emb_len:],
            (q_logits[..., 0], q_logits[..., 1])
        )


class URM(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        self.config = config
        self.inner = URMInner(config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        B = batch["inputs"].shape[0]
        base = self.inner.empty_carry(B)
        return URMCarry(
            current_hidden=base.current_hidden,
            steps=torch.zeros((B,), dtype=torch.int32, device=batch["inputs"].device),
            halted=torch.ones((B,), dtype=torch.bool, device=batch["inputs"].device),
            current_data={k: torch.empty_like(v) for k, v in batch.items()}
        )

    def forward(self, carry: URMCarry, batch: Dict[str, torch.Tensor], compute_target_q: bool = False) -> Tuple[URMCarry, Dict[str, torch.Tensor]]:
        new_carry = self.inner.reset_carry(carry.halted, carry)
        new_steps = torch.where(carry.halted, 0, carry.steps)
        new_current_data = {k: torch.where(carry.halted.view((-1,) + (1,) * (batch[k].ndim - 1)), batch[k], v) for k, v in carry.current_data.items()}

        new_carry, logits, (q_halt_logits, q_continue_logits) = self.inner(new_carry, new_current_data)

        with torch.no_grad():
            new_steps = new_steps + 1
            halted = new_steps >= self.config.loops
            if self.training and self.config.loops > 1:
                halted = halted | (q_halt_logits > 0)
                min_halt_steps = (torch.rand_like(q_halt_logits) < 0.1) * torch.randint_like(new_steps, low=2, high=self.config.loops + 1)
                halted = halted & (new_steps >= min_halt_steps)

        return (
            URMCarry(current_hidden=new_carry.current_hidden, steps=new_steps, halted=halted, current_data=new_current_data),
            {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}
        )

    def get_puzzle_emb_params(self):
        if self.config.puzzle_emb_ndim > 0:
            emb = self.inner.puzzle_emb
            return [emb.weights, emb.local_ids, emb.local_weights]
        return []

    def get_model_params(self):
        if self.config.puzzle_emb_ndim > 0:
            puzzle_local_weights = self.inner.puzzle_emb.local_weights
            return [p for p in self.parameters() if p.requires_grad and p is not puzzle_local_weights]
        return [p for p in self.parameters() if p.requires_grad]


# ──────────────────────────────────────────────────────────
# TTA utility functions
# ──────────────────────────────────────────────────────────

def random_digit_perm():
    """Return (perm, inv_perm) as numpy arrays indexed by digit value.
    perm[d] = new digit for digit d (d=1..9), perm[0]=0 unused.
    inv_perm[d] = original digit for digit d (d=1..9)."""
    import numpy as np
    raw = np.random.permutation(9) + 1
    perm = np.zeros(10, dtype=np.int64)
    inv_perm = np.zeros(10, dtype=np.int64)
    for i in range(9):
        perm[i + 1] = raw[i]
        inv_perm[raw[i]] = i + 1
    return perm, inv_perm


def apply_token_perm(tokens: torch.Tensor, perm) -> torch.Tensor:
    """Apply digit permutation to token tensor.
    Tokens: 0=pad, 1=blank, d+1=digit d (d=1..9).
    perm[d] = new digit for digit d (indexed by digit value 1-9)."""
    mapping = torch.arange(11, device=tokens.device, dtype=tokens.dtype)
    for d in range(1, 10):
        mapping[d + 1] = perm[d] + 1
    return mapping[tokens.long()]


def is_valid_sudoku_grid(grid):
    """Check if a 9x9 grid (values 1-9) forms a valid complete Sudoku."""
    import numpy as np
    for i in range(9):
        if len(set(grid[i, :])) != 9 or set(grid[i, :]) != set(range(1, 10)):
            return False
        if len(set(grid[:, i])) != 9 or set(grid[:, i]) != set(range(1, 10)):
            return False
    for br in range(3):
        for bc in range(3):
            block = grid[br*3:(br+1)*3, bc*3:(bc+1)*3].flatten()
            if len(set(block)) != 9 or set(block) != set(range(1, 10)):
                return False
    return True