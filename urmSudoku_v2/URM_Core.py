"""Universal Reasoning Model (URM) - Core Architecture"""
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
    attn_dropout: float = 0.0
    mlp_dropout: float = 0.0
    pos_encodings: str = "rope"
    forward_dtype: str = "bfloat16"
    vocab_size: int = 11
    seq_len: int = 81
    num_puzzle_identifiers: int = 1
    puzzle_emb_ndim: int = 512
    global_batch_size: int = 128
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

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
    from flash_attn import flash_attn_func
except ImportError:
    def flash_attn_func(q, k, v, causal=False, dropout_p=0.0, **kwargs):
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=causal, dropout_p=dropout_p).transpose(1, 2)

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
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False, attn_dropout=0.0):
        super().__init__()
        self.hidden_size, self.head_dim, self.num_heads = hidden_size, head_dim, num_heads
        self.output_size = head_dim * num_heads
        self.num_key_value_heads, self.causal = num_key_value_heads, causal
        self.qkv_proj = CastedLinear(hidden_size, (num_heads + 2 * num_key_value_heads) * head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        B, S, _ = hidden_states.shape
        qkv = self.qkv_proj(hidden_states).view(B, S, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        q, k, v = qkv[:, :, :self.num_heads], qkv[:, :, self.num_heads:self.num_heads + self.num_key_value_heads], qkv[:, :, self.num_heads + self.num_key_value_heads:]
        if cos_sin is not None:
            q, k = apply_rotary_pos_emb(q, k, *cos_sin)
        attn_output = flash_attn_func(q=q, k=k, v=v, causal=self.causal)
        if isinstance(attn_output, tuple):
            attn_output = attn_output[0]
        return self.o_proj(attn_output.view(B, S, self.output_size))

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

@dataclass
class URMCarry:
    current_hidden: torch.Tensor
    steps: Optional[torch.Tensor] = None
    halted: Optional[torch.Tensor] = None
    current_data: Optional[Dict[str, torch.Tensor]] = None

class URMBlock(nn.Module):
    def __init__(self, config: URMConfig) -> None:
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.hidden_size // config.num_heads, config.num_heads, config.num_heads, causal=False)
        self.mlp = ConvSwiGLU(config.hidden_size, config.expansion)
        self.norm_eps = config.rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = rms_norm(hidden_states + self.self_attn(cos_sin, hidden_states), self.norm_eps)
        return rms_norm(hidden_states + self.mlp(hidden_states), self.norm_eps)

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
        self.rotary_emb = RotaryEmbedding(config.hidden_size // config.num_heads, config.seq_len + self.puzzle_emb_len, config.rope_theta)
        self.layers = nn.ModuleList([URMBlock(config) for _ in range(config.num_layers)])
        self.register_buffer("init_hidden", trunc_normal_init_(torch.empty(config.hidden_size, dtype=self.forward_dtype), std=1), persistent=True)
        with torch.no_grad():
            self.q_head.weight.zero_()
            self.q_head.bias.fill_(-5)

    def _input_embeddings(self, input: torch.Tensor, puzzle_identifiers: torch.Tensor):
        embedding = self.embed_tokens(input.to(torch.int32))
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
        q_logits = self.q_head(hidden_states[:, 0]).to(torch.float32)
        return replace(carry, current_hidden=hidden_states.detach()), self.lm_head(hidden_states)[:, self.puzzle_emb_len:], (q_logits[..., 0], q_logits[..., 1])

class URM(nn.Module):
    def __init__(self, config: URMConfig):
        super().__init__()
        self.config = config
        self.inner = URMInner(config)

    def initial_carry(self, batch: Dict[str, torch.Tensor]) -> URMCarry:
        B = batch["inputs"].shape[0]
        base = self.inner.empty_carry(B)
        return URMCarry(current_hidden=base.current_hidden, steps=torch.zeros((B,), dtype=torch.int32, device=batch["inputs"].device),
                        halted=torch.ones((B,), dtype=torch.bool, device=batch["inputs"].device), current_data={k: torch.empty_like(v) for k, v in batch.items()})

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
        return URMCarry(current_hidden=new_carry.current_hidden, steps=new_steps, halted=halted, current_data=new_current_data), {"logits": logits, "q_halt_logits": q_halt_logits, "q_continue_logits": q_continue_logits}

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