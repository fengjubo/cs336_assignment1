import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
import torch.nn.functional as F


# Implement a Linear class that inherits from torch.nn.Module and performs a linear transformation.


class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {'device': device, 'dtype': dtype}
        self.W = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))

        self.reset_parameters()

    def reset_parameters(self):

        sigma = math.sqrt(2.0 / (self.in_features + self.out_features))

        nn.init.trunc_normal_(self.W, mean=0.0, std=sigma, a=-3.0*sigma, b=3.0*sigma)

    def forward(self, x):
        # return x @ self.W
        return einsum(x, self.W, '... i, o i -> ... o')
    
    
# 输入: (batch_size, seq_len) -> 输出: (batch_size, seq_len, d_model)
# 权重矩阵: (vocab_size, d_model)，这里不是相乘，而是查找
class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):

        super().__init__()

        self.weight = nn.Parameter(torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype))

        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        return self.weight[x]
    

class rmsnorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        in_dtype = x.dtype
        x = x.to(torch.float32)

        ms = x.pow(2).mean(dim = -1, keepdim = True)

        rms = torch.sqrt(ms + self.eps)

        x_normed = x / rms
        result  = self.scale * x_normed

        return result.to(in_dtype)
    
# 实现一个 SwiGLU 前馈网络，该网络由一个 SiLU 激活函数和一个 GLU 组成。
class positionwise_feedforward (nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None,  device=None, dtype=None):
        super().__init__()
        
        if d_ff is None:
            d_ff = int(8/3 * d_model)
            d_ff = (d_ff + 63) //64 * 64  # 向上取整到最接近的8的倍数

        self.d_ff = d_ff

        self.w1 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w2 = Linear(d_model, d_ff, device=device, dtype=dtype)

        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        swish_part = F.silu(self.w1(x))

        gate_part = self.w3(x)

        intermediate = swish_part * gate_part

        output = self.w2(intermediate)

        return output
    
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int, device = None):
        """
        构建 RoPE 模块并预计算 cos 和 sin 缓存。
        
        theta: 旋转角度的基础常数 (Θ)
        d_k: 查询(query)和键(key)向量的维度
        max_seq_len: 允许的最大序列长度
        device: 存储 buffer 的设备
        """
        super().__init__()

        self.d_k = d_k

        #  $\theta_{i,k} = i \cdot \Theta^{-(2k-2)/d}$
        indices = torch.arange(0, d_k, 2).float().to(device)
        inv_freq = 1.0 / (theta ** (indices / d_k ))
        t = torch.arange(max_seq_len, device = device).float()
        # 外积运算: (max_seq_len,) outer (d_k/2,) -> (max_seq_len, d_k/2)
        freqs = torch.outer(t, inv_freq)
        # freqs = einsum(t, inv_freq, 'i, j -> i j')

        emb = torch.repeat_interleave(freqs, 2, dim=-1) # (max_seq_len, d_k)

        # 使用 register_buffer 存储，这样它们会随模型移动到同一设备，但不会被训练
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_adjacent(self, x: torch.Tensor):
        """
        将 [x1, x2, x3, x4] 变换为 [-x2, x1, -x4, x3]
        用于辅助计算旋转矩阵的乘法。
        """
        # 这种写法可以处理任意维度的 x，只要最后一维是 d_k
        x1 = x[..., 0::2] # 取奇数列 (index 0, 2, ...)
        x2 = x[..., 1::2] # 取偶数列 (index 1, 3, ...)
        
        # stack 并在最后一维 flatten 从而实现交错合并
        # stack 后的 shape: (..., d_k/2, 2) -> flatten 变成 (..., d_k)
        res = torch.stack((-x2, x1), dim=-1)
        return res.flatten(-2)
    
    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        """
        x: (..., seq_len, d_k)
        token_positions: (..., seq_len) -> 包含了每个 token 的实际位置索引
        """
        # 1. 根据 token_positions 获取对应的 cos 和 sin 值
        # token_positions 可能有 batch 维度，利用它在 buffer 中索引
        # cos/sin shape: (..., seq_len, d_k)
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        # 2. 应用 RoPE 公式: x' = x * cos + rotate_adjacent(x) * sin
        return (x * cos) + (self._rotate_adjacent(x) * sin)
    
def softmax(x: torch.Tensor, i: int = -1) -> torch.Tensor:
    """
    实现一个数值稳定的 softmax 函数。
    """
    # 1. 减去最大值以避免指数爆炸
    x_max = x.max(dim=i, keepdim=True).values
    x_exp = (x - x_max).exp()

    # 2. 计算 softmax
    sum_exp = x_exp.sum(dim=i, keepdim=True)
    return x_exp / sum_exp

def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    """
    实现一个数值稳定的缩放点积注意力函数。
    q, k, v: (..., seq_len, d_k)
    
    """
    d_k = q.shape[-1]
    
    # 1. 计算缩放点积
    scores = einsum(q, k, '... i d, ... j d -> ... i j') / math.sqrt(d_k)

    # 2. 应用掩码（如果提供）
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # 3. 应用数值稳定的 softmax
    attn_weights = softmax(scores, i=-1)

    # 4. 计算加权值
    return einsum(attn_weights, v, '... i j, ... j d -> ... i d')


class multihead_self_attention(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads: int,
            q_proj_weight: torch.Tensor,
            k_proj_weight: torch.Tensor,
            v_proj_weight: torch.Tensor,
            o_proj_weight: torch.Tensor,
            device = None,
            dtype = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight

        # self.rope = RotaryPositionalEmbedding(
        #     d_k = self.d_k,
        #     theta = theta,
        #     max_seq_len = max_seq_len,
        #     device = device,
        # )

    def forward(self, x: torch.Tensor)-> torch.Tensor:
        """
        x: (B, T, D)
        token_positions: (B, T)
        """
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.d_k

        q = einsum(x, self.q_proj_weight, 'b t d_in, d_out d_in -> b t d_out')
        k = einsum(x, self.k_proj_weight, 'b t d_in, d_out d_in -> b t d_out')
        v = einsum(x, self.v_proj_weight, 'b t d_in, d_out d_in -> b t d_out')

        q = rearrange(q, 'b t (h d) -> b h t d', h = H)
        k = rearrange(k, 'b t (h d) -> b h t d', h = H)
        v = rearrange(v, 'b t (h d) -> b h t d', h = H)


        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask = rearrange(mask, 't1 t2 -> 1 1 t1 t2')

        out = scaled_dot_product_attention(q, k, v, mask=mask)

        out = rearrange(out, 'b h t d -> b t (h d)')

        out = einsum(out, self.o_proj_weight, 'b t d_in, d_out d_in -> b t d_out')

        return out
    

class multihead_self_attention_with_rope(nn.Module):
    def __init__(
            self,
            d_model:int,
            num_heads: int,
            max_seq_len: int,
            theta: float,
            q_proj_weight: torch.Tensor,
            k_proj_weight: torch.Tensor,
            v_proj_weight: torch.Tensor,
            o_proj_weight: torch.Tensor,
            device = None,
            dtype = None,
    ):
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj_weight = q_proj_weight
        self.k_proj_weight = k_proj_weight
        self.v_proj_weight = v_proj_weight
        self.o_proj_weight = o_proj_weight

        self.rope = RotaryPositionalEmbedding(
            d_k = self.d_k,
            theta = theta,
            max_seq_len = max_seq_len,
            device = device,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None)-> torch.Tensor:
        """
        x: (B, T, D)
        token_positions: (B, T)
        """
        B, T, D = x.shape
        H = self.num_heads
        Dh = self.d_k

        if token_positions is None:
            token_positions = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)

        q = einsum(x, self.q_proj_weight, 'b t d_in, d_out d_in -> b t d_out')
        k = einsum(x, self.k_proj_weight, 'b t d_in, d_out d_in -> b t d_out')
        v = einsum(x, self.v_proj_weight, 'b t d_in, d_out d_in -> b t d_out')


        q = rearrange(q, 'b t (h d) -> b h t d', h = H)
        k = rearrange(k, 'b t (h d) -> b h t d', h = H)
        v = rearrange(v, 'b t (h d) -> b h t d', h = H)

        token_positions = rearrange(token_positions, 'b t ->  b 1 t')
        q = self.rope(q, token_positions)
        k = self.rope(k, token_positions)


        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        mask = rearrange(mask, 't1 t2 -> 1 1 t1 t2')

        out = scaled_dot_product_attention(q, k, v, mask=mask)

        out = rearrange(out, 'b h t d -> b t (h d)')

        out = einsum(out, self.o_proj_weight, 'b t d_in, d_out d_in -> b t d_out')

        return out
    
class transformer_block(nn.Module):
    def __init__(
            self,
            d_model: int,
            num_heads: int,
            d_ff: int,
            max_seq_len: int,
            theta: float,
            q_proj_weight: torch.Tensor,

            k_proj_weight: torch.Tensor,

            v_proj_weight: torch.Tensor,

            o_proj_weight: torch.Tensor,

            device=None,

            dtype=None,
        ):
        super().__init__()

        self.ln1 = rmsnorm(d_model, device=device, dtype=dtype)
        
        self.attn = multihead_self_attention_with_rope(
            d_model = d_model,
            num_heads = num_heads,
            max_seq_len = max_seq_len,
            theta = theta,
            q_proj_weight = q_proj_weight,
            k_proj_weight = k_proj_weight,
            v_proj_weight = v_proj_weight,
            o_proj_weight = o_proj_weight,
            device = device,
            dtype = dtype,
        )

        self.ln2 = rmsnorm(d_model, device=device, dtype=dtype)

        self.ffn = positionwise_feedforward(
            d_model = d_model,
            d_ff = d_ff,
            device = device,
            dtype = dtype,
        )

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        # 注意力子层
        x = x + self.attn(self.ln1(x), token_positions)

        # 前馈网络子层
        x = x + self.ffn(self.ln2(x))

        return x
    

class transformer_lm(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            theta: float,
            device=None,
            dtype=None,
        ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList([
            transformer_block(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                max_seq_len=context_length,
                theta=theta,
                q_proj_weight=None,
                k_proj_weight=None,
                v_proj_weight=None,
                o_proj_weight=None,
                device=device,
                dtype=dtype
            ) for _ in range(num_layers)
        ])

        self.ln_final = rmsnorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.token_embeddings(input_ids)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)
        logits = self.lm_head(x)

        return logits
