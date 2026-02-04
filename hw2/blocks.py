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
# 权重矩阵: (vocab_size, d_model)
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



        

