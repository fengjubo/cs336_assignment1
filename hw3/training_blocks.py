import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
import torch.nn.functional as F




# import os
# import sys
# from pathlib import Path

# #  获取当前文件所在目录（tests/）
# current_dir = Path(__file__).parent  # -> /path/to/your-project/tests

# #  项目根目录是 current_dir 的父目录
# project_root = current_dir.parent    # -> /path/to/your-project

# #  把项目根目录加入 sys.path
# sys.path.insert(0, str(project_root))

# from hw2.blocks import Linear, Embedding, rmsnorm, positionwise_feedforward, RotaryPositionalEmbedding, softmax, scaled_dot_product_attention, multihead_self_attention, multihead_self_attention_with_rope, transformer_block, transformer_lm


def cross_entropy(
    x: torch.Tensor, targets: torch.Tensor
):
    """Given a tensor of inputs and targets, compute the average cross-entropy
    loss across examples.

    Args:
        inputs (Float[Tensor, "batch_size vocab_size"]): inputs[i][j] is the
            unnormalized logit of jth class for the ith example.
        targets (Int[Tensor, "batch_size"]): Tensor of shape (batch_size,) with the index of the correct class.
            Each value must be between 0 and `num_classes - 1`.

    Returns:
        Float[Tensor, ""]: The average cross-entropy loss across examples.
    """

    x_max = x.max(dim=-1, keepdim=True).values
    x_shifted = (x - x_max)

    log_sum = x_shifted.exp().sum(dim = -1).log()
    correct_logits = x_shifted.gather(dim = -1, index = targets.unsqueeze(-1)).squeeze(-1)

    loss = log_sum - correct_logits
    
    return loss.mean()



class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr = 1e-3, betas = (0.9, 0.99), eps = 1e-8, weight_decay=1e-2):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1 = group["betas"][0]
            beta2 = group["betas"][1]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                step = state["step"] + 1
                state["step"] = step

                exp_avg.mul_(beta1).add_(grad, alpha = 1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value = 1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                alpha_t = lr * math.sqrt(bias_correction2) / bias_correction1

                fenmu = exp_avg_sq.sqrt() + eps
                p.data.addcdiv_(exp_avg, fenmu, value = -alpha_t)
                p.data.add_(p.data, alpha = -lr * weight_decay)

        return loss
    

def learning_rate_schedule(
    it: int,
    max_lr: float,
    min_lr: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
):
    if it < warmup_iters:
        alpha_t = it / warmup_iters * max_lr
        return alpha_t
    
    if it <= cosine_cycle_iters:
        alpha_t = min_lr + 0.5 * (1 + math.cos((it - warmup_iters) / (cosine_cycle_iters - warmup_iters) * math.pi)) * (max_lr - min_lr)
        return alpha_t
    
    return min_lr

def gradient_clipping(parameters, max_l2_norm: float, eps=1e-6):
    total_norm_sq = 0.0

    for p in parameters:
        if p.grad is None:
            continue
        grad = p.grad.data
        total_norm_sq += grad.pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)

    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)

        for p in parameters:
            if p.grad is None:
                continue
            p.grad.data.mul_(scale)


def initialize_transformer_lm_for_training(model: nn.Module, d_model: int, d_ff: int, device=None, dtype=None):
    factory_kwargs = {"device": device, "dtype": dtype}
    attn_sigma = math.sqrt(2.0 / (d_model + d_model))
    ffn_up_sigma = math.sqrt(2.0 / (d_model + d_ff))
    ffn_down_sigma = math.sqrt(2.0 / (d_ff + d_model))

    for layer in model.layers:
        layer.attn.q_proj_weight = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        layer.attn.k_proj_weight = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        layer.attn.v_proj_weight = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        layer.attn.o_proj_weight = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))

        nn.init.trunc_normal_(layer.attn.q_proj_weight, mean=0.0, std=attn_sigma, a=-3.0 * attn_sigma, b=3.0 * attn_sigma)
        nn.init.trunc_normal_(layer.attn.k_proj_weight, mean=0.0, std=attn_sigma, a=-3.0 * attn_sigma, b=3.0 * attn_sigma)
        nn.init.trunc_normal_(layer.attn.v_proj_weight, mean=0.0, std=attn_sigma, a=-3.0 * attn_sigma, b=3.0 * attn_sigma)
        nn.init.trunc_normal_(layer.attn.o_proj_weight, mean=0.0, std=attn_sigma, a=-3.0 * attn_sigma, b=3.0 * attn_sigma)

        layer.ffn.w1.W = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        layer.ffn.w3.W = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        layer.ffn.w2.W = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))

        nn.init.trunc_normal_(layer.ffn.w1.W, mean=0.0, std=ffn_up_sigma, a=-3.0 * ffn_up_sigma, b=3.0 * ffn_up_sigma)
        nn.init.trunc_normal_(layer.ffn.w3.W, mean=0.0, std=ffn_up_sigma, a=-3.0 * ffn_up_sigma, b=3.0 * ffn_up_sigma)
        nn.init.trunc_normal_(layer.ffn.w2.W, mean=0.0, std=ffn_down_sigma, a=-3.0 * ffn_down_sigma, b=3.0 * ffn_down_sigma)
