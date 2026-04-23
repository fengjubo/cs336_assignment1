import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from hw2.blocks import softmax

def apply_top_p(probs: torch.Tensor, top_p: float | None) -> torch.Tensor:
    if top_p is None or top_p >= 1.0:
        return probs
    
    sorted_probs, sorted_indices = torch.sort(probs, dim = -1, descending = True)
    cumulative_probs = torch.cumsum(sorted_probs, dim = -1)

    sorted_mask = cumulative_probs > top_p    # True 表示这个 token 要被删掉

    sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
    sorted_mask[..., 0] = False

    sorted_probs = sorted_probs.masked_fill(sorted_mask, 0.0)

    sorted_probs = sorted_probs / sorted_probs.sum(dim = -1, keepdim = True)

    filtered_probs = torch.zeros_like(probs)
    filtered_probs.scatter_(dim = -1, index = sorted_indices, src = sorted_probs)

    return filtered_probs

def sample_next_token(logits: torch.Tensor, temperature: float = 1.0, top_p: float | None = None):
    if temperature == 0:
        return torch.argmax(logits, dim = -1, keepdim = True)
    logits = logits / temperature
    probs = softmax(logits, i = -1)

    probs = apply_top_p(probs, top_p)

    next_token = torch.multinomial(probs, num_samples = 1)
    return next_token

@torch.no_grad()
def generate(
    model,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float | None = None,
    endoftext_id: int | None = None,
)->torch.Tensor:
    """
    input_ids: [batch_size, seq_len]
    return:    [batch_size, seq_len + generated_len]
    """
    model.eval()

    generated = input_ids

    for _ in range(max_new_tokens):
        context = generated[:, -context_length:]

        logits = model(context)
        next_logits = logits[:, -1, :] # [B, vocab_size]
        next_token = sample_next_token(next_logits, temperature, top_p)

        generated = torch.cat([generated, next_token], dim = -1)
        if endoftext_id is not None:
            if torch.all(next_token.squeeze(-1) == endoftext_id):
                break

    return generated
