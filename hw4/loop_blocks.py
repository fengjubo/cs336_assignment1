import torch
import torch.nn as nn
import math
from einops import einsum, rearrange
import torch.nn.functional as F
import numpy as np


import os
from collections.abc import Iterable
from typing import IO, Any, BinaryIO

def data_loading(
    dataset, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # dataset 是一维 token ID 序列
    # x: [token_i, ..., token_{i+context_length-1}]
    # y: [token_{i+1}, ..., token_{i+context_length}]
    # 所以起点 i 最大只能到 len(dataset) - context_length - 1

    max_start = len(dataset) - context_length - 1
    if max_start < 0:
        raise ValueError("dataset is too short for the given context_length")

    # 随机采样 batch_size 个起点
    starts = np.random.randint(0, max_start + 1, size=batch_size)

    # 构造输入和目标
    x = np.stack([dataset[i : i + context_length] for i in starts])
    y = np.stack([dataset[i + 1 : i + 1 + context_length] for i in starts])

    # 转成 torch tensor 并放到指定设备
    x_tensor = torch.tensor(x, dtype=torch.long, device=device)
    y_tensor = torch.tensor(y, dtype=torch.long, device=device)

    return x_tensor, y_tensor

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]