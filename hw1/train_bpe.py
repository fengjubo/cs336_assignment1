import regex as re
from typing import Dict, List, Tuple
from collections import Counter
import os

# GPT-2 标准正则模式 (必须完全一致，包含非捕获组)
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # 1. 初始化词表：Special Tokens -> Base Bytes
    vocab = {}
    
    # 1.1 先放 Special Tokens (ID: 0 ~ len-1)
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    # 1.2 再放基础字节 (ID: len ~ len+255)
    offset = len(special_tokens)
    for i in range(256):
        vocab[i + offset] = bytes([i])

    # 下一个可用的 merge ID (从 len+256 开始)
    next_token_id = offset + 256

    # 2. 读取文件
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()


# 第一刀：chunks (切除 Special Tokens)
# 变量名：chunks
# 分隔符：用户定义的特殊 Token (比如 <|endoftext|>)
# 状态：字符串列表 (List of Strings)
# 逻辑：
# 如果文本是 "Hello<|endoftext|>World"
# 它会被切成：['Hello', '<|endoftext|>', 'World']
# 注意：这里还没有处理空格，空格还在字符串里。

    # 按 special tokens 切分
    if special_tokens:
        delimiter = '|'.join(re.escape(token) for token in special_tokens)
        chunks = re.split(f'({delimiter})', text)
    else:
        chunks = [text]
    
    gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)
    
    # 3. 构建初始 token 列表 (Pre-tokenization)
    all_tokens = []
    special_tokens_set = set(special_tokens)

    for chunk in chunks:
        if not chunk:
            continue
        if chunk in special_tokens_set:
            continue
        

# 第二刀：words (切除单词/标点)
# 变量名：words (在循环里的临时变量 gpt2_pat.findall(chunk))
# 分隔符：GPT-2 正则表达式
# 状态：字符串列表 (List of Strings)
# 逻辑：
# 正则会把文本按照“单词”、“数字”、“标点”切开。
# 重点中的重点（最容易晕的地方）：
# 空格不是分隔符！ 在 GPT-2 BPE 里，空格通常被归并到单词的前面。
# 例如 "Hello world" (中间有一个空格)。
# 会被切成：['Hello', ' world']。
# 注意第二个元素是 ' world'（空格包含在内）。

        # 应用 GPT-2 正则
        words = gpt2_pat.findall(chunk)
        for word in words:
            # 必须加上 offset，因为 0~255 的位置可能被 special tokens 挤占了
            byte_seq = [b + offset for b in word.encode('utf-8')]
            all_tokens.append(byte_seq)

    # 4. BPE 训练循环
    merges = []
    
    pair_freq = Counter()
    for byte_seq in all_tokens:
        for i in range(len(byte_seq) - 1):
            pair_freq[(byte_seq[i], byte_seq[i + 1])] += 1

    
    def get_score(p):
        # 优先级：1. 频率(高) 2. 第一个token字节(大) 3. 第二个token字节(大)
        return (pair_freq[p], vocab[p[0]], vocab[p[1]])

    while len(vocab) < vocab_size and pair_freq:
        # 找到分数最高的 pair
        best_pair = max(pair_freq, key=get_score)
        p1, p2 = best_pair
        
        # 记录 merge
        merges.append((vocab[p1], vocab[p2]))
        
        # 新 Token 入库
        vocab[next_token_id] = vocab[p1] + vocab[p2]
        
        # 更新所有 token 序列
        new_all_tokens = []
        target_id = next_token_id
        
        for byte_seq in all_tokens:
            if len(byte_seq) < 2:
                new_all_tokens.append(byte_seq)
                continue
            
            new_seq = []
            i = 0
            seq_len = len(byte_seq)
            while i < seq_len:
                if i < seq_len - 1 and (byte_seq[i], byte_seq[i+1]) == best_pair:
                    new_seq.append(target_id)
                    i += 2
                else:
                    new_seq.append(byte_seq[i])
                    i += 1
            new_all_tokens.append(new_seq)
            
        all_tokens = new_all_tokens
        next_token_id += 1  

        # 重新统计频率
        pair_freq = Counter()
        for byte_seq in all_tokens:
            for i in range(len(byte_seq) - 1):
                pair_freq[(byte_seq[i], byte_seq[i + 1])] += 1

    return vocab, merges