import regex as re
from typing import Dict, List, Tuple
from collections import Counter
import os

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path (str | os.PathLike): Path to BPE tokenizer training data.
        vocab_size (int): Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens (list[str]): A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
            vocab:
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges:
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    # step 1: 初始化词汇表vocabulary with byte values and special tokens
    vocab = {i: bytes([i]) for i in range(256)}
    token_id = 256

    for token in special_tokens:
        vocab[token_id] = token.encode('utf-8')
        token_id += 1

    # step 2: pre-tokenize the corpus (split on special tokens)
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens: # 这个if语句用于判断：special_tokens是否为空列表
        delimiter = '|'.join(re.escape(token) for token in special_tokens)
        chunks = re.split(f'({delimiter})', text)
    else:
        chunks = [text]
    
    # step 3: 把每一个 chunk 变成 byte sequences
    all_tokens = []
    special_tokens_set = set(special_tokens) # 用于快速查找

    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        # 【修改点 1】：如果 chunk 是特殊 token，直接跳过，不要加入训练数据
        if chunk in special_tokens_set:
            continue
            
        # 【修改点 2】：转成 list，而不是 bytes
        byte_seq = list(chunk.encode('utf-8'))
        all_tokens.append(byte_seq)

    # step 4: 统计 byte pair frequencies
    pair_freq = Counter()
    for byte_seq in all_tokens:
        for i in range(len(byte_seq) - 1):
            pair = (byte_seq[i], byte_seq[i + 1])
            pair_freq[pair] += 1

    # step 5: 在达到vocab_size之前，迭代地合并最频繁的byte pairs
    merges = []
    while len(vocab) < vocab_size and pair_freq:
        # 找到最频繁的pair
        best_pair = max(pair_freq, key = lambda x: pair_freq[x])
        # 如果有相同频率的，选择字典序最大的
        candidates = [p for p in pair_freq if pair_freq[p] == pair_freq[best_pair]]
        if len(candidates) > 1:
            best_pair = max(candidates)

        # 合并 pair
        new_token = vocab[best_pair[0]] + vocab[best_pair[1]] 
        vocab[token_id] = new_token
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        

        # 更新频率计数
        # 移除旧pair添加新pair
        del pair_freq[best_pair]
        for i in range (len(all_tokens)):
            byte_seq = all_tokens[i]
            new_byte_seq = []
            j = 0
            while j < len(byte_seq):
                if j < len(byte_seq) - 1 and (byte_seq[j], byte_seq[j + 1]) == best_pair:
                    new_byte_seq.append(token_id)
                    j += 2
                else:
                    new_byte_seq.append(byte_seq[j])
                    j += 1
            all_tokens[i] = new_byte_seq
        
        token_id += 1
        
        # 合并后重新计算频率
        pair_freq = Counter()
        for byte_seq in all_tokens:
            for i in range(len(byte_seq) - 1):
                pair = (byte_seq[i], byte_seq[i + 1])
                pair_freq[pair] += 1

    return vocab, merges
