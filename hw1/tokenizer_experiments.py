# 配置路径
import os
import time
import json
import numpy as np
from pathlib import Path
import sys
from pathlib import Path


#  获取当前文件所在目录（tests/）
current_dir = Path(__file__).parent  # -> /path/to/your-project/tests

#  项目根目录是 current_dir 的父目录
project_root = current_dir.parent    # -> /path/to/your-project

#  把项目根目录加入 sys.path
sys.path.insert(0, str(project_root))
from hw1.tokenizer import Tokenizer 

# 配置路径
TINYSTORIES_PATH = "data/TinyStoriesV2-GPT4-train.txt"
OWT_PATH = "data/owt_train.txt"
TINY_VOCAB = "outputs/tinystories_vocab.json"
TINY_MERGES = "outputs/tinystories_merges.txt"
OWT_VOCAB = "outputs/openwebtext_vocab.json"
OWT_MERGES = "outputs/openwebtext_merges.txt"

SPECIAL_TOKENS = ["<|endoftext|>"]

def load_and_fix_tokenizer(vocab_path, merges_path):
    """加载 Tokenizer 并手动修复因 JSON 序列化丢失的原始字节映射"""
    tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens=SPECIAL_TOKENS)
    
    # 核心修复逻辑：强制注入 256 个原始字节映射
    # 在你的 train_bpe 实现中，字节 ID = 索引 + len(special_tokens)
    offset = len(SPECIAL_TOKENS)
    for i in range(256):
        byte_val = bytes([i])
        token_id = offset + i
        tokenizer.id_to_token[token_id] = byte_val
        tokenizer.token_to_id[byte_val] = token_id
    
    return tokenizer

def get_random_documents(file_path, num_docs=10):
    """从文件中读取前 10 篇文档 (以 <|endoftext|> 分隔)"""
    with open(file_path, "r", encoding="utf-8") as f:
        # 读取一部分数据，足够切分出 10 篇即可
        content = f.read(1024 * 1024 * 5) 
        docs = content.split("<|endoftext|>")
        # 过滤空字符串并取前 10
        return [d.strip() for d in docs if d.strip()][:num_docs]

def compute_stats(tokenizer, docs):
    """计算压缩率和处理速度"""
    total_bytes = 0
    total_tokens = 0
    
    start_time = time.time()
    for doc in docs:
        encoded_ids = tokenizer.encode(doc)
        total_bytes += len(doc.encode("utf-8"))
        total_tokens += len(encoded_ids)
    end_time = time.time()
    
    ratio = total_bytes / total_tokens if total_tokens > 0 else 0
    duration = end_time - start_time
    throughput = total_bytes / duration if duration > 0 else 0
    
    return ratio, throughput

def run_experiments():
    print("Loading and fixing tokenizers...")
    tiny_tokenizer = load_and_fix_tokenizer(TINY_VOCAB, TINY_MERGES)
    owt_tokenizer = load_and_fix_tokenizer(OWT_VOCAB, OWT_MERGES)

    # 抽取文档
    tiny_docs = get_random_documents(TINYSTORIES_PATH)
    owt_docs = get_random_documents(OWT_PATH)

    # (a) 各自的压缩率
    print("\n--- Experiment (a) ---")
    tiny_ratio, _ = compute_stats(tiny_tokenizer, tiny_docs)
    owt_ratio, owt_speed = compute_stats(owt_tokenizer, owt_docs)
    print(f"TinyStories Tokenizer (on TinyStories): {tiny_ratio:.4f} bytes/token")
    print(f"OpenWebText Tokenizer (on OpenWebText): {owt_ratio:.4f} bytes/token")

    # (b) 交叉测试
    print("\n--- Experiment (b) ---")
    cross_ratio, _ = compute_stats(tiny_tokenizer, owt_docs)
    print(f"TinyStories Tokenizer (on OpenWebText data): {cross_ratio:.4f} bytes/token")

    # (c) 吞吐量与 Pile 估算
    print("\n--- Experiment (c) ---")
    mb_per_sec = owt_speed / (1024 * 1024)
    print(f"Throughput: {mb_per_sec:.4f} MB/s")
    
    pile_size_gb = 825
    total_seconds = (pile_size_gb * 1024 * 1024 * 1024) / owt_speed
    days = total_seconds / (24 * 3600)
    print(f"Estimated time to tokenize The Pile (825GB): {days:.2f} days")

    # (d) uint16 验证
    # 模拟编码一段数据并保存为 uint16
    sample_ids = owt_tokenizer.encode(owt_docs[0])
    ids_array = np.array(sample_ids, dtype=np.uint16)
    print("\n--- Experiment (d) ---")
    print(f"Sample IDs (first 10): {sample_ids[:10]}")
    print(f"NumPy uint16 array shape: {ids_array.shape}, dtype: {ids_array.dtype}")

if __name__ == "__main__":
    run_experiments()