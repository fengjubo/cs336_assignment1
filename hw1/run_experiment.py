import os
import json
import time
import tracemalloc
from fastest_train_bpe import train_bpe

import sys
from pathlib import Path


#  获取当前文件所在目录（tests/）
current_dir = Path(__file__).parent  # -> /path/to/your-project/tests

#  项目根目录是 current_dir 的父目录
project_root = current_dir.parent    # -> /path/to/your-project

#  把项目根目录加入 sys.path
sys.path.insert(0, str(project_root))

#  现在可以导入 hw1.fastest_train_bpe
from hw1.fastest_train_bpe import train_bpe


# === 配置路径 ===
# 请修改为你的真实数据路径
TINYSTORIES_PATH = "data/TinyStoriesV2-GPT4-train.txt" 
OWT_PATH = "data/owt_train.txt"  
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_tokenizer(vocab, merges, prefix):
    """保存 vocab 和 merges 到磁盘"""
    # 保存 vocab (dict)
    vocab_str = {k: v.decode('utf-8', errors='replace') for k, v in vocab.items()}
    with open(f"{OUTPUT_DIR}/{prefix}_vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_str, f, indent=2, ensure_ascii=False)
    
    # 保存 merges (txt)
    with open(f"{OUTPUT_DIR}/{prefix}_merges.txt", "w", encoding="utf-8") as f:
        for p1, p2 in merges:
            s1 = p1.decode('utf-8', errors='replace')
            s2 = p2.decode('utf-8', errors='replace')
            f.write(f"{s1} {s2}\n")

def analyze_tokenizer(vocab, prefix):
    """分析最长 Token"""
    longest_token_id = max(vocab, key=lambda k: len(vocab[k]))
    longest_token_bytes = vocab[longest_token_id]
    longest_token_str = longest_token_bytes.decode('utf-8', errors='replace')
    
    print(f"[{prefix}] Longest Token Length: {len(longest_token_bytes)} bytes")
    print(f"[{prefix}] Longest Token Content (repr): {repr(longest_token_str)}")
    return longest_token_str

def run_experiment(input_path, vocab_size, prefix, memory_limit_desc="Unknown", kwargs=None):
    if kwargs is None: kwargs = {}
    

    print(f"\n{'='*20} Running {prefix} (Target Vocab: {vocab_size}) {'='*20}")
    
    if not os.path.exists(input_path):
        print(f"Error: Input file {input_path} not found.")
        return

    # 开始监测内存和时间
    # tracemalloc.start()
    start_time = time.time()

    try:
        vocab, merges = train_bpe(
            input_path=input_path,
            vocab_size=vocab_size,
            special_tokens=["<|endoftext|>"],
            **kwargs  # <--- 确保这里把 max_train_bytes 传进去了
        )
        
        end_time = time.time()
        # current, peak = tracemalloc.get_traced_memory()
        # tracemalloc.stop()
        
        duration = end_time - start_time
        # peak_mb = peak / 1024 / 1024
        
        print(f"\n--- Results for {prefix} ---")
        print(f"Training Time: {duration:.2f} seconds ({duration/60:.2f} minutes)")
        print(f"Peak Memory Usage:  MB (Tracemalloc disabled for speed)")
        print(f"Constraint Check: Memory < {memory_limit_desc}")
        
        save_tokenizer(vocab, merges, prefix)
        analyze_tokenizer(vocab, prefix)
        
    except Exception as e:
        print(f"Failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    # 任务 (a): TinyStories (文件小，全量读)
    # 大约 15MB 左右，秒杀
    # run_experiment(
    #     input_path=TINYSTORIES_PATH, 
    #     vocab_size=10000, 
    #     prefix="tinystories", 
    #     memory_limit_desc="30GB",
    #     kwargs={"max_train_bytes": 30 * 1024 * 1024}
    # )
    
    # 任务 (b): OpenWebText (文件巨大，必须采样!)
    # 24GB 内存建议只读前 1GB (1024 * 1024 * 1024 字节)
    # 如果你想跑久一点，可以设为 2GB，但要注意发热
    # 32000 词表训练 1GB 数据大概需要几十分钟到1小时
    # 参数 max_train_bytes 会被传入 kwargs
    
    # 确保文件存在再跑
    if os.path.exists(OWT_PATH):
        run_experiment(
            input_path=OWT_PATH, 
            vocab_size=32000, 
            prefix="openwebtext", 
            memory_limit_desc="100GB",
            # 新增参数，只读 500MB (500 * 1024 * 1024)
            # 或者 100MB 用于快速测试
            kwargs={"max_train_bytes": 50 * 1024 * 1024}
        )
    else:
        print(f"Warning: {OWT_PATH} does not exist. Skipping OWT experiment.")