import regex as re
import os
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import array
import heapq  # <--- 必须引入堆

# GPT-2 标准正则模式
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

_GPT2_PAT = re.compile(GPT2_SPLIT_PATTERN)

def _process_chunk_safe(chunk_data):
    """
    子进程：处理文本块。
    为了防止切分时切断了单词，我们只处理 regex 匹配。
    """
    chunk_text, offset, special_tokens_set = chunk_data
    if not chunk_text:
        return []
    
    local_tokens = []
    # 如果这块文本正好是一个 special token
    if chunk_text in special_tokens_set:
        return [] 

    # 运行正则
    words = _GPT2_PAT.findall(chunk_text)
    for word in words:
        for b in word.encode("utf-8"):
            local_tokens.append(b + offset)
        local_tokens.append(-1) # 单词边界
    return local_tokens

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    max_train_bytes: int = None,
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    # ================= 1. 初始化词表 =================
    vocab = {}
    for i, token in enumerate(special_tokens):
        vocab[i] = token.encode("utf-8")

    offset = len(special_tokens)
    for i in range(256):
        vocab[i + offset] = bytes([i])

    next_token_id = offset + 256

    # ================= 2. 读取并强制分块多进程预处理 =================
    print(f"Loading data from {input_path}...")
    
    text = ""
    with open(input_path, 'r', encoding='utf-8') as f:
        if max_train_bytes:
            print(f"Sampling first {max_train_bytes / 1024 / 1024:.2f} MB of data...")
            text = f.read(max_train_bytes)
        else:
            text = f.read()

    # [关键修复 1] 无论有无特殊 token，都按行或固定大小切分任务
    # 为了安全起见（不切断单词），我们先按特殊 token 切，再把大块切小
    
    special_tokens_set = set(special_tokens)
    if special_tokens:
        delimiter = '|'.join(re.escape(token) for token in special_tokens)
        raw_splits = re.split(f'({delimiter})', text)
    else:
        raw_splits = [text]
    
    del text # 释放原始大字符串

    process_tasks = []
    is_special_mask = [] # 标记: None=空, str=special_token, int=task_index

    # 目标块大小：例如 1MB
    TARGET_CHUNK_SIZE = 1024 * 1024 

    for part in raw_splits:
        if not part:
            continue
        if part in special_tokens_set:
            is_special_mask.append(part) # 是特殊 token
        else:
            # 是普通文本，检查长度
            if len(part) > TARGET_CHUNK_SIZE:
                # 强制切分大块文本，为了并行
                # 注意：简单的切分可能会切断正则匹配（比如把单词切两半）
                # 最安全的做法是按换行符切，或者容忍极少量的边界错误
                # 这里我们采用按换行符切分作为折中
                sub_lines = part.split('\n')
                buffer = []
                buffer_len = 0
                
                for line in sub_lines:
                    line_full = line + '\n' # 补回换行符（split丢了）
                    buffer.append(line_full)
                    buffer_len += len(line_full)
                    
                    if buffer_len >= TARGET_CHUNK_SIZE:
                        # 打包一个任务
                        chunk_text = "".join(buffer)
                        task_idx = len(process_tasks)
                        process_tasks.append((chunk_text, offset, special_tokens_set))
                        is_special_mask.append(task_idx)
                        buffer = []
                        buffer_len = 0
                
                # 剩下的 buffer
                if buffer:
                    chunk_text = "".join(buffer)
                    task_idx = len(process_tasks)
                    process_tasks.append((chunk_text, offset, special_tokens_set))
                    is_special_mask.append(task_idx)
            else:
                # 够小，直接作为一个任务
                task_idx = len(process_tasks)
                process_tasks.append((part, offset, special_tokens_set))
                is_special_mask.append(task_idx)

    print(f"Pre-tokenizing with {os.cpu_count()} processes ({len(process_tasks)} tasks)...")
    
    flat_tokens = array.array('i')
    processed_results = []
    
    if process_tasks:
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            # 使用 map 保持顺序
            processed_results = list(executor.map(_process_chunk_safe, process_tasks))
    
    # 组装结果
    print("Assembling tokens...")
    for item in is_special_mask:
        if isinstance(item, str): # Special Token
            st_id = special_tokens.index(item)
            flat_tokens.append(st_id)
            flat_tokens.append(-1)
        elif isinstance(item, int): # Task Index
            res = processed_results[item]
            flat_tokens.extend(res)
    
    del processed_results, process_tasks, raw_splits, is_special_mask
    print(f"Pre-tokenization done. {len(flat_tokens)} tokens loaded.")

    # ================= 3. 构建链表与索引 =================
    n_tokens = len(flat_tokens)
    print("Building linked list...")
    pre = array.array('i', range(-1, n_tokens - 1))
    nxt = array.array('i', range(1, n_tokens + 1))
    
    # 快速修正边界
    for i in range(n_tokens):
        if flat_tokens[i] == -1:
            pre[i] = -1
            nxt[i] = -1
            if i > 0: nxt[i-1] = -1
            if i < n_tokens - 1: pre[i+1] = -1

    print("Building stats...")
    stats = Counter()
    indices = defaultdict(list)
    
    current_pos = 0
    while current_pos < n_tokens - 1:
        if flat_tokens[current_pos] != -1 and nxt[current_pos] != -1:
            pair = (flat_tokens[current_pos], flat_tokens[nxt[current_pos]])
            stats[pair] += 1
            indices[pair].append(current_pos)
        current_pos += 1

    # [关键修复 2] 使用堆 (Heap) 来维护最高频 Pair
    # Python 的 heap 是小根堆，所以存 (-count, p0, p1)
    print("Initializing Heap...")
    heap = []
    for pair, count in stats.items():
        # 注意：这里我们用 p0, p1 来 Tie-break，这和 max(key=...) 的字节序略有不同
        # 但在追求极致速度时，这是必要的妥协。
        heapq.heappush(heap, (-count, pair))



    # ================= 4. BPE 训练循环 (优化版) =================
    print("Starting BPE training loop...")
    merges = []
    
    while len(vocab) < vocab_size:
        if not heap:
            break
            
        # 1. 从堆中取出最佳 Pair
        neg_count, best_pair = heapq.heappop(heap)
        count = -neg_count
        
        # 检查是否过期
        if stats[best_pair] != count:
            continue
        if count <= 0:
            continue

        p0, p1 = best_pair
        
        # --- 优化开始：索引列表垃圾回收 ---
        pos_list = indices[best_pair]
        
        # 如果索引列表长度是实际计数的多倍（比如 > 5倍），说明有很多无效的死链接
        # 这时候先做一次清理 (Compaction)，避免对大量无效数据进行 sorted()
        if len(pos_list) > 10 and len(pos_list) > count * 2:
            # 只保留那些真正还是 (p0, p1) 的位置
            # 注意：这里不需要检查 next_node，因为那是合并时的逻辑，这里只看当前位置是否有效
            pos_list = [pos for pos in pos_list if flat_tokens[pos] == p0 and nxt[pos] != -1 and flat_tokens[nxt[pos]] == p1]
            indices[best_pair] = pos_list # 更新回去
        
        # 2. 获取位置并排序 (现在排序的是清理过的列表，快很多)
        # 必须排序，因为我们修改链表顺序需要从前向后（或者保持拓扑一致性）
        occurrences = sorted(pos_list)
        # --- 优化结束 ---

        # 记录 Merge
        merges.append((vocab[p0], vocab[p1]))
        vocab[next_token_id] = vocab[p0] + vocab[p1]
        
        # 3. 遍历位置执行合并
        for i in occurrences:
            # 再次检查（Double check），因为处理序列前面的 merge 可能会影响后面的
            if flat_tokens[i] == -1 or flat_tokens[i] != p0: continue
            if nxt[i] == -1: continue
            if flat_tokens[nxt[i]] != p1: continue

            head = i
            tail = nxt[i]
            prev_node = pre[head]
            next_node = nxt[tail]
            
            # --- 更新邻居统计 ---
            if prev_node != -1:
                old_prev_pair = (flat_tokens[prev_node], flat_tokens[head])
                stats[old_prev_pair] -= 1
                if stats[old_prev_pair] == 0: del stats[old_prev_pair]

            if next_node != -1:
                old_next_pair = (flat_tokens[tail], flat_tokens[next_node])
                stats[old_next_pair] -= 1
                if stats[old_next_pair] == 0: del stats[old_next_pair]

            # --- 更新链表 ---
            flat_tokens[head] = next_token_id
            flat_tokens[tail] = -1 
            
            nxt[head] = next_node
            if next_node != -1:
                pre[next_node] = head
            
            # --- 增加新邻居统计 ---
            if prev_node != -1:
                new_prev_pair = (flat_tokens[prev_node], flat_tokens[head])
                stats[new_prev_pair] += 1
                indices[new_prev_pair].append(prev_node) # 这里只是追加，会导致列表变得无序且冗余
                heapq.heappush(heap, (-stats[new_prev_pair], new_prev_pair))
            
            if next_node != -1:
                new_next_pair = (flat_tokens[head], flat_tokens[next_node])
                stats[new_next_pair] += 1
                indices[new_next_pair].append(head)
                heapq.heappush(heap, (-stats[new_next_pair], new_next_pair))

        # 清理当前 Pair
        del stats[best_pair]
        del indices[best_pair]
        
        next_token_id += 1
        
        if len(vocab) % 100 == 0: # 打印频率调高一点，方便观察
            print(f"Vocab size: {len(vocab)}/{vocab_size} | Heap size: {len(heap)} | Occurrences process: {len(occurrences)}")


    return vocab, merges