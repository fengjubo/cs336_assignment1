import regex as re
import os
from collections import Counter, defaultdict

# GPT-2 标准正则模式
GPT2_SPLIT_PATTERN = (
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
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

    # ================= 2. 预处理文本 =================
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()

    if special_tokens:
        delimiter = '|'.join(re.escape(token) for token in special_tokens)
        chunks = re.split(f'({delimiter})', text)
    else:
        chunks = [text]
    
    gpt2_pat = re.compile(GPT2_SPLIT_PATTERN)
    
    # 使用 flat_tokens 存储 ID，-1 作为单词分隔符
    flat_tokens = [] 
    special_tokens_set = set(special_tokens)
    
    for chunk in chunks:
        if not chunk: continue
        if chunk in special_tokens_set: continue 
        
        words = gpt2_pat.findall(chunk)
        for word in words:
            word_bytes = [b + offset for b in word.encode('utf-8')]
            flat_tokens.extend(word_bytes)
            flat_tokens.append(-1) # 单词边界

    # ================= 3. 构建链表与索引 =================
    n_tokens = len(flat_tokens)
    tokens = flat_tokens
    
    # pre/nxt 数组模拟双向链表
    pre = list(range(-1, n_tokens - 1))
    nxt = list(range(1, n_tokens + 1))
    
    # 修正边界：-1 是分隔符，断开链表
    for i, t in enumerate(tokens):
        if t == -1:
            pre[i] = -1
            nxt[i] = -1
            if i > 0: nxt[i-1] = -1
            if i < n_tokens - 1: pre[i+1] = -1

    # 统计频率 & 倒排索引
    stats = Counter()
    indices = defaultdict(list)
    
    current_pos = 0
    while current_pos < n_tokens - 1:
        # 必须跳过边界 -1
        if tokens[current_pos] != -1 and nxt[current_pos] != -1:
            pair = (tokens[current_pos], tokens[nxt[current_pos]])
            stats[pair] += 1
            indices[pair].append(current_pos)
        current_pos += 1

    # ================= 4. BPE 训练循环 =================
    merges = []
    
    # Tie-Breaking: (频率, 字节序1, 字节序2)
    def get_score(pair):
        return (stats[pair], vocab[pair[0]], vocab[pair[1]])

    while len(vocab) < vocab_size and stats:
        # 1. 选出最佳 Pair
        best_pair = max(stats, key=get_score)
        
        # 频率校验
        if stats[best_pair] <= 0:
            del stats[best_pair]
            if not stats: break
            continue

        p0, p1 = best_pair
        
        # 2. 记录 Merge
        merges.append((vocab[p0], vocab[p1]))
        vocab[next_token_id] = vocab[p0] + vocab[p1]
        
        # 3. 执行合并
        # [CRITICAL Fix 1] 必须排序！确保从左到右处理，对齐标准 BPE 的贪婪行为
        occurrences = sorted(indices[best_pair])
        
        # 由于是惰性删除，indices 里可能包含无效位置，我们逐个检查
        for i in occurrences:
            # [Check 1] 位置 i 是否还是 p0? (可能被前面的合并修改了)
            # [Check 2] 位置 i 是否已被废弃 (-1)?
            if tokens[i] == -1 or tokens[i] != p0:
                continue
                
            # [Check 3] 链表是否完整?
            if nxt[i] == -1:
                continue
                
            # [Check 4] 下一个位置是否还是 p1?
            if tokens[nxt[i]] != p1:
                continue

            # === 确认匹配，开始合并 ===
            head = i
            tail = nxt[i]
            prev_node = pre[head]
            next_node = nxt[tail]
            
            # A. 减少旧邻居统计
            if prev_node != -1:
                old_prev_pair = (tokens[prev_node], tokens[head])
                stats[old_prev_pair] -= 1
                if stats[old_prev_pair] == 0: del stats[old_prev_pair]

            if next_node != -1:
                # 注意：tail 还没改，所以这里 tokens[tail] 还是 p1
                old_next_pair = (tokens[tail], tokens[next_node])
                stats[old_next_pair] -= 1
                if stats[old_next_pair] == 0: del stats[old_next_pair]

            # B. 更新链表结构
            tokens[head] = next_token_id
            
            # [CRITICAL Fix 2] 立即标记 tail 为无效
            # 防止 A A A 这种情况下，处理完 (A0, A1) 后，A1 又被当作头部去尝试合并 A2
            tokens[tail] = -1 
            
            nxt[head] = next_node
            if next_node != -1:
                pre[next_node] = head
            
            # tail 已经断开，不需要处理 pre[tail] / nxt[tail]
            
            # C. 增加新邻居统计
            if prev_node != -1:
                new_prev_pair = (tokens[prev_node], tokens[head])
                stats[new_prev_pair] += 1
                indices[new_prev_pair].append(prev_node)
            
            if next_node != -1:
                new_next_pair = (tokens[head], tokens[next_node])
                stats[new_next_pair] += 1
                indices[new_next_pair].append(head)

        # 清理
        del stats[best_pair]
        del indices[best_pair]
        
        next_token_id += 1

    return vocab, merges