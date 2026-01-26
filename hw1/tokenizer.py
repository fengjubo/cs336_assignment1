import regex as re
import json
from typing import Iterable, Iterator

# GPT-2 标准正则模式
GPT2_SPLIT_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

class Tokenizer:
    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] = None):
        # 1. 基础词表映射
        self.id_to_token = vocab
        self.token_to_id = {v: k for k, v in vocab.items()}
        
        # 2. 合并规则映射 (pair -> rank)
        # 严格按照训练时 merges 列表的顺序（索引越小优先级越高）
        self.merges = {pair: i for i, pair in enumerate(merges)}
        
        # 3. 处理特殊标记
        self.special_tokens = special_tokens or []
        if self.special_tokens:
            # 关键修复：特殊标记必须按长度降序排序，以确保正则匹配时“最长匹配优先”
            # 否则 <|endoftext|><|endoftext|> 会被 <|endoftext|> 提前截断
            sorted_special = sorted(self.special_tokens, key=len, reverse=True)
            pattern = "|".join(re.escape(st) for st in sorted_special)
            self.special_pattern = re.compile(f"({pattern})")
        else:
            self.special_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
        # 加载词表
        with open(vocab_filepath, "r", encoding="utf-8") as f:
            vocab_raw = json.load(f)
            # JSON key 是 string，需要转回 int；value 转回 bytes
            vocab = {int(k): v.encode("utf-8") for k, v in vocab_raw.items()}
            
        # 加载合并规则
        merges = []
        with open(merges_filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line: continue
                
                # 关键修复：处理空格 Token 的解析逻辑
                # 因为规则保存为 "s1 s2"，如果 s1 是空格，行首会有多个空格
                # 这里寻找最后一个空格作为分割符是最稳妥的（BPE token 本身内部不含空格）
                split_idx = line.rfind(" ")
                if split_idx == -1: continue
                
                p1 = line[:split_idx].encode("utf-8")
                p2 = line[split_idx+1:].encode("utf-8")
                merges.append((p1, p2))
        
        return cls(vocab, merges, special_tokens)

    def _bpe_encode_word(self, word_bytes: bytes) -> list[int]:
        """对预分词单元内部进行 BPE 合并"""
        # 初始状态：每个字节为一个 bytes 对象
        parts = [bytes([b]) for b in word_bytes]
        
        while len(parts) > 1:
            # 找到当前所有相邻对中，在 merges 中 rank 最小（最靠前）的一对
            best_pair = None
            min_rank = float('inf')
            
            for i in range(len(parts) - 1):
                pair = (parts[i], parts[i+1])
                rank = self.merges.get(pair, float('inf'))
                if rank < min_rank:
                    min_rank = rank
                    best_pair = pair
            
            if best_pair is None:
                break # 没有更多可合并的规则
            
            # 执行合并：将所有出现的 best_pair 替换为合并后的结果
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and (parts[i], parts[i+1]) == best_pair:
                    new_parts.append(parts[i] + parts[i+1])
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts
            
        # 转为 ID，如果词表中没有（理论上不应发生），由于是字节级 BPE，最终一定能回退到字节 ID
        return [self.token_to_id[p] for p in parts]

    def encode(self, text: str) -> list[int]:
        if not text:
            return []
            
        # 1. 先按特殊标记切分文本
        if self.special_pattern:
            parts = self.special_pattern.split(text)
        else:
            parts = [text]
            
        final_ids = []
        for part in parts:
            if not part:
                continue
            # 如果是特殊标记
            if part in self.special_tokens:
                final_ids.append(self.token_to_id[part.encode("utf-8")])
            else:
                # 2. 对普通文本进行预分词
                words = re.findall(GPT2_SPLIT_PATTERN, part)
                for word in words:
                    # 3. 内部应用 BPE
                    final_ids.extend(self._bpe_encode_word(word.encode("utf-8")))
        return final_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        # 查找字节并拼接，使用 errors='replace' 处理非法的 UTF-8 序列
        byte_data = b"".join(self.id_to_token[i] for i in ids)
        return byte_data.decode("utf-8", errors="replace")