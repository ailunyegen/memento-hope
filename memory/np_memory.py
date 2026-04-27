"""
np_memory.py - Non-Parametric Memory Helper Functions
------------------------------------------------------
This module provides the core functionalities for loading, processing,
and retrieving cases from a JSONL-based case memory bank. It uses
sentence-transformers for semantic similarity search.
"""

import json
import sys
from typing import List, Tuple, Dict, Any
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

def load_jsonl(path: str) -> List[dict]:
    """Loads a JSONL file into a list of dictionaries."""
    items = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for ln, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    items.append(obj)
                except Exception as e:
                    print(f"[WARN] Failed to parse line {ln}, skipped: {e}", file=sys.stderr)
    except FileNotFoundError:
        print(f"[ERROR] Memory file not found: {path}", file=sys.stderr)
    return items

def extract_pairs(items: List[dict], key_field: str, value_field: str) -> List[Tuple[str, object, int]]:
    """Extracts key-value pairs from a list of dictionaries."""
    pairs = []
    for i, obj in enumerate(items):
        if key_field in obj and value_field in obj:
            pairs.append((str(obj[key_field]), obj[value_field], i))
        elif len(obj) == 2:
            ks = list(obj.keys())
            pairs.append((str(obj[ks[0]]), obj[ks[1]], i))
    return pairs


@torch.no_grad()
def embed_texts(
    texts: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModel,
    device: torch.device,
    batch_size: int = 64,
    max_length: int = 256,
) -> torch.Tensor:
    """Embeds a list of texts using the provided transformer model."""
    vecs = []
    model.eval() # type: ignore # 切换到评估模式
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch = texts[i : i + batch_size]
        # AutoTokenizer 实例是可调用的
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt"
        ) # type: ignore
        enc = {k: v.to(device) for k, v in enc.items()}
        # AutoModel 实例是可调用的
        out = model(**enc, return_dict=True) # type: ignore
        if hasattr(out, "pooler_output") and out.pooler_output is not None:
            e = out.pooler_output
        else:
            e = out.last_hidden_state[:, 0, :]
        e = F.normalize(e, p=2, dim=1)  
        vecs.append(e.cpu())
    return torch.cat(vecs, dim=0)


def retrieve(
    task: str,
    pairs: List[Tuple[str, object, int]],
    tokenizer: Any, # 使用 Any 类型避免 Pylance 报错
    model: Any,     # 使用 Any 类型避免 Pylance 报错
    device_str: str = "auto",
    top_k: int = 5,
    max_length: int = 256,
) -> List[dict]:
    """Retrieves the top_k most similar cases to a given task."""
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    
    model.to(device)

    keys = [p[0] for p in pairs] 
    key_vecs = embed_texts(keys, tokenizer, model, device, max_length=max_length)
    query_vec = embed_texts([task], tokenizer, model, device, max_length=max_length)[0].unsqueeze(0)

    sims = (query_vec @ key_vecs.T).squeeze(0) 
    k = min(top_k, len(pairs))
    topk_scores, topk_idx = torch.topk(sims, k)

    results = []
    for rank, (score, idx) in enumerate(zip(topk_scores.tolist(), topk_idx.tolist()), 1):
        key, value, line_index = pairs[idx] 
        results.append(
            {
                "rank": rank,
                "score": round(float(score), 6),
                "question": key,
                "plan": value,
                "line_index": line_index,  
            }
        )
    return results
