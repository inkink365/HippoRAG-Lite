from typing import List
import numpy as np
from tqdm import tqdm

def retrieve_knn(query_ids: List[str], key_ids: List[str], query_vecs, key_vecs, k=2047, 
                 query_batch_size=1000, key_batch_size=10000):
    """用numpy替代torch实现KNN检索"""
    # 向量归一化
    query_vecs = np.array(query_vecs, dtype=np.float32)
    query_vecs = query_vecs / np.linalg.norm(query_vecs, axis=1, keepdims=True)
    
    key_vecs = np.array(key_vecs, dtype=np.float32)
    key_vecs = key_vecs / np.linalg.norm(key_vecs, axis=1, keepdims=True)
    
    results = {}
    
    # 处理所有查询向量
    for query_start_idx in tqdm(range(0, len(query_vecs), query_batch_size), 
                                desc="Processing queries", 
                                total=(len(query_vecs) + query_batch_size - 1) // query_batch_size):
        query_end_idx = min(query_start_idx + query_batch_size, len(query_vecs))
        query_batch = query_vecs[query_start_idx:query_end_idx]
        
        # 初始化当前批次的top-k结果
        batch_topk_scores = np.full((len(query_batch), k), -np.inf, dtype=np.float32)
        batch_topk_indices = np.zeros((len(query_batch), k), dtype=np.int32)
        
        # 处理所有关键向量
        for key_start_idx in range(0, len(key_vecs), key_batch_size):
            key_end_idx = min(key_start_idx + key_batch_size, len(key_vecs))
            key_batch = key_vecs[key_start_idx:key_end_idx]
            
            # 计算批次相似度 (query_batch_size × key_batch_size)
            batch_similarity = np.dot(query_batch, key_batch.T)
            
            # 更新当前批次的top-k
            for i in range(len(query_batch)):
                # 获取当前查询的相似度
                similarities = batch_similarity[i]
                
                # 合并现有top-k和当前批次
                combined_scores = np.concatenate([batch_topk_scores[i], similarities])
                combined_indices = np.concatenate([
                    batch_topk_indices[i], 
                    np.arange(key_start_idx, key_end_idx)
                ])
                
                # 选择top-k
                topk_indices = np.argpartition(-combined_scores, k)[:k]
                top_scores = combined_scores[topk_indices]
                top_indices = combined_indices[topk_indices]
                
                # 按相似度降序排序
                sorted_order = np.argsort(-top_scores)
                batch_topk_scores[i] = top_scores[sorted_order]
                batch_topk_indices[i] = top_indices[sorted_order]
        
        # 保存当前查询批次的结果
        for i in range(len(query_batch)):
            query_idx = query_ids[query_start_idx + i]
            topk_indices = batch_topk_indices[i]
            topk_scores = batch_topk_scores[i]
            
            # 映射到实际ID
            topk_key_ids = [key_ids[idx] for idx in topk_indices]
            results[query_idx] = (topk_key_ids, topk_scores.tolist())
    
    return results