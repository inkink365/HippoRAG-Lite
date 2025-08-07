from copy import deepcopy
from typing import List, Optional
import os

import numpy as np
import requests
from tqdm import tqdm

from ..utils.config_utils import BaseConfig
from ..utils.logging_utils import get_logger
from .base import BaseEmbeddingModel, EmbeddingConfig, make_cache_embed

logger = get_logger(__name__)

class Qwen3EmbeddingModel(BaseEmbeddingModel):
    def __init__(self, global_config: Optional[BaseConfig] = None, embedding_model_name: Optional[str] = None) -> None:
        super().__init__(global_config=global_config)
        # 支持通过参数覆盖模型名称（解决"unexpected keyword argument 'embedding_model_name'"错误）
        if embedding_model_name is not None:
            self.embedding_model_name = embedding_model_name
            logger.debug(f"Overriding embedding model name with: {self.embedding_model_name}")
        self._init_embedding_config()
        # 初始化硅基流动API客户端
        self._init_siliconflow_client()

    def _init_siliconflow_client(self) -> None:
        """初始化硅基流动API连接参数"""
        # 从全局配置或环境变量获取API密钥
        self.api_key = self.global_config.api_key
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set")
        # 硅基流动嵌入API端点
        self.api_url = self.global_config.embedding_base_url
        # 请求头部（固定格式）
        self.headers = {
            "Authorization": self.api_key,
            "Content-Type": "application/json"
        }
        logger.debug("Silicon Flow API client initialized")

    def _init_embedding_config(self) -> None:
        """初始化嵌入模型配置（参考OpenAI实现）"""
        config_dict = {
            "embedding_model_name": self.embedding_model_name,
            "norm": self.global_config.embedding_return_as_normalized,
            "model_init_params": {
                "pretrained_model_name_or_path": self.embedding_model_name,
                # 硅基流动API无需本地模型参数，此处保留结构用于兼容性
            },
            "encode_params": {
                "max_length": self.global_config.embedding_max_seq_len or 8192,  # Qwen3支持长文本
                "batch_size": self.global_config.embedding_batch_size or 16,
                "num_workers": 32
            },
        }
        self.embedding_config = EmbeddingConfig.from_dict(config_dict=config_dict)
        logger.debug(f"Qwen3 embedding config initialized: {self.embedding_config}")

    def encode(self, texts: List[str]) -> np.ndarray:
        """单批文本编码（适配硅基流动API格式）"""
        # 文本预处理（同OpenAI逻辑）
        texts = [t.replace("\n", " ") for t in texts]
        texts = [t if t != "" else " " for t in texts]
        
        # 构造硅基流动API请求体
        payload = {
            "model": self.embedding_model_name,  # 模型名称，如"Qwen/Qwen3-Embedding-8B"
            "input": texts  # 支持单文本或文本列表
        }
        
        # 发送POST请求
        try:
            response = requests.post(
                url=self.api_url,
                headers=self.headers,
                json=payload,
                timeout=30  # 超时设置
            )
            response.raise_for_status()  # 触发HTTP错误（如401、404）
        except requests.exceptions.RequestException as e:
            logger.error(f"Silicon Flow API request failed: {str(e)}")
            raise
        
        # 解析响应
        result = response.json()
        if "data" not in result:
            raise ValueError(f"Invalid response from Silicon Flow API: {result}")
        
        # 提取嵌入向量并转换为numpy数组
        embeddings = np.array([item["embedding"] for item in result["data"]])
        return embeddings

    def batch_encode(self, texts: List[str], **kwargs) -> np.ndarray:
        """批量文本编码（分片处理大列表）"""
        if isinstance(texts, str):
            texts = [texts]
        params = deepcopy(self.embedding_config.encode_params)
        if kwargs:
            params.update(kwargs)
        
        batch_size = params.pop("batch_size", 16)
        logger.debug(f"Batch encoding with batch size: {batch_size}")
        
        if len(texts) <= batch_size:
            results = self.encode(texts)
        else:
            # 大列表分片处理，显示进度条
            pbar = tqdm(total=len(texts), desc="Qwen3 Batch Encoding")
            results = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                results.append(self.encode(batch))
                pbar.update(len(batch))
            pbar.close()
            results = np.concatenate(results)
        
        # 向量归一化（同OpenAI逻辑）
        if self.embedding_config.norm:
            results = (results.T / np.linalg.norm(results, axis=1)).T
        return results