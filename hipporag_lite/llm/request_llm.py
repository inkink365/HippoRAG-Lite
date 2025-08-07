import functools
import hashlib
import json
import os
import sqlite3
from copy import deepcopy
from typing import List, Tuple

import requests
from filelock import FileLock
from tenacity import retry, stop_after_attempt, wait_fixed

from ..utils.config_utils import BaseConfig
from ..utils.llm_utils import TextChatMessage
from ..utils.logging_utils import get_logger
from .base import BaseLLM, LLMConfig

logger = get_logger(__name__)

def cache_response(func):
    """复用缓存装饰器，基于请求参数哈希缓存响应"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        # 提取输入消息（缓存键的核心参数）
        if args:
            messages = args[0]
        else:
            messages = kwargs.get("messages")
        if messages is None:
            raise ValueError("Missing required 'messages' parameter for caching.")

        # 提取模型、温度等生成参数（用于缓存键）
        gen_params = self.llm_config.generate_params
        model = kwargs.get("model", gen_params.get("model"))
        temperature = kwargs.get("temperature", gen_params.get("temperature"))
        seed = kwargs.get("seed", gen_params.get("seed"))

        # 生成缓存键（哈希请求参数）
        key_data = {
            "messages": messages,
            "model": model,
            "temperature": temperature,
            "seed": seed,
        }
        key_str = json.dumps(key_data, sort_keys=True, default=str)
        key_hash = hashlib.sha256(key_str.encode("utf-8")).hexdigest()

        # 缓存文件锁（避免并发冲突）
        lock_file = self.cache_file_name + ".lock"

        # 尝试从缓存读取
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            conn.commit()
            c.execute("SELECT message, metadata FROM cache WHERE key = ?", (key_hash,))
            row = c.fetchone()
            conn.close()
            if row is not None:
                message, metadata_str = row
                return message, json.loads(metadata_str), True

        # 缓存未命中，调用原函数
        result = func(self, *args, **kwargs)
        message, metadata = result

        # 写入缓存
        with FileLock(lock_file):
            conn = sqlite3.connect(self.cache_file_name)
            c = conn.cursor()
            c.execute("""
                CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    message TEXT,
                    metadata TEXT
                )
            """)
            c.execute("INSERT OR REPLACE INTO cache (key, message, metadata) VALUES (?, ?, ?)",
                      (key_hash, message, json.dumps(metadata)))
            conn.commit()
            conn.close()

        return message, metadata, False
    return wrapper

def dynamic_retry_decorator(func):
    """复用动态重试装饰器"""
    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        dynamic_retry = retry(
            stop=stop_after_attempt(self.max_retries),
            wait=wait_fixed(2.718)  # 重试间隔e秒，可按需调整
        )
        decorated_func = dynamic_retry(func)
        return decorated_func(self, *args, **kwargs)
    return wrapper

class SiliconFlowLLM(BaseLLM):
    """基于requests库的硅基流动API调用实现"""
    @classmethod
    def from_experiment_config(cls, global_config: BaseConfig) -> "SiliconFlowLLM":
        """从全局配置创建实例"""
        cache_dir = os.path.join(global_config.save_dir, "llm_cache")
        return cls(cache_dir=cache_dir, global_config=global_config)

    def __init__(self, cache_dir: str, global_config: BaseConfig,** kwargs) -> None:
        super().__init__(global_config)
        self.cache_dir = cache_dir
        self.global_config = global_config

        # 硅基流动API核心配置
        self.llm_name = global_config.llm_name  # 模型名（如硅基流动的模型ID）
        self.api_url = global_config.llm_base_url  # 硅基流动API端点
        self.api_key = global_config.api_key # 硅基流动API密钥

        # 初始化缓存目录和文件
        os.makedirs(self.cache_dir, exist_ok=True)
        cache_filename = f"{self.llm_name.replace('/', '_')}_silicon_flow_cache.sqlite"
        self.cache_file_name = os.path.join(self.cache_dir, cache_filename)

        # 初始化LLM配置
        self._init_llm_config()

        # 重试配置
        self.max_retries = kwargs.get("max_retries", 3)

    def _init_llm_config(self) -> None:
        """初始化LLM配置（从全局配置提取参数）"""
        config_dict = self.global_config.__dict__
        # 生成参数（与硅基流动API对齐）
        config_dict['generate_params'] = {
            "model": self.llm_name,
            "temperature": config_dict.get("temperature", 0.0),
            "max_tokens": config_dict.get("max_new_tokens", 400),
            "n": config_dict.get("num_gen_choices", 1),
            "seed": config_dict.get("seed", 0),
        }
        self.llm_config = LLMConfig.from_dict(config_dict=config_dict)
        logger.debug(f"SiliconFlowLLM config initialized: {self.llm_config}")

    @cache_response
    @dynamic_retry_decorator
    def infer(
        self,
        messages: List[TextChatMessage],
        **kwargs
    ) -> Tuple[List[TextChatMessage], dict]:
        """调用硅基流动API进行同步推理"""
        # 构造请求参数（合并默认配置和临时参数）
        params = deepcopy(self.llm_config.generate_params)
        params.update(kwargs)
        params["messages"] = messages  # 输入对话历史

        # 构造HTTP请求头（含认证）
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"{self.api_key}"
        }

        # logger.debug(f"Sending request to Silicon Flow API: {self.api_url}, params: {params}")

        # 发送POST请求
        response = requests.post(
            url=self.api_url,
            headers=headers,
            json=params,
            timeout=300  # 5分钟超时（按需调整）
        )
        response.raise_for_status()  # 抛出HTTP错误（如401、500）
        response_data = response.json()

        # 解析响应
        response_message = response_data["choices"][0]["message"]["content"]
        assert isinstance(response_message, str), "response_message should be a string"

        metadata = {
            "prompt_tokens": response_data.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response_data.get("usage", {}).get("completion_tokens", 0),
            "finish_reason": response_data["choices"][0].get("finish_reason", "unknown"),
        }

        return response_message, metadata
