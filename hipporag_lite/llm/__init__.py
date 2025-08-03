from .base import BaseLLM
from ..utils.logging_utils import get_logger
from ..utils.config_utils import BaseConfig
from .request_llm import SiliconFlowLLM  # 替换原OpenAI/Bedrock导入

logger = get_logger(__name__)

def _get_llm_class(config: BaseConfig):
    # 仅返回硅基流动LLM实例
    return SiliconFlowLLM.from_experiment_config(config)
    