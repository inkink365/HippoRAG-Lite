import os
import asyncio
import importlib.util
from string import Template
from typing import Dict, List, Union, Any, Optional
from dataclasses import dataclass, field, asdict


from ..utils.logging_utils import get_logger

logger = get_logger(__name__)


@dataclass
class PromptTemplateManager:
    # templates_dir: Optional[str] = field(
    #     default=None, 
    #     metadata={"help": "包含模板脚本的目录。默认为定义此类的目录下的 `templates` 文件夹。"}
    # )
    role_mapping: Dict[str, str] = field(
        default_factory=lambda: {"system": "system", "user": "user", "assistant": "assistant"},
        metadata={"help": "从提示模板文件中的默认角色到特定LLM提供商定义的角色的映射。"}
    )
    templates: Dict[str, Union[Template, List[Dict[str, Any]]]] = field(
        init=False, 
        default_factory=dict,
        metadata={"help": "从提示模板名称到模板的字典。提示模板可以是Template实例或聊天历史（聊天历史是一个字典列表，其中包含作为Template实例的content）。"}
    )

    
    def __post_init__(self) -> None:
        """
        初始化模板目录并加载模板。
        """
        # if self.templates_dir is None:
        #     current_file_path = os.path.abspath(__file__)
        #     package_dir = os.path.dirname(current_file_path)
        #     self.templates_dir = os.path.join(package_dir, "templates")
        current_file_path = os.path.abspath(__file__)
        package_dir = os.path.dirname(current_file_path)
        
        # 每个*.py文件（排除__init__.py）包含一个prompt_template变量（字符串或内容为原始字符串的聊天历史，用于转换为Template）的目录的绝对路径
        self.templates_dir = os.path.join(package_dir, "templates") 

        self._load_templates()

    
    
    def _load_templates(self) -> None:
        """
        从模板目录中的Python脚本加载所有模板。
        """
        if not os.path.exists(self.templates_dir):
            logger.error(f"模板目录 '{self.templates_dir}' 不存在。")
            raise FileNotFoundError(f"模板目录 '{self.templates_dir}' 不存在。")
        
        
        logger.info(f"从目录加载模板：{self.templates_dir}")
        for filename in os.listdir(self.templates_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                script_name = os.path.splitext(filename)[0]

                try:
                    try:
                        module_name = f"hipporag_lite.prompts.templates.{script_name}"
                        module = importlib.import_module(module_name)
                    except ModuleNotFoundError:
                        module_name = f".prompts.templates.{script_name}"
                        module = importlib.import_module(module_name, 'hipporag_lite')

                    # spec = importlib.util.spec_from_file_location(script_name, script_path)
                    # module = importlib.util.module_from_spec(spec)
                    # spec.loader.exec_module(module)

                    if not hasattr(module, "prompt_template"):
                        logger.error(f"模块 '{module_name}' 未定义 'prompt_template'。")
                        raise AttributeError(f"模块 '{module_name}' 未定义 'prompt_template'。")

                    prompt_template = module.prompt_template
                    logger.debug(f"从 {module_name} 加载模板")
                    
                    if isinstance(prompt_template, Template):
                        self.templates[script_name] = prompt_template
                    elif isinstance(prompt_template, str):
                        self.templates[script_name] = Template(prompt_template)
                    elif isinstance(prompt_template, list) and all(
                        isinstance(item, dict) and "role" in item and "content" in item for item in prompt_template
                    ):
                        # 根据提供的角色映射调整角色
                        for item in prompt_template:
                            item["role"] = self.role_mapping.get(item["role"], item["role"])
                            item["content"] = item["content"] if isinstance(item["content"], Template) else Template(item["content"])
                        self.templates[script_name] = prompt_template
                    else:
                        raise TypeError(
                            f"'{module_name}.py' 中的prompt_template格式无效。必须是Template或List[Dict]。"
                        )

                    logger.debug(f"成功从 '{module_name}.py' 加载模板 '{script_name}'。")

                except Exception as e:
                    logger.error(f"从 '{module_name}.py' 加载模板失败：{e}")
                    raise

    def render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        """
        使用提供的变量渲染模板。

        参数:
            name (str): 模板的名称。
            kwargs: 模板的占位符值。

        返回:
            Union[str, List[Dict[str, Any]]]: 渲染后的模板或聊天历史。

        异常:
            ValueError: 如果缺少必需的变量。
        """
        template = self.get_template(name)
        if isinstance(template, Template):
            # 渲染单个字符串模板
            try:
                result = template.substitute(**kwargs)
                logger.debug(f"使用变量 {kwargs} 成功渲染模板 '{name}'。")
                return result
            except KeyError as e:
                logger.error(f"模板 '{name}' 缺少变量：{e}")
                raise ValueError(f"模板 '{name}' 缺少变量：{e}")
        elif isinstance(template, list):
            # 渲染聊天历史
            try:
                rendered_list = [
                    {"role": item["role"], "content": item["content"].substitute(**kwargs)}
                    for item in template
                ]
                logger.debug(f"使用变量 {kwargs} 成功渲染聊天历史模板 '{name}'。")
                return rendered_list
            except KeyError as e:
                logger.error(f"聊天历史模板 '{name}' 缺少变量：{e}")
                raise ValueError(f"聊天历史模板 '{name}' 缺少变量：{e}")
    
    def sync_render(self, name: str, **kwargs) -> Union[str, List[Dict[str, Any]]]:
        return asyncio.run(self.render(name, **kwargs))

    def list_template_names(self) -> List[str]:
        """
        列出所有可用的模板名称。

        返回:
            List[str]: 模板名称列表。
        """
        logger.info("列出所有可用的模板名称。")
        
        return list(self.templates.keys())

    def get_template(self, name: str) -> Union[Template, List[Dict[str, Any]]]:
        """
        通过名称检索模板。

        参数:
            name (str): 模板的名称。

        返回:
            Union[Template, List[Dict[str, Any]]]: 请求的模板。

        异常:
            KeyError: 如果未找到模板。
        """
        if name not in self.templates:
            logger.error(f"未找到模板 '{name}'。")
            raise KeyError(f"未找到模板 '{name}'。")
        logger.debug(f"检索到模板 '{name}'。")
        
        return self.templates[name]
    
    def print_template(self, name: str) -> None:
        """
        打印给定模板名称的提示模板字符串或聊天历史结构。

        参数:
            name (str): 模板的名称。

        异常:
            KeyError: 如果未找到模板。
        """
        try:
            template = self.get_template(name)
            print(f"模板名称: {name}")
            if isinstance(template, Template):
                print(template.template)
            elif isinstance(template, list):
                for item in template:
                    print(f"角色: {item['role']}, 内容: {item['content']}")
            logger.info(f"已打印模板 '{name}'。")
        except KeyError as e:
            logger.error(f"打印模板 '{name}' 失败：{e}")
            raise
    
    
    def is_template_name_valid(self, name: str) -> bool:
        return name in self.templates