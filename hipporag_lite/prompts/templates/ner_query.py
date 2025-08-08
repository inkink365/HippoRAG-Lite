ner_system = """你是一个高效的实体提取系统。
"""

query_prompt_one_shot_input = """请提取对解决以下问题至关重要的所有命名实体。
将命名实体以JSON格式呈现。

问题：《Arthur's Magazine》和《First for Women》哪本杂志创办得更早？

"""
query_prompt_one_shot_output = """
{"named_entities": ["《First for Women》", "《Arthur's Magazine》"]}
"""

prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": query_prompt_one_shot_input},
    {"role": "assistant", "content": query_prompt_one_shot_output},
    {"role": "user", "content": "问题：${query}"}
]