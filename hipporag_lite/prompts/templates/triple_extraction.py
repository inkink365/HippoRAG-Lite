from .ner import one_shot_ner_paragraph, one_shot_ner_output
from ...utils.llm_utils import convert_format_to_template

ner_conditioned_re_system = """你的任务是从给定的段落和命名实体列表中构建RDF（资源描述框架）图。
请以三元组的JSON列表形式回应，每个三元组代表RDF图中的一个关系。

请注意以下要求：
- 每个三元组应包含每个段落的实体列表中的至少一个实体，最好是两个。
- 明确解析代词的具体指代对象，以保持清晰性。

"""


ner_conditioned_re_frame = """将段落转换为JSON字典，包含命名实体列表和三元组列表。
段落：
```
{passage}
```

{named_entity_json}
"""

ner_conditioned_re_input = ner_conditioned_re_frame.format(passage=one_shot_ner_paragraph, named_entity_json=one_shot_ner_output)


ner_conditioned_re_output = """{"triples": [
            ["Radio City", "位于", "印度"],
            ["Radio City", "是", "私人调频电台"],
            ["Radio City", "成立于", "2001年7月3日"],
            ["Radio City", "播放", "印地语歌曲"],
            ["Radio City", "播放", "英语歌曲"],
            ["Radio City", "进军", "新媒体领域"],
            ["Radio City", "推出", "PlanetRadiocity.com"],
            ["PlanetRadiocity.com", "上线于", "2008年5月"],
            ["PlanetRadiocity.com", "是", "音乐门户网站"],
            ["PlanetRadiocity.com", "提供", "新闻"],
            ["PlanetRadiocity.com", "提供", "视频"],
            ["PlanetRadiocity.com", "提供", "歌曲"]
    ]
}
"""


prompt_template = [
    {"role": "system", "content": ner_conditioned_re_system},
    {"role": "user", "content": ner_conditioned_re_input},
    {"role": "assistant", "content": ner_conditioned_re_output},
    {"role": "user", "content": convert_format_to_template(original_string=ner_conditioned_re_frame, placeholder_mapping=None, static_values=None)}
]