ner_system = """你的任务是从给定段落中提取命名实体。
请以JSON列表的形式返回实体。
"""

one_shot_ner_paragraph = """Radio City
Radio City是印度第一家私人调频电台，于2001年7月3日开播。
它播放印地语、英语和地方语言歌曲。
Radio City最近于2008年5月进军新媒体领域，推出了音乐门户网站——PlanetRadiocity.com，该网站提供音乐相关新闻、视频、歌曲和其他音乐相关功能。"""


one_shot_ner_output = """{"named_entities":
    ["Radio City", "印度", "2001年7月3日", "印地语", "英语", "2008年5月", "PlanetRadiocity.com"]
}
"""


prompt_template = [
    {"role": "system", "content": ner_system},
    {"role": "user", "content": one_shot_ner_paragraph},
    {"role": "assistant", "content": one_shot_ner_output},
    {"role": "user", "content": "${passage}"}
]