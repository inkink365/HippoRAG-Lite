def get_query_instruction(linking_method):
    instructions = {
        'ner_to_node': '给定一个短语，检索与之同义或相关的最佳匹配短语。',
        'query_to_node': '给定一个问题，检索该问题中提到的相关短语。',
        'query_to_fact': '给定一个问题，检索与该问题匹配的相关三元组事实。',
        'query_to_sentence': '给定一个问题，检索最能回答该问题的相关句子。',
        'query_to_passage': '给定一个问题，检索最能回答该问题的相关文档。',
    }
    default_instruction = '给定一个问题，检索最能回答该问题的相关文档。'
    return instructions.get(linking_method, default_instruction)