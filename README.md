### HippoRAG 精简版
鉴于许多应用需求轻量级模块，同时api+cpu能够取得不错的效果，特此对hipporag项目进行了一些修改。同时对中文社区（siliconflow api）进行了深入的支持。
尽管相当不完善，但依然具有一定的可用性。

* v0.0.2更新，出于模块化考虑，我们去除了对环境变量的依赖，而是直接作为参数显式传入即可
* v0.0.3更新，汉化了提示词

#### 快速上手
```shell
conda create -n hipporag python=3.10

conda activate hipporag

pip install hipporag-lite
```
__示例：__
```python

from hipporag_lite import HippoRAG

# Prepare datasets and evaluation
docs = [
    "Oliver Badman is a politician.",
    "George Rankin is a politician.",
    "Thomas Marwick is a politician.",
    "Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince.",
    "Erik Hort's birthplace is Montebello.",
    "Marina is bom in Minsk.",
    "Montebello is a part of Rockland County."
]

save_dir = 'outputs'
llm_model_name = 'Pro/deepseek-ai/DeepSeek-V3' # 使用硅基流动的llm与embedding模型
embedding_model_name = 'Qwen/Qwen3-Embedding-8B'
llm_base_url = 'https://api.siliconflow.cn/v1/chat/completions'
embedding_base_url = 'https://api.siliconflow.cn/v1/embeddings'

# Startup a HippoRAG instance
try:
    hipporag = HippoRAG(api_key="Bearer sk-...", # 你的siliconflow api_key
                        save_dir=save_dir, 
                        llm_model_name=llm_model_name,
                        embedding_model_name=embedding_model_name,
                        llm_base_url=llm_base_url,
                        embedding_base_url=embedding_base_url)
    print("HippoRAG instance created successfully.")
except Exception as e:
    print(f"Error creating HippoRAG instance: {e}")

# Run indexing
try:
    hipporag.index(docs=docs)
    print("Indexing completed successfully.")
except Exception as e:
    print(f"Error during indexing: {e}")

# Separate Retrieval & QA
queries = [
    "What is George Rankin's occupation?",
    "How did Cinderella reach her happy ending?",
    "What county is Erik Hort's birthplace a part of?"
]

try:
    retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
    print("Retrieval completed successfully.")
except Exception as e:
    print(f"Error during retrieval: {e}")

try:
    qa_results = hipporag.rag_qa(retrieval_results)
    print("QA completed successfully.")
except Exception as e:
    print(f"Error during QA: {e}")

# Combined Retrieval & QA
try:
    rag_results = hipporag.rag_qa(queries=queries)
    print("Combined Retrieval & QA completed successfully.")
except Exception as e:
    print(f"Error during combined Retrieval & QA: {e}")

# For Evaluation
answers = [
    ["Politician"],
    ["By going to the ball."],
    ["Rockland County"]
]

gold_docs = [
    ["George Rankin is a politician."],
    ["Cinderella attended the royal ball.",
    "The prince used the lost glass slipper to search the kingdom.",
    "When the slipper fit perfectly, Cinderella was reunited with the prince."],
    ["Erik Hort's birthplace is Montebello.",
    "Montebello is a part of Rockland County."]
]

try:
    rag_results = hipporag.rag_qa(queries=queries, 
                                  gold_docs=gold_docs,
                                  gold_answers=answers)
    print(rag_results[3])
    print(rag_results[4])
    print("Evaluation completed successfully.")
except Exception as e:
    print(f"Error during evaluation: {e}")

```
```python
# windows 例程
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()

    from hipporag_lite import HippoRAG

    # Prepare datasets and evaluation
    docs = [
        "Oliver Badman is a politician.",
        "George Rankin is a politician.",
        "Thomas Marwick is a politician.",
        "Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince.",
        "Erik Hort's birthplace is Montebello.",
        "Marina is bom in Minsk.",
        "Montebello is a part of Rockland County."
    ]

    save_dir = 'outputs'
    llm_model_name = 'Pro/deepseek-ai/DeepSeek-V3' # 使用硅基流动的llm与embedding模型
    embedding_model_name = 'Qwen/Qwen3-Embedding-8B'
    llm_base_url = 'https://api.siliconflow.cn/v1/chat/completions'
    embedding_base_url = 'https://api.siliconflow.cn/v1/embeddings'

    # Startup a HippoRAG instance
    try:
        hipporag = HippoRAG(api_key="Bearer sk-...", # 你的siliconflow api_key
                            save_dir=save_dir, 
                            llm_model_name=llm_model_name,
                            embedding_model_name=embedding_model_name,
                            llm_base_url=llm_base_url,
                            embedding_base_url=embedding_base_url)
        print("HippoRAG instance created successfully.")
    except Exception as e:
        print(f"Error creating HippoRAG instance: {e}")

    # Run indexing
    try:
        hipporag.index(docs=docs)
        print("Indexing completed successfully.")
    except Exception as e:
        print(f"Error during indexing: {e}")

    # Separate Retrieval & QA
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    try:
        retrieval_results = hipporag.retrieve(queries=queries, num_to_retrieve=2)
        print("Retrieval completed successfully.")
    except Exception as e:
        print(f"Error during retrieval: {e}")

    try:
        qa_results = hipporag.rag_qa(retrieval_results)
        print("QA completed successfully.")
    except Exception as e:
        print(f"Error during QA: {e}")

    # Combined Retrieval & QA
    try:
        rag_results = hipporag.rag_qa(queries=queries)
        print("Combined Retrieval & QA completed successfully.")
    except Exception as e:
        print(f"Error during combined Retrieval & QA: {e}")

    # For Evaluation
    answers = [
        ["Politician"],
        ["By going to the ball."],
        ["Rockland County"]
    ]

    gold_docs = [
        ["George Rankin is a politician."],
        ["Cinderella attended the royal ball.",
        "The prince used the lost glass slipper to search the kingdom.",
        "When the slipper fit perfectly, Cinderella was reunited with the prince."],
        ["Erik Hort's birthplace is Montebello.",
        "Montebello is a part of Rockland County."]
    ]

    try:
        rag_results = hipporag.rag_qa(queries=queries, 
                                    gold_docs=gold_docs,
                                    gold_answers=answers)
        print(rag_results[3])
        print(rag_results[4])
        print("Evaluation completed successfully.")
    except Exception as e:
        print(f"Error during evaluation: {e}")

```

原项目主页：https://github.com/OSU-NLP-Group/HippoRAG
