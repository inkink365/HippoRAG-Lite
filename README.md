### HippoRAG 精简版
鉴于许多应用需求轻量级模块，同时api+cpu能够取得不错的效果，特此对hipporag项目进行了一些修改。同时对中文社区（siliconflow api）进行了深入的支持。
尽管相当不完善，但依然具有一定的可用性。

* v0.0.2更新，出于模块化考虑，我们去除了对环境变量的依赖，而是直接作为参数显式传入即可
* v0.0.3更新，汉化了提示词
* v0.1.0更新，异步化三大常用函数（index，delete，retrieve），新增三大数据结构常用函数（save，size，clear）（异步）

#### 快速上手
```shell
conda create -n hipporag python=3.10

conda activate hipporag

pip install hipporag-lite
```
__示例：__
```python
import multiprocessing
import asyncio
import logging

# 定义一个异步主函数来处理所有操作
async def main():
    from hipporag_lite import HippoRAG

    # 准备数据集
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
    llm_model_name = 'Pro/deepseek-ai/DeepSeek-V3'
    embedding_model_name = 'Qwen/Qwen3-Embedding-8B'
    llm_base_url = 'https://api.siliconflow.cn/v1/chat/completions'
    embedding_base_url = 'https://api.siliconflow.cn/v1/embeddings'

    try:
        hipporag = HippoRAG(
            api_key="Bearer sk-...", # 你的siliconflow api_key
            save_dir=save_dir,
            llm_model_name=llm_model_name,
            embedding_model_name=embedding_model_name,
            llm_base_url=llm_base_url,
            embedding_base_url=embedding_base_url
        )
        print("HippoRAG实例创建成功")
        print(f"初始索引大小: {hipporag.size()} 文档")
    except Exception as e:
        print(f"创建HippoRAG实例失败: {e}")
        return

    # 异步处理索引操作
    try:
        await hipporag.index(docs=docs)  # 使用await调用异步方法
        print(f"索引操作完成，当前大小: {hipporag.size()} 文档")
    except Exception as e:
        print(f"索引失败: {e}")

    try:
        await hipporag.save()  # 异步保存
        print("系统状态保存成功")
    except Exception as e:
        print(f"保存失败: {e}")

    # 处理查询
    queries = [
        "What is George Rankin's occupation?",
        "How did Cinderella reach her happy ending?",
        "What county is Erik Hort's birthplace a part of?"
    ]

    try:
        retrieval_results = await hipporag.retrieve(queries=queries, num_to_retrieve=2)
        print(f"检索完成: 共处理 {len(retrieval_results)} 个查询")
    except Exception as e:
        logging.exception(e)
        print(f"检索失败: {e}")

    # 删除文档
    docs_to_delete = [
        "Oliver Badman is a politician.",
        "Thomas Marwick is a politician."
    ]
    
    try:
        print(f"删除前索引大小: {hipporag.size()} 文档")
        await hipporag.delete(docs_to_delete=docs_to_delete)  # 异步删除
        print(f"删除完成: 移除了 {len(docs_to_delete)} 个文档，当前大小: {hipporag.size()} 文档")
    except Exception as e:
        print(f"删除失败: {e}")

    # 清空系统
    try:
        print(f"清空前索引大小: {hipporag.size()} 文档")
        await hipporag.clear()  # 异步清空
        print(f"系统已清空，当前大小: {hipporag.size()} 文档")
    except Exception as e:
        logging.exception(e)
        print(f"清空失败: {e}")
    
    # 验证状态
    try:
        if hipporag.size() == 0:
            print("验证: 文档存储已清空")
        else:
            print(f"警告: 清空后仍有 {hipporag.size()} 个文档")
            
        if hipporag.graph.vcount() == 0:
            print("验证: 知识图谱已重置")
    except Exception as e:
        print(f"状态验证失败: {e}")

    print("所有操作完成")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 在主线程中运行异步主函数
    asyncio.run(main())

```

原项目主页：https://github.com/OSU-NLP-Group/HippoRAG
