import json
import time
from openai import OpenAI

# 加载测试题（挑 5 条代表性的）
questions = [
    {"id": 1, "q": "如何修改飞书登录密码？", "type": "正常"},
    {"id": 6, "q": "多维表格能导出Excel吗？", "type": "正常"},
    {"id": 7, "q": "个人版和企业版有什么区别？", "type": "边界"},
    {"id": 10, "q": "怎么在飞书里点外卖？", "type": "盲区"},
    {"id": 14, "q": " ", "type": "空"},
]

# 模型配置
models = [
    {
        "name": "DeepSeek-Chat",
        "client": OpenAI(
            api_key="sk-e245a81977704a6b8f3fe95032b927d2",
            base_url="https://api.deepseek.com"
        ),
        "model_id": "deepseek-chat",
        "price_input": 1.0,   # 元/百万Token
        "price_output": 2.0
    },
    {
        "name": "GLM-4-Flash",
        "client": OpenAI(
            api_key="cb756f47713e48518b43a3ce320afa22.RAAAR2m2gh4Ac6Dy",
            base_url="https://open.bigmodel.cn/api/paas/v4/"
        ),
        "model_id": "glm-4-flash",
        "price_input": 0.0,   # 免费
        "price_output": 0.0
    }
]

import chromadb
from chromadb.utils import embedding_functions

chroma = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma.get_or_create_collection(
    name="feishu_kb",
    embedding_function=ef
)

def retrieve(query):
    r = collection.query(query_texts=[query], n_results=5)
    docs = r['documents'][0]
    dists = r['distances'][0]
    return [d for d, s in zip(docs, dists) if s < 1.5]

system_prompt_base = """你是飞书客服助手。根据以下文档回答问题。
如果文档有答案就引用，文档没有就说不知道。
回答控制在200字以内。

参考文档：
{context}"""

all_results = []

for model_info in models:
    print("=" * 50)
    print("测试模型: " + model_info["name"])
    print("=" * 50)

    for q in questions:
        docs = retrieve(q["q"])
        context = "\n\n---\n\n".join(docs) if docs else "（无相关文档）"
        system = system_prompt_base.replace("{context}", context)

        t0 = time.time()
        try:
            r = model_info["client"].chat.completions.create(
                model=model_info["model_id"],
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": q["q"]}
                ],
                temperature=0.3,
                stream=False
            )
            answer = r.choices[0].message.content
            latency = round(time.time() - t0, 2)
            input_tokens = r.usage.prompt_tokens
            output_tokens = r.usage.completion_tokens
            cost = round((input_tokens * model_info["price_input"] + 
                         output_tokens * model_info["price_output"]) / 1_000_000, 4)

            result = {
                "model": model_info["name"],
                "id": q["id"],
                "question": q["q"],
                "type": q["type"],
                "answer": answer[:200],
                "latency_s": latency,
                "tokens_in": input_tokens,
                "tokens_out": output_tokens,
                "cost_yuan": cost
            }
        except Exception as e:
            result = {
                "model": model_info["name"],
                "id": q["id"],
                "question": q["q"],
                "error": str(e)
            }

        all_results.append(result)
        print("Q{} [{}] {}s, {}tokens, {}元".format(
            q["id"], q["type"], 
            result.get("latency_s", "?"),
            result.get("tokens_in", 0) + result.get("tokens_out", 0),
            result.get("cost_yuan", "?")
        ))
        time.sleep(0.3)

# 汇总
print("\n" + "=" * 50)
print("=== 模型对比汇总 ===")
print("=" * 50)

for model_name in ["DeepSeek-Chat", "GLM-4-Flash"]:
    items = [r for r in all_results if r["model"] == model_name and "error" not in r]
    if items:
        avg_latency = round(sum(r["latency_s"] for r in items) / len(items), 2)
        avg_tokens = round(sum(r["tokens_in"] + r["tokens_out"] for r in items) / len(items), 0)
        avg_cost = round(sum(r["cost_yuan"] for r in items) / len(items), 4)
        monthly_cost = round(avg_cost * 500000, 0)  # 假设月50万次
        print("{}: 均延迟{}s | 均Token{} | 均成本{}元/次 | 月50万次成本≈{}元".format(
            model_name, avg_latency, avg_tokens, avg_cost, monthly_cost
        ))

with open("model_comparison.json", "w", encoding="utf-8") as f:
    json.dump(all_results, f, ensure_ascii=False, indent=2)

print("\n已保存: model_comparison.json")
