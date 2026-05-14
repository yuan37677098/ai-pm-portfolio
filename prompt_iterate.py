import chromadb
import json
import time
from openai import OpenAI
from chromadb.utils import embedding_functions

# 初始化 RAG 组件
llm = OpenAI(
    api_key="sk-e245a81977704a6b8f3fe95032b927d2",
    base_url="https://api.deepseek.com"
)
chroma = chromadb.PersistentClient(path="./chroma_db")
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = chroma.get_or_create_collection(
    name="feishu_kb",
    embedding_function=ef
)

def retrieve(query, top_k=5):
    r = collection.query(query_texts=[query], n_results=top_k)
    return r['documents'][0], r['distances'][0]

def generate(query, system_prompt):
    docs, scores = retrieve(query)
    relevant = [d for d, s in zip(docs, scores) if s < 1.5]
    
    if not relevant:
        context = "（无相关文档）"
    else:
        context = "\n\n---\n\n".join(relevant)
    
    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt.replace("{context}", context)},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        stream=False
    )
    return response.choices[0].message.content

# 3 个 Prompt 版本
prompts = {
    "v1_简单版": """你是飞书客服助手。根据文档回答用户问题。文档中没有的就说不知道。

参考文档：
{context}""",

    "v2_加规则": """你是飞书官方客服助手。请严格按照以下规则回答：

1. 先看参考文档是否与用户问题相关
2. 相关 → 提取关键信息回答，200字以内
3. 不相关 → 回复"抱歉，我目前的知识库中没有这方面的信息。建议前往飞书帮助中心(feishu.cn/hc)查看更多内容。"
4. 回答中提及某个功能时，引用文档来源说明

参考文档：
{context}""",

    "v3_完整版": """你是飞书官方智能客服助手。你的名字是"飞小书"。

【核心规则】
1. 先判断参考文档是否包含用户问题的相关信息
2. 有相关信息 → 用简洁的语言回答，控制在200字以内，注明信息来源
3. 没有相关信息 → 回复"抱歉，我目前的知识库中暂未收录这方面的信息。建议您前往飞书帮助中心(feishu.cn/hc)搜索相关内容，或联系人工客服。"
4. 部分覆盖 → 回答已有信息，并诚实说明"关于XX部分，文档中暂时没有详细说明"
5. 无意义输入 → 友好引导："您好，我没有太理解您的问题。可以再描述一下吗？"

【回答风格】
- 亲切友好但不啰嗦
- 用自然口语，不要说"根据文档"这种生硬表述
- 不确定的细节标注"仅供参考"

参考文档：
{context}"""
}

# 加载测试题
with open("test_questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)

# 对每个版本跑全部测试题
for ver_name, ver_prompt in prompts.items():
    print("=" * 50)
    print("测试版本: " + ver_name)
    print("=" * 50)
    
    results = []
    for q in questions:
        t0 = time.time()
        answer = generate(q["question"], ver_prompt)
        elapsed = round(time.time() - t0, 2)
        
        r = {
            "id": q["id"],
            "question": q["question"],
            "type": q["type"],
            "answer": answer,
            "latency": elapsed
        }
        results.append(r)
        print("Q{} [{}] {}s".format(q["id"], q["type"], elapsed))
        print("  " + answer[:120])
        print()
        
        time.sleep(0.3)
    
    # 保存这轮的结果
    filename = "results_" + ver_name + ".json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("已保存: " + filename)
    print()

print("全部完成！3个版本的结果已保存。")
