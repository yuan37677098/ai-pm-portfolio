import chromadb
import json
from openai import OpenAI
from chromadb.utils import embedding_functions
import gradio as gr

# ---------- 初始化 RAG ----------
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
feedback_log = []

# ---------- RAG 核心 ----------
def retrieve(query, top_k=5):
    r = collection.query(query_texts=[query], n_results=top_k)
    docs = r['documents'][0]
    dists = r['distances'][0]
    relevant = []
    sources = set()
    for d, s, m in zip(docs, dists, r['metadatas'][0]):
        if s < 1.5:
            relevant.append(d)
            sources.add(m.get('source', 'unknown'))
    return relevant, list(sources)

def generate_response(query, relevant_docs):
    if not relevant_docs:
        return "抱歉，我目前的知识库中暂未收录这方面的信息。建议您前往飞书帮助中心(feishu.cn/hc)搜索相关内容，或联系人工客服。"

    context = "\n\n---\n\n".join(relevant_docs)

    system_prompt = """你是飞书官方智能客服助手，你的名字是"飞小书"。

【核心规则】
1. 先判断参考文档是否包含用户问题的相关信息
2. 有相关信息 → 用简洁的语言回答，控制在200字以内
3. 没有相关信息 → 回复"抱歉，我目前的知识库中暂未收录这方面的信息。"
4. 部分覆盖 → 回答已有信息，并诚实说明哪部分没有
5. 无意义输入 → 友好引导用户重新描述问题

【回答风格】
- 亲切友好但不啰嗦
- 用自然口语
- 不确定的细节标注"仅供参考"

参考文档：
""" + context

    response = llm.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ],
        temperature=0.3,
        stream=True
    )

    full_text = ""
    for chunk in response:
        if chunk.choices[0].delta.content:
            full_text += chunk.choices[0].delta.content
            yield full_text

def chat(message, history):
    # 空输入处理
    if not message or not message.strip():
        yield "您好！我是飞小书，飞书官方智能客服助手。请问有什么可以帮助您的？例如：\n\n• 如何修改密码？\n• 怎么创建群聊？\n• 多维表格怎么导出？"
        return

    # 火星文处理
    keywords = ["火星", "外星", "ufo", "异次元"]
    if any(k in message.lower() for k in keywords):
        yield "我暂时只支持地球上的飞书使用问题～换个话题试试？"
        return

    # RAG 检索 + 生成
    docs, sources = retrieve(message)

    response_text = ""
    for text in generate_response(message, docs):
        response_text = text
        yield text

    # 加引用
    if sources and response_text and "未收录" not in response_text:
        src_list = "、".join(sources[:3])
        final = response_text + "\n\n📎 参考来源：" + src_list
        yield final

# ---------- 反馈处理 ----------
def feedback_thumbs_up(value, history):
    if history:
        q, a = history[-1]
        feedback_log.append({"question": q, "answer": a, "feedback": "up"})
        with open("feedback_log.json", "w", encoding="utf-8") as f:
            json.dump(feedback_log, f, ensure_ascii=False, indent=2)

def feedback_thumbs_down(value, history):
    if history:
        q, a = history[-1]
        feedback_log.append({"question": q, "answer": a, "feedback": "down"})
        with open("feedback_log.json", "w", encoding="utf-8") as f:
            json.dump(feedback_log, f, ensure_ascii=False, indent=2)

# ---------- Gradio 界面 ----------
demo = gr.ChatInterface(
    fn=chat,
    title="飞小书 · AI 智能客服助手",
    description="基于飞书帮助文档的智能问答助手。试试问我飞书使用问题！",
    examples=[
        "如何修改密码？",
        "怎么创建群聊？",
        "多维表格能导出Excel吗？",
        "视频会议最多支持多少人？",
        "个人版和企业版有什么区别？",
    ],
    chatbot=gr.Chatbot(height=500, placeholder="输入你的飞书使用问题...")
)

if __name__ == "__main__":
    demo.launch(share=False)
