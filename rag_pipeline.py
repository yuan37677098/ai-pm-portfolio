import chromadb
import json
from openai import OpenAI
from chromadb.utils import embedding_functions

class RAG:
    def __init__(self):
        self.llm = OpenAI(
            api_key="sk-9dcd469784584f40a18b0db200e25288",
            base_url="https://api.deepseek.com"
        )
        
        self.chroma = chromadb.PersistentClient(path="./chroma_db")
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        self.collection = self.chroma.get_or_create_collection(
            name="feishu_kb",
            embedding_function=self.ef
        )
    
    def retrieve(self, query, top_k=5):
        r = self.collection.query(query_texts=[query], n_results=top_k)
        return r['documents'][0], r['distances'][0]
    
    def ask(self, query):
        docs, scores = self.retrieve(query)
        
        relevant = [d for d, s in zip(docs, scores) if s < 1.5]
        if not relevant:
            print("知识库中没有相关信息，建议联系人工客服。")
            return
        
        context = "\n\n---\n\n".join(relevant)
        
        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是飞书客服助手。根据以下飞书帮助文档回答用户问题。如果文档有答案就引用。如果文档没有就说不知道。参考文档：\n" + context},
                {"role": "user", "content": query}
            ],
            temperature=0.3,
            stream=True
        )
        
        for chunk in response:
            if chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="")
        print()

rag = RAG()
print("飞书AI客服就绪，输入问题试试（输入 q 退出）\n")

while True:
    q = input("用户: ")
    if q == "q":
        break
    print("客服: ", end="")
    rag.ask(q)
    print()
