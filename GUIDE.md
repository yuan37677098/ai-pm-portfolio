# AI PM 作品集：完整操作指南

## 项目概述

做一个 **AI 智能客服助手**（飞书/钉钉/任意产品），用 RAG 技术实现：用户提问 → 检索知识库 → LLM 生成回答。从数据到产品全流程自己走一遍。

---

## 目录

1. 环境搭建
2. API 验证
3. 数据采集与清洗
4. 文档切片
5. 向量索引建库
6. RAG 核心流水线
7. Prompt 迭代
8. 评估体系
9. 模型选型对比
10. Demo 产品化
11. 作品集包装
12. 上传 GitHub
13. 录 Demo 视频
14. 面试使用指南

---

## 一、环境搭建

```bash
# Anaconda Prompt 中执行
conda create -n ai-pm python=3.10 -y
conda activate ai-pm
mkdir ai-pm-portfolio
cd ai-pm-portfolio
pip install openai chromadb gradio pandas numpy python-dotenv sentence-transformers
```

> ⚠️ **特别注意**：Python 3.13 太新，部分库不兼容，必须用 3.10。每次打开新终端先执行 `conda activate ai-pm`。

---

## 二、API 验证

### DeepSeek（主力模型）

```python
# test_ds.py
from openai import OpenAI

client = OpenAI(
    api_key="你的DeepSeek密钥",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "用一句话解释什么是AI产品经理"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

### GLM（备选/对比模型）

```python
# test_glm.py
from openai import OpenAI

client = OpenAI(
    api_key="你的GLM密钥",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

response = client.chat.completions.create(
    model="glm-4-flash",
    messages=[{"role": "user", "content": "你好"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

> ⚠️ **特别注意**：DeepSeek 和 GLM 都兼容 OpenAI SDK，只需改 `base_url` 和 `api_key`。在 VSCode 中新建 `.py` 文件，在 Anaconda Prompt 中执行。

---

## 三、数据采集与清洗

### 3.1 采集

**方法**：浏览器打开目标产品的帮助中心 → 打开文章 → `Ctrl+A` 全选 → `Ctrl+C` 复制 → VSCode 新建文件 → `Ctrl+V` 粘贴 → 保存到 `data/doc_001.txt`。

> ⚠️ **特别注意**：如果帮助中心是图文教程，图片里的文字会丢失。选纯文本知识源更好，或者自己编 10 篇补充文档（项目实战中完全合理）。

### 3.2 合成数据（补盲区）

当真实文档覆盖不全时，自己写补充文档。参见完整指南中 `make_docs.py`。

> 💡 **关键能力**：PM 要能判断数据盲区——哪些主题文档里缺了，用合成数据填补。面试时坦诚讲"知识库覆盖面不够，补了合成数据"，这体现数据意识。

### 3.3 清洗

```python
# clean_data.py
import os
import re

def clean(text):
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', '', text)
    return text.strip()

total = 0
for fname in sorted(os.listdir("data")):
    if fname.endswith(".txt"):
        p = f"data/{fname}"
        with open(p, "r", encoding="utf-8") as f:
            raw = f.read()
        cleaned = clean(raw)
        with open(p, "w", encoding="utf-8") as f:
            f.write(cleaned)
        print(f"OK {fname}: {len(cleaned)} 字")
        total += len(cleaned)
print(f"\n总计: {total} 字, 约 {total // 400} 个文本块")
```

> ⚠️ **特别注意**：Windows 中文路径、VSCode 编码问题会导致乱码。遇到乱码用记事本（`notepad xxx.py`）重新保存，不要用 VSCode 修复。

---

## 四、文档切片（Chunking）

```python
# chunk_docs.py
import os, json

def chunk_text(text, chunk_size=500):
    paragraphs = text.split("\n\n")
    chunks = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= chunk_size:
            chunks.append(para)
        else:
            sentences = para.replace("。", "。|").replace("！", "！|").replace("？", "？|").split("|")
            current = ""
            for s in sentences:
                if len(current) + len(s) <= chunk_size:
                    current += s
                else:
                    if current:
                        chunks.append(current)
                    current = s
            if current:
                chunks.append(current)
    return chunks

all_chunks = []
for fname in sorted(os.listdir("data")):
    if fname.endswith(".txt"):
        with open(f"data/{fname}", "r", encoding="utf-8") as f:
            text = f.read()
        subs = chunk_text(text)
        for sub in subs:
            all_chunks.append({"content": sub, "source": fname})
        print(f"OK {fname}: {len(subs)} 块")

with open("chunks.json", "w", encoding="utf-8") as f:
    json.dump(all_chunks, f, ensure_ascii=False, indent=2)
print(f"\n总计 {len(all_chunks)} 块")
```

> 💡 **关键知识**：Chunk Size 一般选 300-800。太大 → 检索不精确；太小 → 缺少上下文。500 是中文文档的常用值。

---

## 五、向量索引建库

```python
# build_index.py
import chromadb, json

with open("chunks.json", "r", encoding="utf-8") as f:
    chunks = json.load(f)

chroma_client = chromadb.PersistentClient(path="./chroma_db")

from chromadb.utils import embedding_functions
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

collection = chroma_client.get_or_create_collection(
    name="feishu_kb",
    embedding_function=ef
)

for i, c in enumerate(chunks):
    collection.add(
        documents=[c["content"]],
        metadatas=[{"source": c["source"]}],
        ids=[f"c_{i}"]
    )
    print("OK {}/{}: {}".format(i+1, len(chunks), c['source']))

print("Done! Total: " + str(collection.count()))
```

> ⚠️ **特别注意**：
> - HuggingFace 在国内被墙，运行前必须 `set HF_ENDPOINT=https://hf-mirror.com`
> - 集合名必须 3-63 字符，`kb` 太短会报错，用 `feishu_kb`
> - 如果内存不够（报 memory allocation），用 `all-MiniLM-L6-v2` 而不是 `paraphrase-multilingual-MiniLM-L12-v2`

---

## 六、RAG 核心流水线

```python
# rag_pipeline.py
import chromadb, json
from openai import OpenAI
from chromadb.utils import embedding_functions

class RAG:
    def __init__(self):
        self.llm = OpenAI(
            api_key="你的DeepSeek密钥",
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
                {"role": "system", "content": "你是飞书客服助手..." + context},
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
while True:
    q = input("用户: ")
    if q == "q":
        break
    print("客服: ", end="")
    rag.ask(q)
    print()
```

> 💡 **关键知识**：
> - `temperature=0.3`：客服场景偏确定，不宜太高
> - `top_k=5`：召回 5 条文档块，太少可能漏，太多噪声多
> - 距离阈值 `s < 1.5`：超过 1.5 说明不相关（all-MiniLM 的余弦距离）
> - `stream=True`：流式输出，用户不等整个答案

---

## 七、Prompt 迭代（3 轮）

### 7.1 建测试题库

```json
// test_questions.json
[
  {"id":1, "question":"如何修改密码？", "type":"正常"},
  {"id":2, "question":"怎么创建群聊？", "type":"正常"},
  {"id":5, "question":"怎么发起审批？", "type":"正常"},
  {"id":7, "question":"个人版和企业版有什么区别？", "type":"边界"},
  {"id":10, "question":"怎么在飞书里点外卖？", "type":"盲区"},
  {"id":13, "question":"火星三日游怎么规划？", "type":"乱输入"},
  {"id":14, "question":" ", "type":"空"}
]
```

**5 种题型必须覆盖**：正常、边界、盲区、乱输入、空输入。

### 7.2 迭代脚本

核心逻辑：同一个测试题集，用 3 个不同的 System Prompt 各跑一遍，对比结果。

### 7.3 Prompt v3（最优版本）

```
你是飞书官方智能客服助手，你的名字是"飞小书"。

【核心规则】
1. 先判断参考文档是否包含用户问题的相关信息
2. 有 → 用简洁语言回答，200字以内，注明来源
3. 没有 → "抱歉，我目前的知识库中暂未收录这方面的信息。建议前往帮助中心搜索或联系人工客服。"
4. 部分覆盖 → 诚实说明哪部分没有
5. 无意义输入 → 友好引导重新描述问题

【回答风格】
- 亲切友好但不啰嗦
- 用自然口语
```

> 💡 **关键能力**：Prompt 优化的三个方向——角色设定（人设）、规则分层（行为约束）、兜底话术（边界处理）。v3 比 v1/v2 好在这三个方面都有。

---

## 八、评估体系（AI 裁判打分）

```python
# eval_judge.py 核心逻辑
def score(question, answer, qtype):
    prompt = """你是一个AI客服质量评估员。请从5个维度打分（1-5）：
- 准确性：事实是否正确、有无编造
- 完整性：是否覆盖了用户的信息需求
- 流畅度：表达是否自然、好懂
- 安全性：有无风险内容或过度承诺
- 盲区处理：不知道时是否诚实并给出替代建议
输出JSON：{"准确性":N,"完整性":N,"流畅度":N,"安全性":N,"盲区处理":N,"总体":N}"""
    
    r = judge.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return r.choices[0].message.content
```

> 💡 **关键知识**：
> - GPT-as-judge 是目前 AI 产品评估的主流方法
> - 准确性、安全性打 5 分不难，**完整性最难**（知识库天花板）
> - 按题型分层统计（正常/边界/盲区），各有各的预期

---

## 九、模型选型对比

```python
# model_compare.py 核心
models = [
    {
        "name": "DeepSeek-Chat",
        "client": OpenAI(api_key="ds-key", base_url="https://api.deepseek.com"),
        "model_id": "deepseek-chat",
        "price_input": 1.0,    # 元/百万Token
        "price_output": 2.0
    },
    {
        "name": "GLM-4-Flash",
        "client": OpenAI(api_key="glm-key", base_url="https://open.bigmodel.cn/api/paas/v4/"),
        "model_id": "glm-4-flash",
        "price_input": 0.0,
        "price_output": 0.0
    }
]
```

对比维度：**延迟、Token 消耗、单次成本、月成本预估**。

> 💡 **关键能力**：AI PM 不是在选"最好的模型"，而是在选"够用且便宜的"。三角权衡：质量 vs 延迟 vs 成本。

---

## 十、Demo 产品化（Gradio）

```python
# app.py 核心
demo = gr.ChatInterface(
    fn=chat,
    title="飞小书 · AI 智能客服助手",
    description="基于飞书帮助文档的智能问答助手",
    examples=["如何修改密码？", "怎么创建群聊？", ...],
    chatbot=gr.Chatbot(height=500, placeholder="输入你的问题...")
)
demo.launch(share=False)
```

> ⚠️ **特别注意**：
> - 新版 Gradio 不支持 `theme="soft"` 参数，遇到 `TypeError` 删掉它
> - 空输入、火星文等边界 case 要在 `chat()` 函数里提前判断，不进 RAG
> - `share=False` 只在本机跑，`share=True` 生成公网链接（有时被墙）
> - 流式输出用 `yield` 而不是 `return`

---

## 十一、作品集包装（README）

参见完整指南中的 README.md 模板。8 个必写部分：

1. 项目背景
2. 技术架构
3. 关键数据
4. Prompt 迭代记录
5. 模型选型对比
6. Bad Case 发现
7. 产品化规划
8. 个人反思

---

## 十二、上传 GitHub

```bash
# 在项目目录下
echo chroma_db/ > .gitignore
echo venv/ >> .gitignore

git init
git add .
git commit -m "feat: AI智能客服助手 - RAG端到端项目"
git branch -M main
git remote add origin https://github.com/你的用户名/ai-pm-portfolio.git
git push -u origin main
```

> ⚠️ **特别注意**：
> - `chroma_db/` 目录通常几百 MB，必须写入 `.gitignore`，否则推不上去
> - 大文件（>100MB）GitHub 会拒绝，不要提交视频到 Git，用网盘链接放 README
> - 推送时如果弹登录框，用 GitHub 账号密码不行，需要去 GitHub → Settings → Developer settings → Personal access tokens → 创建一个 token，用 token 作为密码

---

## 十三、Demo 视频脚本（3 分钟）

| 时间段 | 内容 | 操作 |
|--------|------|------|
| 0:00-0:25 | 开场介绍 | 展示页面，说清项目是什么 |
| 0:25-1:20 | 3 个正常问题 | 展示流式输出 + 引用来源 |
| 1:20-2:00 | 边界 + 盲区测试 | 点外卖（盲区）、火星（乱输入） |
| 2:00-2:25 | 反馈功能 | 点 👍 |
| 2:25-2:50 | 技术架构 | 切 VSCode 展示文件结构 |
| 2:50-3:00 | 收尾 | 一句话总结感悟 |

> **录制**：Win + G → Win + Alt + R 开始/停止录制，视频在 `此电脑/视频/捕获/`

---

## 十四、面试使用指南

### 面试前准备

1. 确认 `app.py` 本地能启动
2. 打开 `eval_summary.json` 看一眼关键数字
3. 打开 `README.md` 在浏览器预览

### 面试 10 问速查表

| 问题 | 你亮什么 |
|------|----------|
| 你做过 AI 产品吗？ | 打开 Demo 给他试用 |
| 怎么评估 AI 质量？ | `eval_summary.json` — 6 维度 AI 裁判打分 |
| 怎么优化效果？ | 3 轮 Prompt 迭代记录 + Bad Case |
| 怎么选模型？ | `model_comparison.json` — 成本/质量/延迟三角 |
| 你懂技术吗？ | RAG 架构 → Chunking → 向量检索 → LLM 生成 |
| 怎么控制幻觉？ | RAG 有据才答，不知道就说不知道 |
| 怎么处理 Bad Case？ | 数据 vs 检索 vs Prompt 三分法归因 |
| 成本怎么算？ | 单次 ¥0.0003，月 50 万次 ¥150 |
| 数据飞轮怎么转？ | 👍👎 → Bad Case 库 → 改进 Prompt → 回归验证 |
| 最大收获？ | 数据是第一公民，PM 在成本/质量/延迟间权衡 |

### 避坑指南

| 常见翻车 | 预防 |
|----------|------|
| Demo 现场打不开 | 提前启动 `app.py`，放浏览器后台 |
| 忘掉关键数字 | 面试前 5 分钟看一遍 README |
| 被问"和其他 AI 产品有什么区别" | 强调你是从数据到产品全流程，不是只画原型 |
| 被质疑"你没算法背景" | "我会评估模型质量、管理数据飞轮、做成本分析——这些是工程做不到的" |

---

## 附录：完整文件依赖清单

```
pip install openai chromadb gradio pandas numpy python-dotenv sentence-transformers
```

## 附录：每次打开项目的启动命令

```bash
conda activate ai-pm
cd C:\Users\裕安\ai-pm-portfolio
set HF_ENDPOINT=https://hf-mirror.com
python app.py
```

---

全文结束。这份指南覆盖了你从零到作品集完成的所有步骤。
