import os
import json

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

print(f"\n总计 {len(all_chunks)} 个文本块 → chunks.json")
