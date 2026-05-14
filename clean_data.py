import os
import re

def clean(text):
    # 去掉连续3个以上空行
    text = re.sub(r'\n{3,}', '\n\n', text)
    # 去掉残留 HTML 标签
    text = re.sub(r'<[^>]+>', '', text)
    # 去掉乱码控制字符
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
