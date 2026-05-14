import chromadb
import json

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
