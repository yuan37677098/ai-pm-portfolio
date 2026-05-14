from openai import OpenAI

client = OpenAI(
    api_key="sk-9dcd469784584f40a18b0db200e25288",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "Hello, what is AI product manager?"}],
    stream=True
)

for chunk in response:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
