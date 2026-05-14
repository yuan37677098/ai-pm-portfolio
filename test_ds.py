from openai import OpenAI

client = OpenAI(
    api_key="sk-e245a81977704a6b8f3fe95032b927d2",
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
