from openai import OpenAI
import os

client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=os.getenv("HF_TOKEN")
)

r = client.chat.completions.create(
    model="microsoft/Phi-3.5-mini-instruct",
    messages=[{"role": "user", "content": "say hello"}],
    max_tokens=10
)

print(r.choices[0].message.content)