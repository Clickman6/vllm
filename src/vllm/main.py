import os
import json
import time
import http.client

import mlflow
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

MODEL = os.getenv("VLLM_MODEL")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")

HEADERS = {"Content-Type": "application/json"}

connection = http.client.HTTPConnection(VLLM_HOST, VLLM_PORT)
client = OpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="EMPTY")

def request(prompt, max_new_tokens=8):
    payload = {
        "model": MODEL,
        "max_tokens": max_new_tokens,
        "temperature": 0.2,
        "chat_template_kwargs": {"enable_thinking": False},
        "stop": ["\n"],
        "messages": [
            {"role": "system", "content": "Respond with exactly one word without punctuation, numbers, symbols, explanations"},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Paris"},
            {"role": "user", "content": "What is the capital of Russia?"},
            {"role": "assistant", "content": "Moscow"},
            {"role": "user", "content": prompt}
        ],
    }

    try:
        connection.request("POST", "/v1/chat/completions", headers=HEADERS, body=json.dumps(payload))

        res = connection.getresponse()
        data = res.read()
        response = json.loads(data.decode())

        output = response["choices"][0]["message"]["content"]
        print(f"http.client: Prompt: {prompt} → {output}")
    except Exception as e:
        print(f"Error: {e}")

def openai_api(prompt, max_new_tokens=8):
    try:
        completion = client.chat.completions.create(
            model=MODEL,
            max_tokens=max_new_tokens,
            temperature=0.2,
            stop=["\n"],
            messages=[
                {"role": "system", "content": "Respond with exactly the name of the capital without punctuation, numbers, symbols, or explanations."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "Paris"},
                {"role": "user", "content": "What is the capital of Russia?"},
                {"role": "assistant", "content": "Moscow"},
                {"role": "user", "content": prompt}
            ],
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )

        output = completion.choices[0].message.content

        print(f"openai: Prompt: {prompt} → {output}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    # request("What is the capital of Germany?")
    openai_api("What is the capital of India?")
