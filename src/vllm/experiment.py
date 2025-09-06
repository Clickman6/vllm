import os
import json
import time
import mlflow

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("VLLM_MODEL")
VLLM_HOST = os.getenv("VLLM_HOST")
VLLM_PORT = os.getenv("VLLM_PORT")

client = OpenAI(base_url=f"http://{VLLM_HOST}:{VLLM_PORT}/v1", api_key="EMPTY")

def load_dataset(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data

def run(model=MODEL, messages=[], dataset=[], name="Capital"):
    max_tokens = 8
    temperature = 0.2

    print(f"Running experiment: {name}")
    print(f"Start...")
    mlflow.set_experiment('vllm')
    with mlflow.start_run(run_name=name):
        mlflow.log_params({"model_id": model, "max_tokens": max_tokens, "temperature": temperature})

        mlflow.log_text(json.dumps(messages, ensure_ascii=False, indent=2), "instruction.json")
        mlflow.log_text(json.dumps(dataset, ensure_ascii=False, indent=2), "dataset.json")

        outputs = []
        for i, data in enumerate(dataset):
            print(f"\nProcessing {i+1}/{len(dataset)}")
            print(f"Q: {data['question']}")

            message = [{"role": "user", "content": data['question']}]
            output, metrics = request(model, messages + message, max_tokens, temperature)

            print(f"A: {output}")

            outputs.append({"answer": output})
            mlflow.log_metrics(metrics)

        mlflow.log_text(json.dumps(outputs, ensure_ascii=False, indent=2), "outputs.json")
        print(f"Done.")

def request(model, messages, max_tokens, temperature):
    try:
        start = time.perf_counter()
        completion = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=messages,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            }
        )

        latency = time.perf_counter() - start
        output = completion.choices[0].message.content

        metrics = {
            "latency_seconds": latency,
            "completion_tokens": completion.usage.completion_tokens,
            "prompt_tokens": completion.usage.prompt_tokens,
            "total_tokens": completion.usage.total_tokens,
            "output_length": len(output)
        }

        return (output, metrics)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    dataset = load_dataset("datasets/capitals.json")
    messages = [
        {"role": "system", "content": "Respond with exactly one word without punctuation, numbers, symbols, explanations"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "What is the capital of Russia?"},
        {"role": "assistant", "content": "Moscow"}
    ]

    run(name="Capital (No Instruction)", dataset=dataset)
    run(name="Capital", dataset=dataset, messages=messages)


