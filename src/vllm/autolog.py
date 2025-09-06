import os
import json
import time
import mlflow
import pandas as pd

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("VLLM_MODEL")
mlflow.openai.autolog()
client = OpenAI()

def judge(question, answer, prediction):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a judge. Rate the model's answer from 0.0 to 1.0. "
                "Respond with a single number only. "
                "Focus on quality: correctness, completeness, clarity, and relevance. "
                "Do not focus on exact wording. "
                "Do not output anything except the number."
            )
        },

        {
            "role": "user",
            "content": "Question: What is the capital of France?\nModel's answer: Paris\nExpected answer: Paris"
        },
        {
            "role": "assistant",
            "content": "1.00"
        },
        {
            "role": "user",
            "content": "Question: What is the capital of Germany?\nModel's answer: The capital of Germany is Berlin\nExpected answer: Berlin"
        },
        {
            "role": "assistant",
            "content": "1.00"
        },

        {
            "role": "user",
            "content": (
                f"Question: {question}\n"
                f"Model's answer: {prediction}\n"
                f"Expected answer: {answer}"
            )
        }
    ]
    
    completion = client.chat.completions.create(
        model=MODEL,
        max_tokens=8,
        messages=messages,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        }
    )

    output = completion.choices[0].message.content
    try:
        result = float(output)
    except ValueError:
        result = 0.0

    return result

def run(model=MODEL, messages=[], dataset=[], name="Capital"):
    max_tokens = 10
    temperature = 0.2

    mlflow.set_experiment('vllm')
    print(f"Running experiment: {name}")
    print(f"Start...")
    with mlflow.start_run(run_name=name):
        mlflow.log_text(dataset.to_csv(index=False), "dataset.csv")
        mlflow.log_input(mlflow.data.from_pandas(dataset, source="datasets.csv", name=f"{name} Dataset"), context="testing")
        mlflow.log_text(json.dumps(messages, ensure_ascii=False, indent=2), "instruction.json")
        mlflow.log_params({"model_id": model, "max_tokens": max_tokens, "temperature": temperature})

        outputs = []
        for i, data in dataset.iterrows():
            print(f"\nProcessing {i+1}/{len(dataset)}")
            print(f"Q: {data['question']}")

            message = [{"role": "user", "content": data['question']}]
            output, metrics = request(model, messages + message, max_tokens, temperature)

            print(f"A: {output}")
            judge_scope = judge(data['question'], data['answer'], output)
            print(f"Judge score: {judge_scope}")

            outputs.append({"answer": output})
            mlflow.log_metrics(metrics, step=i)
            mlflow.log_metric('judge_scope', judge_scope, step=i)

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
    dataset = pd.read_csv("datasets/capitals.csv", delimiter=";")

    messages = [
        {"role": "system", "content": "Respond with exactly one word without punctuation, numbers, symbols, explanations"},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
        {"role": "user", "content": "What is the capital of Russia?"},
        {"role": "assistant", "content": "Moscow"}
    ]

    run(name="Capital (No Instruction)", dataset=dataset)
    run(name="Capital", dataset=dataset, messages=messages)
