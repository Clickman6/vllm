import json
import time
import http.client

HEADERS = {"Content-Type": "application/json"}

connection = http.client.HTTPConnection("http://llm.localhost")

def send_request(prompt, max_new_tokens=8):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens
    }

    try:
        connection.request("POST", "/generate", headers=HEADERS, body=json.dumps(payload))
        res = connection.getresponse()
        data = res.read()
        response = json.loads(data.decode())
        print(f"Prompt: {prompt} → {response['output'][:50]}...")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    for i in range(30):
        send_request(f"ping {i}")
        time.sleep(0.2)  # 5 запросов в секунду