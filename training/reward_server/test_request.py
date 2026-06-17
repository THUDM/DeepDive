import json
import os

import requests


port = os.getenv("RM_TRAIN_PORT", "8888")
payload = {
    "history": [
        {"role": "user", "content": "Who wrote Pride and Prejudice?"},
        {"role": "assistant", "content": "The answer is Jane Austen."},
    ],
    "label": "Jane Austen",
    "task_unfinished": False,
    "remote_env_info": {
        "forbidden_texts": ["Who wrote Pride and Prejudice?"],
    },
}

resp = requests.post(f"http://127.0.0.1:{port}/evaluate", json=payload, timeout=60)
print(resp.status_code)
print(json.dumps(resp.json(), indent=2, ensure_ascii=False))
