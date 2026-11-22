# -*- coding: utf-8 -*-
"""Web-intent probe: time-sensitive question through the API with web on."""
from __future__ import annotations

import json
import time
import urllib.request

API = "http://127.0.0.1:8000/v1/chat/completions"
KEY = "local-dev-key"

body = json.dumps({
    "model": "extra-code",
    "messages": [{"role": "user", "content": "Какой сейчас курс доллара к рублю?"}],
    "max_tokens": 300,
    "web": True,
    "web_pages": 3,
}).encode("utf-8")
req = urllib.request.Request(API, data=body, headers={
    "Content-Type": "application/json",
    "Authorization": f"Bearer {KEY}",
})
t0 = time.perf_counter()
with urllib.request.urlopen(req, timeout=130.0) as r:
    data = json.loads(r.read().decode("utf-8"))
dt = time.perf_counter() - t0
msg = data["choices"][0]["message"]
print(f"time: {dt:.1f}s")
print("answer:", msg["content"][:400])
print("sources:", data.get("sources", msg.get("sources", "n/a")))
