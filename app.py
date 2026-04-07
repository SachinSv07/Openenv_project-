from __future__ import annotations

import html
import io
import os
import threading
import uvicorn
from contextlib import redirect_stdout
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from env import SupportEnv
from baseline import run_baseline
from models import Action

app = FastAPI()
env = SupportEnv(task_level=os.getenv("OPENENV_TASK", "hard"))

RESULT_LOG = "Starting benchmark run..."
RESULT_LOCK = threading.Lock()

def _run_baseline_once():
    global RESULT_LOG
    buffer = io.StringIO()
    try:
        with redirect_stdout(buffer):
            run_baseline(task_level=os.getenv("OPENENV_TASK", "hard"))
        text = buffer.getvalue().strip()
    except Exception as exc:
        text = f"Runtime error while executing baseline: {{exc}}".format(exc=str(exc))

    with RESULT_LOCK:
        RESULT_LOG = text or "No output generated."

@app.on_event("startup")
async def startup_event():
    worker = threading.Thread(target=_run_baseline_once, daemon=True)
    worker.start()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/reset")
def reset():
    return env.reset()

@app.post("/step")
def step(action: Action):
    obs, reward, done, info = env.step(action)
    return {
        "observation": obs.model_dump() if obs else None,
        "reward": reward.model_dump(),
        "done": done,
        "info": info
    }

@app.get("/")
async def root():
    with RESULT_LOCK:
        body_text = RESULT_LOG
    escaped = html.escape(body_text)
    page = """
<html><head><title>OpenEnv Support Benchmark</title>
<meta charset='utf-8'></head><body>
<h1>OpenEnv Support Benchmark</h1>
<p>Container is running. Latest baseline output:</p>
<pre>{escaped}</pre>
<h2>FastAPI Endpoints Ready:</h2>
<ul>
<li>GET /health</li>
<li>POST /reset</li>
<li>POST /step</li>
</ul>
</body></html>
    """.format(escaped=escaped)
    return HTMLResponse(content=page)

if __name__ == "__main__":
    port = int(os.getenv("PORT", "7860"))
    uvicorn.run(app, host="0.0.0.0", port=port)

