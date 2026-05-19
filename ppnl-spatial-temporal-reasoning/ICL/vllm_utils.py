"""
Utilities for optionally launching a vLLM OpenAI-compatible server as a
subprocess from within an inference script, and for post-processing model
responses.

Typical usage in an inference script::

    from vllm_utils import launch_vllm_server, strip_thinking, vllm_extra_body

    if args.launch_vllm:
        launch_vllm_server(args.model, args.base_url, args.tensor_parallel_size)

    # In call_api():
    extra_body = vllm_extra_body(model)   # disables thinking for Qwen models
    response = client.chat.completions.create(..., extra_body=extra_body)
    text = strip_thinking(response.choices[0].message.content)
"""

import atexit
import re
import subprocess
import time
import urllib.request
from urllib.parse import urlparse


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks from a model response."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def vllm_extra_body(model: str) -> dict | None:
    """
    Return an extra_body dict to pass to the vLLM API for the given model,
    or None if no extra configuration is needed.

    Currently disables the built-in chain-of-thought for Qwen3 and DeepSeek
    reasoning models, which otherwise consume most of the token budget on
    reasoning before answering.
    """
    m = model.lower()
    if "qwen" in m or "deepseek" in m:
        return {"chat_template_kwargs": {"enable_thinking": False}}
    return None


def _parse_base_url(base_url: str):
    """Return (host, port) parsed from a URL like http://localhost:8000/v1."""
    p = urlparse(base_url)
    return p.hostname or "localhost", p.port or 8000


def _wait_for_server(base_url: str, timeout: int = 300):
    """
    Poll until the vLLM server responds or *timeout* seconds have elapsed.
    Checks both /health and /v1/models endpoints (different vLLM versions
    expose different readiness endpoints).
    """
    root = base_url.rstrip("/")
    # Strip trailing /v1 if present to get the server root.
    if root.endswith("/v1"):
        root = root[:-3]
    check_urls = [f"{root}/health", f"{root}/v1/models"]

    deadline = time.time() + timeout
    while time.time() < deadline:
        for url in check_urls:
            try:
                urllib.request.urlopen(url, timeout=2)
                return
            except Exception:
                pass
        time.sleep(5)

    raise RuntimeError(
        f"vLLM server at {base_url} did not become ready within {timeout}s. "
        "Check the server logs for errors."
    )


def launch_vllm_server(model: str, base_url: str, tensor_parallel_size: int = 1):
    """
    Spawn a vLLM OpenAI-compatible server and wait until it is ready.

    The process is registered with *atexit* so it is terminated when the
    calling script exits (normally or via exception).

    Args:
        model:                HuggingFace model name (e.g. deepseek-ai/deepseek-moe-16b-chat).
        base_url:             Full base URL including /v1, e.g. http://localhost:8000/v1.
        tensor_parallel_size: Number of GPUs to use for tensor parallelism.
    """
    host, port = _parse_base_url(base_url)
    cmd = [
        "vllm", "serve", model,
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--host", host,
        "--port", str(port),
    ]
    print(f"[vllm_utils] Launching vLLM server: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd)
    atexit.register(proc.terminate)
    print(f"[vllm_utils] Waiting for server at {base_url} ...")
    _wait_for_server(base_url)
    print("[vllm_utils] vLLM server is ready.")
    return proc
