"""Teacher-model chat API client for the online rephraser rollout.

Adapted from ``Inferapi/Chat_with_model_API_source1.py`` (``single_call_api`` +
its ThreadPoolExecutor batching) into a small, file-less, config-driven helper
that can be called from inside the training loop.

``batch_chat`` takes a list of chat ``messages`` arrays and returns a list of
``(ok, text)`` in the SAME order. It replaces the local ``generate_sequences``
call for summarize-replacement candidates: a strong external teacher produces
the rewrite, and the policy's tokenizer re-tokenizes its text downstream. The
teacher only returns text (no logprobs), so callers must not derive an
off-policy behavior logprob from it.

Endpoint is OpenAI-compatible ``/v1/chat/completions``. Config keys (all optional
except a url/api_list): url, api_list, api_key, model, threads, retry, timeout,
temperature, top_p, max_tokens, enable_thinking. The api key is read from cfg or
from env ``TEACHER_API_KEY`` / ``DASHSCOPE_API_KEY`` (never hard-code secrets).
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Tuple

import requests

# Default location of the standalone teacher-API config, resolved relative to this
# file so it works regardless of the caller's CWD / Hydra config. Override with the
# TEACHER_API_CONFIG env var.
_DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config", "teacher_api.yaml")


def load_teacher_config(path: str | None = None):
    """Load the standalone teacher-API config, kept fully separate from the
    trainer's Hydra config.

    Resolution order for the path: explicit ``path`` arg -> env
    ``TEACHER_API_CONFIG`` -> the bundled ``config/teacher_api.yaml``. Returns an
    OmegaConf DictConfig. Missing file -> a config with ``enable=False`` so the
    caller cleanly no-ops instead of crashing.
    """
    from omegaconf import OmegaConf

    cfg_path = path or os.environ.get("TEACHER_API_CONFIG") or _DEFAULT_CONFIG_PATH
    if not os.path.isfile(cfg_path):
        print(f"[teacher_api] config not found at {cfg_path}; treating as disabled.")
        return OmegaConf.create({"enable": False})
    cfg = OmegaConf.load(cfg_path)
    print(f"[teacher_api] loaded config from {cfg_path}")
    return cfg


def _cfg_get(cfg: Any, key: str, default=None):
    """Read a key from an OmegaConf DictConfig or a plain dict."""
    if cfg is None:
        return default
    getter = getattr(cfg, "get", None)
    if callable(getter):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _resolve_api_key(cfg: Any) -> str:
    return (
        _cfg_get(cfg, "api_key", "")
        or os.environ.get("TEACHER_API_KEY", "")
        or os.environ.get("DASHSCOPE_API_KEY", "")
        or ""
    )


def _resolve_endpoints(cfg: Any) -> List[str]:
    url = _cfg_get(cfg, "url", "") or _cfg_get(cfg, "api_url", "") or ""
    if url:
        return [str(url)]
    api_list = _cfg_get(cfg, "api_list", []) or []
    endpoints = [str(u) for u in list(api_list) if u]
    if not endpoints:
        raise ValueError(
            "teacher_api: no endpoint configured (set teacher_api.url or teacher_api.api_list)"
        )
    return endpoints


def _build_payload(messages: List[dict], cfg: Any) -> dict:
    payload = {
        "model": _cfg_get(cfg, "model", "Qwen3-235B-A22B-Instruct-2507"),
        "messages": messages,
        "stream": False,
        "temperature": float(_cfg_get(cfg, "temperature", 0.7)),
        "top_p": float(_cfg_get(cfg, "top_p", 0.95)),
        "max_tokens": int(_cfg_get(cfg, "max_tokens", 16384)),
    }
    if not bool(_cfg_get(cfg, "enable_thinking", False)):
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    return payload


def _single_chat(messages: List[dict], cfg: Any, api_url: str, headers: dict) -> Tuple[bool, str]:
    retry = int(_cfg_get(cfg, "retry", 3))
    timeout = int(_cfg_get(cfg, "timeout", 1500))
    payload = _build_payload(messages, cfg)
    for attempt in range(retry + 1):
        try:
            resp = requests.post(api_url, headers=headers, json=payload, timeout=timeout)
            resp.raise_for_status()
            result = resp.json()
            choices = result.get("choices") if isinstance(result, dict) else None
            if choices:
                return True, choices[0]["message"]["content"]
            return False, "api response missing 'choices'"
        except Exception as e:  # noqa: BLE001 - report any failure back to the caller
            if attempt == retry:
                return False, f"api error: {e}"
            time.sleep((2 ** attempt) * 0.1)  # exponential backoff
    return False, "unknown error"


def batch_chat(messages_list: List[List[dict]], cfg: Any) -> List[Tuple[bool, str]]:
    """Call the teacher chat API once per messages array, concurrently.

    Returns a list of ``(ok, text)`` aligned to ``messages_list``. Empty input
    returns ``[]``. Never raises on per-item failure — a failed item comes back
    as ``(False, reason)`` so the caller can drop/score it.
    """
    if not messages_list:
        return []
    endpoints = _resolve_endpoints(cfg)
    headers = {"Content-Type": "application/json"}
    api_key = _resolve_api_key(cfg)
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    threads = int(_cfg_get(cfg, "threads", 100))
    n = len(messages_list)
    results: List[Tuple[bool, str]] = [(False, "not run")] * n

    def _task(i: int) -> Tuple[int, Tuple[bool, str]]:
        url = endpoints[i % len(endpoints)]
        return i, _single_chat(messages_list[i], cfg, url, headers)

    with ThreadPoolExecutor(max_workers=max(1, min(threads, n))) as ex:
        for i, res in ex.map(_task, range(n)):
            results[i] = res
    return results
