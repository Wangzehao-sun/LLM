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
from concurrent.futures import ThreadPoolExecutor, as_completed
from concurrent.futures import TimeoutError as FuturesTimeoutError
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

    MASTER SWITCH: the env var ``TEACHER_API_ENABLE`` (set from the training
    script) OVERRIDES the file's ``enable`` when present (1/true/yes/on ->
    enabled, else disabled). This lets the shell script toggle teacher on/off
    while url/model/... stay in the standalone yaml.
    """
    from omegaconf import OmegaConf

    cfg_path = path or os.environ.get("TEACHER_API_CONFIG") or _DEFAULT_CONFIG_PATH
    if os.path.isfile(cfg_path):
        cfg = OmegaConf.load(cfg_path)
        print(f"[teacher_api] loaded config from {cfg_path}")
    else:
        print(f"[teacher_api] config not found at {cfg_path}; enable defaults to False.")
        cfg = OmegaConf.create({"enable": False})

    env_enable = os.environ.get("TEACHER_API_ENABLE")
    if env_enable is not None:
        cfg.enable = str(env_enable).strip().lower() in ("1", "true", "yes", "on", "y", "t")
        print(f"[teacher_api] enable overridden by TEACHER_API_ENABLE -> {bool(cfg.enable)}")
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

    OBSERVABILITY: results are consumed as they complete (not in submit order), so
    progress is reported while requests are still in flight. Every ``log_every``
    completions (and at least every ``log_interval_s`` seconds) one line is printed
    with completed/total, ok/failed counts and elapsed time. This makes a slow-but-
    working teacher distinguishable from a hung one — previously the call was fully
    silent until the last request returned, so "server is queueing" and "server is
    dead" looked identical.

    BOUNDED WAIT: ``wall_clock_budget`` (seconds, 0/None = unlimited) caps the total
    time spent here. When it expires, in-flight requests are abandoned and their rows
    come back as ``(False, 'wall_clock_budget exceeded')``. Callers already treat a
    failed row as "no candidate" (-> that question simply isn't replaced), so giving
    up is safe and keeps training moving instead of blocking the step forever.
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

    budget = float(_cfg_get(cfg, "wall_clock_budget", 0) or 0)
    log_every = max(1, int(_cfg_get(cfg, "log_every", 0) or max(1, n // 10)))
    log_interval_s = float(_cfg_get(cfg, "log_interval_s", 30) or 0)
    workers = max(1, min(threads, n))
    print(
        f"[teacher_api] dispatching {n} request(s), {workers} concurrent, "
        f"timeout={_cfg_get(cfg, 'timeout', 1500)}s retry={_cfg_get(cfg, 'retry', 3)}"
        + (f", wall_clock_budget={budget:g}s" if budget > 0 else "")
    )

    def _task(i: int) -> Tuple[int, Tuple[bool, str]]:
        url = endpoints[i % len(endpoints)]
        return i, _single_chat(messages_list[i], cfg, url, headers)

    t0 = time.time()
    done = n_ok = 0
    last_log = t0
    timed_out = False
    ex = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [ex.submit(_task, i) for i in range(n)]
        try:
            for fut in as_completed(futures, timeout=budget if budget > 0 else None):
                i, res = fut.result()
                results[i] = res
                done += 1
                n_ok += bool(res[0])
                now = time.time()
                if done % log_every == 0 or done == n or (
                    log_interval_s > 0 and now - last_log >= log_interval_s
                ):
                    last_log = now
                    print(
                        f"[teacher_api]   {done}/{n} done "
                        f"(ok={n_ok}, failed={done - n_ok}) {now - t0:.0f}s elapsed",
                        flush=True,
                    )
        except FuturesTimeoutError:
            timed_out = True
            for fut in futures:
                fut.cancel()
            for i in range(n):
                if results[i][1] == "not run":
                    results[i] = (False, "wall_clock_budget exceeded")
    finally:
        # Don't block on abandoned in-flight requests when the budget blew.
        ex.shutdown(wait=not timed_out, cancel_futures=timed_out)

    elapsed = time.time() - t0
    if timed_out:
        print(
            f"[teacher_api] WALL-CLOCK BUDGET ({budget:g}s) EXCEEDED: only {done}/{n} "
            f"completed (ok={n_ok}); abandoning the rest. Those questions keep their "
            f"original rollouts. Raise teacher_api.wall_clock_budget or lower max_tokens/"
            f"threads if this repeats.",
            flush=True,
        )
    else:
        print(
            f"[teacher_api] all {n} request(s) returned in {elapsed:.0f}s "
            f"(ok={n_ok}, failed={n - n_ok})",
            flush=True,
        )
    return results
