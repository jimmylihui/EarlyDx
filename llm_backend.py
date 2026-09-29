"""Local corpus-processing models and direct Azure/Anthropic prediction APIs.

Verifier, teacher and judge stay local (§R). Hosted baselines require the operator to confirm
account-level data handling arrangements; environment flags do not grant those arrangements.
"""

import asyncio
import hashlib
import json
import os
from pathlib import Path
from urllib.parse import urlparse

import httpx

CONFIG_PATH = os.environ.get(
    "EARLYDX_BACKENDS", str(Path(__file__).with_name("backends.json"))
)
EXPECTED = {"verifier": "minimaxm3", "teacher": "mimov25", "judge": "qwen3527b"}
HOSTED = {"gpt-5.5": "azure", "claude-opus-4.8": "anthropic"}
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


class ComplianceError(RuntimeError):
    pass


class LLMCallError(RuntimeError):
    pass


def required(cfg, fields, role):
    for field in fields:
        value = cfg.get(field, "")
        if (
            not isinstance(value, str)
            or not value.strip()
            or "<" in value
            or "TO FILL" in value
        ):
            raise ValueError(f"{role}: configure the actual {field} before running")


def check(role, cfg):
    provider = cfg.get("provider")
    if role in EXPECTED:
        if provider != "local":
            raise ComplianceError(
                f"{role}: the corpus-processing stages require a local backend"
            )
        url = urlparse(cfg.get("base_url", ""))
        hosts = {"localhost", "127.0.0.1", "::1"} | {
            x.strip()
            for x in os.environ.get("EARLYDX_LOCAL_HOSTS", "").split(",")
            if x.strip()
        }
        if (
            url.scheme not in {"http", "https"}
            or url.hostname not in hosts
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ComplianceError(
                "Use loopback or an institutional host listed in EARLYDX_LOCAL_HOSTS"
            )
        required(cfg, ("model", "checkpoint", "revision"), role)
        model = "".join(c for c in cfg["model"].lower() if c.isalnum())
        if EXPECTED[role] not in model:
            raise ValueError(
                f"{role}: configured model does not match the paper role (see MODELS.md)"
            )
    elif role in HOSTED:
        if provider != HOSTED[role]:
            raise ComplianceError(
                f"{role}: requires the direct {HOSTED[role]} provider"
            )
        required(cfg, ("model", "revision"), role)
        if provider == "azure":
            if cfg["model"] != "gpt-5.5":
                raise ValueError("gpt-5.5 role must record the GPT-5.5 model")
            required(cfg, ("endpoint", "deployment"), role)
            url = urlparse(cfg["endpoint"])
            if (
                url.scheme != "https"
                or not (url.hostname or "").endswith(".openai.azure.com")
                or url.username
                or url.password
                or url.path not in ("", "/")
                or url.query
                or url.fragment
                or url.port not in (None, 443)
            ):
                raise ComplianceError(
                    "Azure endpoint must be https://RESOURCE.openai.azure.com with no path/query"
                )
            if os.environ.get("EARLYDX_AZURE_REVIEW_OPTOUT") != "1":
                raise ComplianceError(
                    "Confirm approved Azure human-review opt-out with EARLYDX_AZURE_REVIEW_OPTOUT=1"
                )
        else:
            if not cfg["model"].startswith("claude-opus-4-8"):
                raise ValueError(
                    "claude-opus-4.8 role must use a Claude Opus 4.8 model ID"
                )
            if (
                cfg.get("base_url", "https://api.anthropic.com").rstrip("/")
                != "https://api.anthropic.com"
            ):
                raise ComplianceError(
                    "Anthropic requests must go directly to api.anthropic.com"
                )
            if os.environ.get("EARLYDX_ANTHROPIC_ZDR") != "1":
                raise ComplianceError(
                    "Confirm applicable Anthropic ZDR with EARLYDX_ANTHROPIC_ZDR=1"
                )
    else:
        raise ValueError(f"Unknown backend role: {role}")


def config(role):
    with open(CONFIG_PATH) as f:
        cfg = json.load(f)
    if role not in cfg:
        raise ValueError(f"No backend configured for {role}")
    check(role, cfg[role])
    return cfg[role]


def provenance(role):
    c = config(role)
    fields = (
        "provider",
        "model",
        "checkpoint",
        "revision",
        "base_url",
        "endpoint",
        "deployment",
        "anthropic_version",
        "workspace_id",
    )
    out = {k: c[k] for k in fields if k in c}
    if c["provider"] == "azure":
        out.update(
            api="chat-completions-v1", store=False, human_review_optout_confirmed=True
        )
    elif c["provider"] == "anthropic":
        out.update(
            api="messages",
            anthropic_version=c.get("anthropic_version", "2023-06-01"),
            base_url="https://api.anthropic.com",
            zdr_confirmed=True,
        )
    return out


def describe(role):
    c = provenance(role)
    return f"{role}: {c['provider']} / {c['model']} @ {c['revision']}"


def cache_path(role, base):
    tag = hashlib.sha256(
        json.dumps(provenance(role), sort_keys=True).encode()
    ).hexdigest()[:16]
    stem, ext = os.path.splitext(base)
    return f"{stem}.{tag}{ext}"


def _request(cfg, prompt, max_tokens, temperature):
    provider = cfg["provider"]
    messages = [{"role": "user", "content": prompt}]
    if provider == "azure":
        key_name = cfg.get("api_key_env", "AZURE_OPENAI_API_KEY")
        url = cfg["endpoint"].rstrip("/") + "/openai/v1/chat/completions"
        body = {
            "model": cfg["deployment"],
            "messages": messages,
            "max_completion_tokens": max_tokens,
            "store": False,
        }
        headers = {"api-key": os.environ.get(key_name, "")}
    elif provider == "anthropic":
        key_name = cfg.get("api_key_env", "ANTHROPIC_API_KEY")
        url = ANTHROPIC_URL
        body = {"model": cfg["model"], "messages": messages, "max_tokens": max_tokens}
        headers = {
            "x-api-key": os.environ.get(key_name, ""),
            "anthropic-version": cfg.get("anthropic_version", "2023-06-01"),
        }
        if cfg.get("workspace_id"):
            headers["anthropic-workspace-id"] = cfg["workspace_id"]
    else:
        key_name = cfg.get("api_key_env")
        url = cfg["base_url"].rstrip("/") + "/chat/completions"
        body = {
            "model": cfg["model"],
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        headers = (
            {"Authorization": f"Bearer {os.environ.get(key_name, '')}"}
            if key_name
            else {}
        )
    if key_name and not os.environ.get(key_name):
        raise ValueError(f"Missing API credential in environment variable {key_name}")
    return url, headers, body


def _parse(cfg, response):
    data = response.json()
    if cfg["provider"] == "anthropic":
        text = "".join(b["text"] for b in data["content"] if b.get("type") == "text")
        finish = data.get("stop_reason")
        refused = finish == "refusal"
    else:
        choice = data["choices"][0]
        text = choice["message"].get("content") or ""
        finish = choice.get("finish_reason")
        refused = bool(choice["message"].get("refusal")) or finish == "content_filter"
    if not isinstance(text, str) or (not text.strip() and not refused):
        raise ValueError("Empty model response")
    return {
        "text": text,
        "model": data.get("model"),
        "response_id": data.get("id"),
        "request_id": response.headers.get("request-id")
        or response.headers.get("apim-request-id")
        or response.headers.get("x-request-id"),
        "finish_reason": finish,
        "refused": refused,
        "usage": data.get("usage"),
    }


async def complete(
    role, prompt, *, max_tokens, temperature=0.0, retries=5, client=None
):
    """Return text and provider metadata. Retry transient failures, not auth/config errors."""
    if retries < 1 or max_tokens < 1:
        raise ValueError("retries and max_tokens must be positive")
    cfg = config(role)
    url, headers, body = _request(cfg, prompt, max_tokens, temperature)
    owned = client is None
    if owned:
        client = httpx.AsyncClient(trust_env=False, follow_redirects=False)
    last_status = None
    try:
        for attempt in range(retries):
            try:
                response = await client.post(
                    url,
                    headers=headers,
                    json=body,
                    timeout=cfg.get("timeout", 300),
                    follow_redirects=False,
                )
                last_status = response.status_code
                if (
                    response.status_code >= 400
                    and response.status_code not in (408, 409, 429)
                    and response.status_code < 500
                ):
                    raise LLMCallError(
                        f"{role}: HTTP {response.status_code}; check account, model and request configuration"
                    )
                response.raise_for_status()
                return _parse(cfg, response)
            except (httpx.HTTPError, ValueError, KeyError, IndexError, TypeError):
                if attempt + 1 < retries:
                    await asyncio.sleep(min(2**attempt, 30))
    finally:
        if owned:
            await client.aclose()
    raise LLMCallError(
        f"{role}: failed after {retries} attempts (HTTP status {last_status}); no prediction or judgment recorded"
    )


async def chat(role, prompt, **kwargs):
    """Text-only compatibility interface for verifier, teacher and judge callers."""
    result = await complete(role, prompt, **kwargs)
    return result["text"]
