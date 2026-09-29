"""Provider contract tests with synthetic data and in-memory HTTP responses only."""

import asyncio
import json
import sys
from pathlib import Path

import httpx
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import earlydx as dx
import llm_backend as llm
from pipeline.eval_unified import load_system
from pipeline.infer_api import run


@pytest.fixture
def configured(tmp_path, monkeypatch):
    cfg = {
        "gpt-5.5": {
            "provider": "azure",
            "endpoint": "https://synthetic.openai.azure.com",
            "deployment": "synthetic-deployment",
            "model": "gpt-5.5",
            "revision": "test-only",
        },
        "claude-opus-4.8": {
            "provider": "anthropic",
            "model": "claude-opus-4-8",
            "revision": "test-only",
        },
    }
    path = tmp_path / "backends.json"
    path.write_text(json.dumps(cfg))
    monkeypatch.setattr(llm, "CONFIG_PATH", str(path))
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "synthetic-azure-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "synthetic-anthropic-key")
    monkeypatch.setenv("EARLYDX_AZURE_REVIEW_OPTOUT", "1")
    monkeypatch.setenv("EARLYDX_ANTHROPIC_ZDR", "1")
    return cfg


def response(provider, text="<answer>Condition A</answer>"):
    if provider == "azure":
        return httpx.Response(
            200,
            json={
                "id": "test-response",
                "model": "gpt-5.5-test",
                "choices": [{"message": {"content": text}, "finish_reason": "stop"}],
                "usage": {"total_tokens": 9},
            },
            headers={"apim-request-id": "test-request"},
        )
    return httpx.Response(
        200,
        json={
            "id": "test-response",
            "model": "claude-opus-4-8",
            "content": [
                {"type": "thinking", "thinking": "hidden"},
                {"type": "text", "text": text},
            ],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 4, "output_tokens": 5},
        },
        headers={"request-id": "test-request"},
    )


@pytest.mark.parametrize(
    "role,provider", [("gpt-5.5", "azure"), ("claude-opus-4.8", "anthropic")]
)
def test_direct_provider_wire_format(configured, role, provider):
    async def exercise():
        def handler(request):
            body = json.loads(request.content)
            assert body["messages"] == [{"role": "user", "content": "synthetic prompt"}]
            assert "temperature" not in body
            if provider == "azure":
                assert (
                    str(request.url)
                    == "https://synthetic.openai.azure.com/openai/v1/chat/completions"
                )
                assert request.headers["api-key"] == "synthetic-azure-key"
                assert body["model"] == "synthetic-deployment"
                assert body["max_completion_tokens"] == 512 and body["store"] is False
                assert "max_tokens" not in body
            else:
                assert str(request.url) == "https://api.anthropic.com/v1/messages"
                assert request.headers["x-api-key"] == "synthetic-anthropic-key"
                assert request.headers["anthropic-version"] == "2023-06-01"
                assert body["max_tokens"] == 512
                assert "store" not in body
            return response(provider)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            reply = await llm.complete(
                role, "synthetic prompt", max_tokens=512, client=client
            )
            assert reply["text"] == "<answer>Condition A</answer>"
            assert reply["request_id"] == "test-request"
            assert reply["response_id"] == "test-response"
            assert reply["refused"] is False

    asyncio.run(exercise())
    assert "synthetic-azure-key" not in json.dumps(llm.provenance(role))
    assert "synthetic-anthropic-key" not in json.dumps(llm.provenance(role))


def test_account_attestations_and_routes(configured, monkeypatch):
    monkeypatch.delenv("EARLYDX_AZURE_REVIEW_OPTOUT")
    with pytest.raises(llm.ComplianceError, match="opt-out"):
        llm.check("gpt-5.5", configured["gpt-5.5"])
    monkeypatch.delenv("EARLYDX_ANTHROPIC_ZDR")
    with pytest.raises(llm.ComplianceError, match="ZDR"):
        llm.check("claude-opus-4.8", configured["claude-opus-4.8"])
    for endpoint in (
        "https://synthetic.openai.azure.com.evil.invalid",
        "http://synthetic.openai.azure.com",
        "https://synthetic.openai.azure.com/proxy",
        "https://user:secret@synthetic.openai.azure.com",
    ):
        with pytest.raises(llm.ComplianceError):
            llm.check("gpt-5.5", {**configured["gpt-5.5"], "endpoint": endpoint})
    with pytest.raises(llm.ComplianceError):
        llm.check(
            "claude-opus-4.8",
            {**configured["claude-opus-4.8"], "base_url": "https://router.invalid"},
        )
    with pytest.raises(llm.ComplianceError, match="local"):
        llm.check("judge", configured["gpt-5.5"])


def test_auth_failure_is_not_retried_or_logged(configured):
    attempts = []

    async def exercise():
        def handler(request):
            attempts.append(1)
            return httpx.Response(401, json={"error": "sensitive response body"})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(llm.LLMCallError) as err:
                await llm.complete(
                    "gpt-5.5",
                    "sensitive synthetic prompt",
                    max_tokens=12,
                    client=client,
                )
            assert "HTTP 401" in str(err.value)
            assert "sensitive" not in str(err.value)

    asyncio.run(exercise())
    assert len(attempts) == 1


def test_transient_retry_and_redirect_block(configured, monkeypatch):
    async def no_sleep(_):
        pass

    monkeypatch.setattr(llm.asyncio, "sleep", no_sleep)

    async def exercise():
        attempts = []

        def rate_limit(request):
            attempts.append(1)
            return httpx.Response(429) if len(attempts) == 1 else response("azure")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(rate_limit)
        ) as client:
            assert (
                await llm.complete("gpt-5.5", "synthetic", max_tokens=12, client=client)
            )["text"]
        assert len(attempts) == 2
        seen = []

        def redirect(request):
            seen.append(request.url.host)
            return httpx.Response(
                307, headers={"Location": "https://not-authorized.invalid"}
            )

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(redirect), follow_redirects=True
        ) as client:
            with pytest.raises(llm.LLMCallError):
                await llm.complete(
                    "gpt-5.5", "synthetic", max_tokens=12, retries=1, client=client
                )
        assert seen == ["synthetic.openai.azure.com"]

    asyncio.run(exercise())


def records():
    return [
        {
            "subject_id": i,
            "hadm_id": i + 10,
            "stay_id": i + 100,
            "split": "test",
            "evidence": {"window_hours": 0, "timestamp": "charttime"},
            "messages": [
                {"role": "user", "content": f"Synthetic encounter {i}" + dx.QUESTION},
                {
                    "role": "assistant",
                    "content": "<answer>GOLD MUST NOT BE SENT</answer>",
                },
            ],
        }
        for i in (1, 2)
    ]


def test_prediction_resume_no_gold_and_unified_eval(configured, tmp_path):
    test = tmp_path / "test.jsonl"
    out = tmp_path / "pred.jsonl"
    rows = records()
    dx.write_rows(test, rows)
    argv = ["--role", "gpt-5.5", "--test", str(test), "--out", str(out)]
    requests = []

    async def exercise():
        def handler(request):
            requests.append(json.loads(request.content))
            assert b"GOLD MUST NOT BE SENT" not in request.content
            return response("azure")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await run(argv, client=client)
            await run(argv, client=client)

    asyncio.run(exercise())
    assert len(requests) == 2
    generated = dx.read_rows(out)
    assert {r["stay_id"] for r in generated} == {101, 102}
    assert all(
        "gold" not in r and r["api"]["model"] == "gpt-5.5-test" for r in generated
    )
    predictions, _ = load_system(
        {"name": "synthetic", "files": ["pred.jsonl"]}, tmp_path, rows, 0, "charttime"
    )
    assert predictions == {101: ["Condition A"], 102: ["Condition A"]}
    # A transport-interrupted output must not be mistaken for an intentional subsample.
    dx.write_rows(out, generated[:1])
    with pytest.raises(ValueError, match="incomplete stage"):
        load_system(
            {"name": "synthetic", "files": ["pred.jsonl"]},
            tmp_path,
            rows,
            0,
            "charttime",
        )


def test_failed_request_resume_and_model_outcomes(configured, tmp_path):
    test = tmp_path / "test.jsonl"
    out = tmp_path / "pred.jsonl"
    dx.write_rows(test, records())
    argv = ["--role", "claude-opus-4.8", "--test", str(test), "--out", str(out)]
    seen = []

    async def exercise():
        def first(request):
            text = json.loads(request.content)["messages"][0]["content"]
            if "encounter 2" in text:
                return httpx.Response(401)
            return response("anthropic", "Malformed diagnosis output")

        async with httpx.AsyncClient(transport=httpx.MockTransport(first)) as client:
            with pytest.raises(RuntimeError, match="Incomplete API predictions"):
                await run(argv, client=client)
        saved = dx.read_rows(out)
        assert len(saved) == 1 and saved[0]["fmt"] is False and saved[0]["pred"] == []

        def second(request):
            text = json.loads(request.content)["messages"][0]["content"]
            seen.append(text)
            return httpx.Response(
                200,
                json={
                    "model": "claude-opus-4-8",
                    "content": [],
                    "stop_reason": "refusal",
                },
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(second)) as client:
            await run(argv, client=client)

    asyncio.run(exercise())
    assert len(seen) == 1 and "encounter 2" in seen[0]
    saved = dx.read_rows(out)
    assert (
        len(saved) == 2
        and saved[1]["api"]["refused"] is True
        and saved[1]["pred"] == []
    )


def test_dry_run_no_keys_no_requests(configured, tmp_path, monkeypatch):
    monkeypatch.delenv("AZURE_OPENAI_API_KEY")
    test = tmp_path / "test.jsonl"
    out = tmp_path / "pred.jsonl"
    dx.write_rows(test, records())

    async def exercise():
        def handler(_):
            pytest.fail("Dry run must not send a request")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await run(
                [
                    "--role",
                    "gpt-5.5",
                    "--test",
                    str(test),
                    "--out",
                    str(out),
                    "--dry-run",
                ],
                client=client,
            )

    asyncio.run(exercise())
    assert not out.exists() and not Path(str(out) + ".meta.json").exists()
