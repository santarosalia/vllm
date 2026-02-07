"""
vLLM Proxy - FastAPI 서버
모든 요청을 vLLM 백엔드로 패스스루하고, 요청 프롬프트/토큰수/응답 프롬프트를 로그 출력합니다.
"""

import json
import logging
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

logger = logging.getLogger("vllm-proxy")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

app = FastAPI(title="vLLM Proxy", version="1.0.0")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:5678").rstrip("/")
PROXY_TIMEOUT = float(os.getenv("PROXY_TIMEOUT", "300"))


def extract_prompt_from_body(body: dict[str, Any]) -> str:
    """요청 바디에서 프롬프트(메시지 또는 prompt) 추출."""
    if "messages" in body:
        parts = []
        for m in body["messages"]:
            role = m.get("role", "unknown")
            content = m.get("content")
            if isinstance(content, str):
                parts.append(f"[{role}] {content}")
            elif isinstance(content, list):
                # multimodal
                texts = [c.get("text", str(c)) for c in content if isinstance(c, dict)]
                parts.append(f"[{role}] " + " ".join(texts))
        return "\n".join(parts) if parts else json.dumps(body["messages"])
    if "prompt" in body:
        p = body["prompt"]
        return p if isinstance(p, str) else json.dumps(p)
    return "(prompt not found)"


def extract_usage_from_response(data: dict[str, Any]) -> dict[str, int]:
    """응답에서 usage(토큰 수) 추출."""
    usage = data.get("usage") or {}
    return {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "total_tokens": usage.get("total_tokens", 0),
    }


def extract_response_text(data: dict[str, Any]) -> str:
    """응답에서 생성된 텍스트 추출."""
    choices = data.get("choices") or []
    if not choices:
        return "(no choices)"
    choice = choices[0]
    if "message" in choice:
        msg = choice["message"]
        content = msg.get("content")
        return content if isinstance(content, str) else json.dumps(content)
    if "text" in choice:
        return choice["text"]
    return "(no content)"


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS"])
async def proxy(path: str, request: Request) -> Response:
    """모든 요청을 vLLM 백엔드로 패스스루."""
    url = f"{VLLM_BASE_URL}/{path}"
    method = request.method

    # 요청 바디 (POST 등)
    body = None
    if method in ("POST", "PUT", "PATCH") and path.strip():
        try:
            body = await request.json()
        except Exception:
            body = None

    # 로깅용: 프롬프트 추출 (chat/completions, completions 등)
    if body and ("messages" in body or "prompt" in body):
        req_prompt = extract_prompt_from_body(body)
        logger.info("=== REQUEST PROMPT ===\n%s", req_prompt[:2000] + ("..." if len(req_prompt) > 2000 else ""))

    headers = dict(request.headers)
    # 호스트 제거해서 백엔드가 자신의 호스트로 받도록
    headers.pop("host", None)

    raw_body = await request.body() if method in ("POST", "PUT", "PATCH") else None

    if request.headers.get("accept") == "text/event-stream" or (body and body.get("stream")):
        # 스트리밍: 클라이언트를 제너레이터 안에서 생성해 스트림 수명과 맞춤
        full_content: list[str] = []

        async def stream():
            nonlocal full_content
            async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
                async with client.stream(method, url, content=raw_body, headers=headers) as r:
                    async for chunk in r.aiter_bytes():
                        yield chunk
                        try:
                            line = chunk.decode(errors="ignore")
                            if "data: " in line and "[DONE]" not in line:
                                m = re.search(r'"content":"([^"]*)"', line)
                                if m:
                                    full_content.append(m.group(1).encode().decode("unicode_escape"))
                        except Exception:
                            pass
            if full_content:
                response_text = "".join(full_content)
                logger.info("=== RESPONSE PROMPT (stream) ===\n%s",
                            response_text[:2000] + ("..." if len(response_text) > 2000 else ""))

        return StreamingResponse(
            stream(),
            status_code=200,
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
        )

    # 비스트리밍
    async with httpx.AsyncClient(timeout=PROXY_TIMEOUT) as client:
        resp = await client.request(method, url, content=raw_body, headers=headers)

    # 비스트리밍 응답 로깅
    if resp.status_code == 200 and resp.content:
        try:
            data = resp.json()
            usage = extract_usage_from_response(data)
            logger.info("=== TOKEN USAGE === prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                        usage["prompt_tokens"], usage["completion_tokens"], usage["total_tokens"])
            response_text = extract_response_text(data)
            logger.info("=== RESPONSE PROMPT ===\n%s", response_text[:2000] + ("..." if len(response_text) > 2000 else ""))
        except Exception as e:
            logger.debug("Response log parse skip: %s", e)

    return Response(
        content=resp.content,
        status_code=resp.status_code,
        headers=dict(resp.headers),
        media_type=resp.headers.get("content-type"),
    )


@app.get("/health")
async def health():
    """프록시 서버 헬스체크."""
    return {"status": "ok", "vllm_base_url": VLLM_BASE_URL}
