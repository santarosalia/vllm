"""
vLLM Proxy - FastAPI 서버
모든 요청을 vLLM 백엔드로 패스스루하고, 요청 프롬프트/토큰수/응답 프롬프트를 로그 출력합니다.
"""

import json
import logging
import os
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
            if content is None:
                # content 없음 (tool_calls 등) -> role만이라도 표시
                extra = ""
                if "tool_calls" in m and m["tool_calls"]:
                    extra = " (tool_calls)"
                parts.append(f"[{role}] (no content){extra}")
            elif isinstance(content, str):
                parts.append(f"[{role}] {content}")
            elif isinstance(content, list):
                # multimodal: [{ "type": "text", "text": "..." }, ...]
                texts = []
                for c in content:
                    if isinstance(c, dict):
                        texts.append(c.get("text") or c.get("input") or str(c))
                    else:
                        texts.append(str(c))
                parts.append(f"[{role}] " + " ".join(t for t in texts if t))
                if not texts or not any(t for t in texts):
                    parts[-1] = f"[{role}] (empty parts)"
            else:
                parts.append(f"[{role}] {json.dumps(content)[:500]}")
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
        logger.info("=== REQUEST PROMPT ===\n%s", req_prompt)

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
                            line = chunk.decode("utf-8", errors="replace")
                            if line.strip().startswith("data: ") and "[DONE]" not in line:
                                payload = line.split("data: ", 1)[1].strip()
                                data = json.loads(payload)
                                for choice in data.get("choices") or []:
                                    delta = choice.get("delta") or {}
                                    if "content" in delta and delta["content"]:
                                        full_content.append(delta["content"])
                        except Exception:
                            pass
            if full_content:
                response_text = "".join(full_content)
                logger.info("=== RESPONSE PROMPT (stream) ===\n%s",
                            response_text)

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
            logger.info("=== RESPONSE PROMPT ===\n%s", response_text)
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
