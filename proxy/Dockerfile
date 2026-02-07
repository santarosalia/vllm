# vLLM Proxy - FastAPI 패스스루 + 로깅
FROM python:3.11-slim

WORKDIR /app

# uv 설치 및 의존성 설치 (빌드 시 uv lock 생성)
COPY pyproject.toml ./
RUN pip install uv \
    && uv lock \
    && uv sync --frozen --no-dev --no-install-project

COPY app/ ./app/

ENV VLLM_BASE_URL=http://localhost:8000
ENV PROXY_TIMEOUT=300

EXPOSE 8080

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]
