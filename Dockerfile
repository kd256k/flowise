# =============================================================================
# Flowise - 배수지 수요 예측 LSTM 서비스
# Base: NVIDIA CUDA 12.6 + cuDNN 9 (Ubuntu 22.04)
# Python 3.14 (deadsnakes PPA)
# =============================================================================

# ── Stage 1: Base ──────────────────────────────────────────────────────────
FROM nvidia/cuda:12.6.1-cudnn-runtime-ubuntu22.04 AS base

# 비대화형 모드 (tzdata 등 프롬프트 방지)
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    curl \
    git \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.14 \
    python3.14-venv \
    python3.14-dev \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.14 \
    && ln -sf /usr/bin/python3.14 /usr/bin/python \
    && ln -sf /usr/bin/python3.14 /usr/bin/python3 \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# pip 기본 설정
RUN python -m pip install --no-cache-dir --upgrade pip setuptools wheel

# ── Stage 2: Dependencies ──────────────────────────────────────────────────
FROM base AS deps

WORKDIR /app

# requirements 먼저 복사 → 레이어 캐싱 활용
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

# ── Stage 3: Final ─────────────────────────────────────────────────────────
FROM deps AS final

WORKDIR /app

# 소스코드 복사 (개발 시 volume mount로 오버라이드 가능)
COPY src/ ./src/

# 환경변수 기본값 (docker-compose에서 오버라이드)
ENV REDIS_HOST=10.125.121.184
ENV REDIS_PORT=6379
ENV MYSQL_HOST=10.125.121.184
ENV MYSQL_PORT=3306
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# 헬스체크
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/docs')" || exit 1

# 기본 실행 (docker-compose에서 command로 오버라이드)
CMD ["python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
