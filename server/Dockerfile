# ---------- base ----------
FROM pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# ffmpeg는 STT 변환에 필수
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg build-essential \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 복사/설치(캐시 효율↑)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 앱 코드
COPY . .

# 기본 환경변수(필요시 compose/.env에서 덮어쓰기)
ENV HOST=0.0.0.0 \
    PORT=8000 \
    MODEL_PATH=/app/model_bs32 \
    INDEX_PATH=/app/store \
    RAW_DATA_ROOT=/app/data \
    USE_CPU_WHISPER=0 \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch

# 임시 디렉토리
RUN mkdir -p /app/temp /app/.cache/huggingface /app/.cache/torch

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "info"]
