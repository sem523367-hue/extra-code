# syntax=docker/dockerfile:1
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY pyproject.toml ./
COPY server ./server
COPY model ./model

# llama-cpp-python compiles llama.cpp; CPU build by default. For CUDA add
# --build-arg CMAKE_ARGS="-DGGML_CUDA=ON" and a CUDA base image.
ARG CMAKE_ARGS="-DGGML_NATIVE=OFF"
ENV CMAKE_ARGS=${CMAKE_ARGS}
RUN pip wheel --wheel-dir /wheels llama-cpp-python \
 && pip wheel --wheel-dir /wheels fastapi "uvicorn[standard]" pydantic httpx numpy tokenizers

FROM python:3.12-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 extracode
WORKDIR /app
COPY --from=builder /wheels /wheels
RUN pip install --no-index --find-links=/wheels /wheels/*.whl && rm -rf /wheels
COPY server ./server
COPY model ./model
RUN mkdir -p /app/server/data /models && chown -R extracode:extracode /app /models
USER extracode
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD curl -fsS http://127.0.0.1:8000/health || exit 1
CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
