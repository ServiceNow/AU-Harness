FROM pytorch/pytorch:2.3.0-cuda11.8-cudnn8-devel

RUN mkdir -p /app /cache && chmod 777 /app /cache

WORKDIR /app

RUN groupadd -r appuser && useradd -r -g appuser -m appuser
RUN chown -R appuser:appuser /app /cache

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    MODEL_PATH="google/gemma-3n-E4B-it" \
    HF_TOKEN="" \
    PORT=8000 \
    PYTHONPATH="/app" \
    HF_HOME="tmp/cache/huggingface"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    git \
    curl \
    ninja-build \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# Copy project files
COPY requirements.txt /app/
COPY app.py models.py utils.py infer.py /app/

RUN chown -R appuser:appuser /app

RUN pip install --no-cache-dir -r requirements.txt

RUN pip install --no-cache-dir deepspeed==0.15.4

RUN pip install --no-cache-dir flash-attn==2.7.3

WORKDIR /app

EXPOSE 8000

COPY --chown=appuser:appuser start.sh /app/start.sh
RUN chmod +x /app/start.sh

USER appuser

CMD ["/app/start.sh"]
