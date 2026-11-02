FROM python:3.9-slim-bookworm
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates curl gnupg \
    && curl -fsSL https://packages.cloud.google.com/apt/doc/apt-key.gpg | gpg --dearmor -o /usr/share/keyrings/coral.gpg \
    && echo 'deb [signed-by=/usr/share/keyrings/coral.gpg] https://packages.cloud.google.com/apt coral-edgetpu-stable main' > /etc/apt/sources.list.d/coral.list \
    && apt-get update && apt-get install -y --no-install-recommends libedgetpu1-std \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml ./
COPY indoorcam ./indoorcam
RUN pip install . \
    && pip install 'https://github.com/google-coral/pycoral/releases/download/v2.0.0/tflite_runtime-2.5.0.post1-cp39-cp39-linux_x86_64.whl#sha256=132f8c184b99d69e40040d8067eb962d9435cd2579f5563c33b8673d257d353d' \
    && useradd --uid 10001 --create-home camera
COPY models ./models
COPY config ./config
USER camera
ENTRYPOINT ["python", "-m", "indoorcam.app"]
CMD ["--config", "/run/indoorcam/config.yaml"]
