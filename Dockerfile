FROM python:3.11-slim

# ffmpeg for audio probing/conversion; build tools for ctranslate2 wheels fallback
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY app.py .

# runtime dirs (never baked with data)
RUN mkdir -p data reports

EXPOSE 7860

# Whisper weights download on first startup (tiny model); override with
# CC_WHISPER_MODEL / CC_PRELOAD_MODEL. All secrets via environment.
CMD ["python", "app.py"]
