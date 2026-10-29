FROM python:3.11-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Run as non-root user (standard container best practice)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    TF_ENABLE_ONEDNN_OPTS=0 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR $HOME/app

# Install Python requirements
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy all project code and models
COPY --chown=user . $HOME/app

# Expose port (Render automatically provides $PORT, defaults to 7860/5000)
EXPOSE 7860
ENV PORT=7860
ENV HOST=0.0.0.0

CMD ["python", "app/backend/app.py"]
