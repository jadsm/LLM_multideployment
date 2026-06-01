FROM python:3.11-slim

WORKDIR /app

# Install build dependencies for math libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .


# Install torch specifically for CPU to avoid CUDA dependency weight
#RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
#RUN pip install --no-cache-dir transformers safetensors accelerate fastapi uvicorn pydantic
RUN pip install --no-cache-dir -r requirements.txt


# Explicitly copy files
COPY ./model /app/model/
#COPY /model/* /app/model/
COPY inference.py /app/inference.py

# Force CPU usage via environment variable
ENV CUDA_VISIBLE_DEVICES=""

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]