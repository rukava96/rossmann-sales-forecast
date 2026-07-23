FROM python:3.10-slim

WORKDIR /app

# Install build tools for some packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies from pyproject.toml
COPY pyproject.toml .
RUN pip install --no-cache-dir ".[dev]"

COPY . .

ENV PYTHONPATH=/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]