# 1. Use Python 3.9 slim (works with Mediapipe prebuilt wheels)
FROM python:3.9-slim

# 2. Set a working directory
WORKDIR /app

# 3. Install system dependencies (needed for opencv, mediapipe, etc.)
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# 4. Upgrade pip to latest
RUN pip install --upgrade pip

# 5. Copy requirements first (better caching)
COPY backend/requirements.txt .

# 6. Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# 7. Copy the backend code into container
COPY backend/ .

# 8. Expose port for FastAPI
EXPOSE 8000

# 9. Command to run backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
