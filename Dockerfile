FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nodejs \
    npm \
    git \
    ffmpeg \
    libsm6 \
    libxext6 \
    curl \
    build-essential \
    pkg-config \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Verify npm and npx installation
RUN npm -v && npx -v

# Install prettier and markdown plugins globally
RUN npm install -g prettier@3.4.2 @prettier/plugin-markdown

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install UV
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.cargo/bin:${PATH}"

# Copy application code
COPY . .

# Create data directory with proper permissions
RUN mkdir -p /data && chmod 755 /data

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV AIPROXY_TOKEN=${AIPROXY_TOKEN}
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]