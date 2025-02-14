FROM python:3.12-slim-bookworm

# Set working directory to /app
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Create a virtual environment
RUN python -m venv venv

# Activate the virtual environment
ENV PATH="/app/venv/bin:$PATH"

# Install dependencies
RUN pip install -r requirements.txt

# Copy application code
COPY app.py .

# Expose port
EXPOSE 8000

# Run command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]