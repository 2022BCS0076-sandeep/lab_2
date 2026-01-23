FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Copy requirements first (for better caching)
COPY requirements.txt .

# Install system and Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy entire project
COPY . .

# Expose port for FastAPI
EXPOSE 8000

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
