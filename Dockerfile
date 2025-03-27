FROM python:3.11.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libmagic1 \
    libmagic-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements/requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copy the rest of the application
COPY . .

# Create necessary directories
RUN mkdir -p uploads storage

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

# Expose the ports the apps run on
EXPOSE 8000 8501

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Create a script to run either FastAPI or Streamlit
RUN echo '#!/bin/bash\n\
if [ "$SERVICE_TYPE" = "streamlit" ]; then\n\
    streamlit run src/channel/streamlit/app.py --server.port 8501 --server.address 0.0.0.0\n\
else\n\
    python src/channel/fastapi/main.py\n\
fi' > /app/start.sh && chmod +x /app/start.sh

# Command to run the application
CMD ["/app/start.sh"] 