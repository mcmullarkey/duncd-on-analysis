FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .

# Create templates directory and copy template
RUN mkdir -p templates
COPY templates/index.html templates/

# Environment variable for RSS feed URL (to be provided at runtime)
ENV RSS_FEED_URL=""

# Explicitly expose port 5000
EXPOSE 8080

# Use environment variables to configure Flask
ENV FLASK_APP=app.py
ENV FLASK_ENV=development
ENV PYTHONUNBUFFERED=1

CMD ["python", "app.py"]