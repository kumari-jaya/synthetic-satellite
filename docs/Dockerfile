# Use Python slim image
FROM python:3.9-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    VORTX_DOCS_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy documentation files
COPY mkdocs.yml .
COPY docs/ docs/

# Build documentation
RUN mkdocs build

# Expose port
EXPOSE 8080

# Set entrypoint
ENTRYPOINT ["mkdocs"]

# Default command
CMD ["serve", "-a", "0.0.0.0:8080"] 