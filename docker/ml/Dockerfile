# Stage 1: Build dependencies
FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04 as builder

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VORTX_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    python3.9-dev \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /build

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies with ML extras
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Copy the package
COPY . .

# Install the package with ML extras
RUN pip3 install --no-cache-dir -e .[ml]

# Stage 2: Runtime
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    VORTX_ENV=production

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    python3.9 \
    python3-pip \
    gdal-bin \
    libgdal-dev \
    proj-bin \
    libproj-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
COPY --from=builder /build /app

# Create model directory
RUN mkdir -p /app/models

# Create non-root user
RUN useradd -m -u 1000 vortx && \
    chown -R vortx:vortx /app

# Switch to non-root user
USER vortx

# Expose ports
EXPOSE 8001

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Set entrypoint
ENTRYPOINT ["vo"]

# Default command
CMD ["serve-ml", "--host", "0.0.0.0", "--port", "8001"] 