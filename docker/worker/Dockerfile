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

# Install dependencies
RUN pip3 install --no-cache-dir -r requirements.txt && \
    pip3 install --no-cache-dir "celery[redis]>=5.3.0"

# Copy the package
COPY . .

# Install the package
RUN pip3 install --no-cache-dir -e .

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

# Create data directory
RUN mkdir -p /app/data

# Create non-root user
RUN useradd -m -u 1000 vortx && \
    chown -R vortx:vortx /app

# Switch to non-root user
USER vortx

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD celery -A vortx.worker inspect ping -d celery@$HOSTNAME || exit 1

# Set entrypoint
ENTRYPOINT ["celery"]

# Default command
CMD ["-A", "vortx.worker", "worker", "--loglevel=info"] 