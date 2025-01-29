import multiprocessing

# Gunicorn configuration for GPU workloads
bind = "127.0.0.1:8000"
workers = multiprocessing.cpu_count() * 2 + 1
worker_class = "sync"  # Using sync workers for GPU operations
timeout = 300  # Increased timeout for long-running ML operations
keepalive = 24
threads = 4
worker_connections = 1000
max_requests = 100
max_requests_jitter = 50

# Logging
accesslog = "/var/log/gunicorn/access.log"
errorlog = "/var/log/gunicorn/error.log"
loglevel = "info"

# Process naming
proc_name = "synthetic_satellite_api"

# SSL Configuration (if needed)
# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile" 