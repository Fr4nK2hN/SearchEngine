# Multi-stage build for smaller image size
FROM docker.m.daocloud.io/library/python:3.13.7-slim as builder

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM docker.m.daocloud.io/library/python:3.13.7-slim

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app

# Set the working directory
WORKDIR /app

# Copy Python packages from builder stage
COPY --from=builder /root/.local /home/app/.local

# Copy application code
COPY --chown=app:app . .

# Make sure scripts in .local are usable
ENV PATH=/home/app/.local/bin:$PATH

# Switch to non-root user
USER app

# Make port 5000 available
EXPOSE 5000

# Run the application
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "app:app"]