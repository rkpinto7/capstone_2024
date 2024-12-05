# Use a specific platform tag
FROM --platform=linux/amd64 python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DJANGO_SETTINGS_MODULE=capstone.settings \
    DEBUG=False

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy static files first
COPY static /app/static

# Create necessary directories
RUN mkdir -p /app/media/excel_uploads \
    && mkdir -p /app/staticfiles

# Copy remaining project files
COPY . .

# Collect static files before changing permissions
RUN python manage.py collectstatic --noinput --clear

# Security: Run as non-root user
RUN useradd -m myuser \
    && chown -R myuser:myuser /app /opt/venv /app/staticfiles /app/media
USER myuser

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/ || exit 1

# Expose port
EXPOSE 8000

# Start gunicorn with proper settings
CMD ["gunicorn", \
     "--bind", "0.0.0.0:8000", \
     "--workers", "3", \
     "--timeout", "120", \
     "--access-logfile", "-", \
     "--error-logfile", "-", \
     "capstone.wsgi:application"]