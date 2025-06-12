# Gunakan Python 3.11 dengan versi slim untuk mengurangi ukuran image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install sistem dependencies yang diperlukan
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3-dev \
    python3-pip \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pastikan menggunakan environment virtual untuk isolasi dependencies
RUN python -m venv venv
ENV PATH="/app/venv/bin:$PATH"

# Copy dan install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi
COPY . .

# Pastikan aplikasi berjalan dengan user non-root untuk keamanan
RUN adduser --disabled-password --gecos "" appuser
USER appuser

# Expose port aplikasi
EXPOSE 5001

# Jalankan aplikasi
CMD ["python", "app.py"]
