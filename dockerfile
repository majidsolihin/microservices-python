# Gunakan Python 3.11 untuk menghindari masalah distutils
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    python3-pip \
    libpq-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt requirements.txt

# Upgrade pip dan setuptools sebelum install dependencies
RUN pip install --upgrade pip setuptools && \
    pip install --no-cache-dir -r requirements.txt

# Copy seluruh aplikasi
COPY . .

# Expose port untuk aplikasi
EXPOSE 5001

# Jalankan aplikasi
CMD ["python", "app.py"]
