FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (including python3-distutils)
RUN apt-get update && apt-get install -y \
    python3-dev \
    libpq-dev \
    python3-distutils \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file and install dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY . .

# Expose port for the app
EXPOSE 5001

# Run the application
CMD ["python", "app.py"]
