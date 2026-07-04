FROM python:3.10-slim-bullseye

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Install system dependencies required for OpenCV, MySQL, audio processing and wkhtmltopdf
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    pkg-config \
    default-libmysqlclient-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    wkhtmltopdf \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies before copying the rest of the application
# This caches the pip install step if requirements.txt hasn't changed.
COPY requirements.txt /app/
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy the current directory contents into the container at /app
COPY . /app/

# Expose the port the app runs on (adjust if different)
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]
