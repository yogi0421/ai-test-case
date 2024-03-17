FROM python:3.8-slim

WORKDIR /home/app

# Install system dependencies required for Python packages or GPU support
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential \
    libmagic-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    jq && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Copy the Python requirements file and install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip && \
    pip install --default-timeout=360 -r requirements.txt --no-cache-dir

# Copy the rest of the application's code
COPY . /home/app

RUN mkdir /tmp/inference-result

CMD ["uvicorn", "app.api:app", "--host", "0.0.0.0", "--port", "80", "--workers", "2", "--lifespan", "off"]