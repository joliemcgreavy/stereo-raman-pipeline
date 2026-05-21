FROM python:3.11-slim

# System dependencies for OpenCV headless on Linux
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first — this layer is cached so rebuilds
# after code-only changes are fast (pip install doesn't re-run)
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt

# Copy the rest of the project
COPY . .

# HF Spaces requires port 7860 — configured in .streamlit/config.toml
EXPOSE 7860

CMD ["streamlit", "run", "module3_dashboard/app.py"]
