# ==========================================
# STAGE 1: The "Builder" (Heavy, contains compilers)
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install the heavy C++ compilers needed for llama.cpp
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    gcc \
    g++

# Copy your requirements and install them into a specific local folder
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# ==========================================
# STAGE 2: The "Runtime" (Tiny, only what you need)
# ==========================================
FROM python:3.11-slim

WORKDIR /app

# Llama.cpp requires OpenMP for CPU multi-threading. We install just this tiny library.
RUN apt-get update && apt-get install -y libgomp1 && rm -rf /var/lib/apt/lists/*

# Copy ONLY the compiled Python packages from the Builder stage.
# The heavy C++ compilers are left behind!
COPY --from=builder /root/.local /root/.local

# Tell Python where to find those packages
ENV PATH=/root/.local/bin:$PATH

# Copy your actual Python script into the container
COPY main.py .

# Tell Docker what to do when the container starts
CMD ["python", "main.py"]