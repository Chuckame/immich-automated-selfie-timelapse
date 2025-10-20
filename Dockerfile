FROM python:3.10-slim

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    ffmpeg \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download the shape predictor model
RUN apt-get update && apt-get install -y wget && \
    wget -nd https://github.com/JeffTrain/selfie/raw/master/shape_predictor_68_face_landmarks.dat && \
    apt-get remove -y wget && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Create output directory
RUN mkdir -p /app/output

# Expose the port the app runs on
EXPOSE 5000

# Command to run the application
CMD ["python", "main.py"]