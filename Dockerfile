# Use an official Python runtime as a parent image
# Use a more stable Python version
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies for OpenCV and PyTorch
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Make sure all files are readable
RUN chmod -R 755 /app

# Expose the port the app runs on
EXPOSE 7860

# Command to run the application
CMD ["python", "app.py"]