FROM python:3.11-slim

# Install tesseract and other dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    poppler-utils \
    && apt-get clean

# Set working directory
WORKDIR /app

# Copy all code
COPY . /app

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port
EXPOSE 10000

# Start the FastAPI server
CMD ["uvicorn", "disbursement:app", "--host", "0.0.0.0", "--port", "10000"]
