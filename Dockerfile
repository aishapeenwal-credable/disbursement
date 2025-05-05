FROM python:3.11-slim

# Install tesseract and dependencies
RUN apt-get update && \
    apt-get install -y tesseract-ocr libtesseract-dev poppler-utils gcc && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Expose port
EXPOSE 10000

# Start the app
CMD ["uvicorn", "disbursement:app", "--host", "0.0.0.0", "--port", "10000"]
