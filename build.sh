#!/usr/bin/env bash

# Update and install Tesseract OCR
apt-get update && apt-get install -y tesseract-ocr

# Install Python packages
pip install -r requirements.txt
