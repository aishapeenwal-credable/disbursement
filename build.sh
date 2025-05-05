#!/usr/bin/env bash

set -o errexit
set -o nounset

# Install Tesseract OCR
apt-get update
apt-get install -y tesseract-ocr libglib2.0-0 libsm6 libxext6 libxrender1

# Install Python deps
pip install --upgrade pip
pip install -r requirements.txt
