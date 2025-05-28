import os
import uuid
import json
import re
import tempfile
import logging
from io import BytesIO
from typing import List, Dict

import pandas as pd
import requests
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract
from pydantic import BaseModel
import chardet

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://preview--docu-query-flow.lovable.app",
        "https://web.postman.co",
        "https://preview--disbursement-match-wizard.lovable.app",
        "https://preview--disbursement-request.lovable.app",
        "https://disbursement-request.lovable.app",
        "https://id-preview--c5a7e71e-1325-42ec-bb47-cffe6e600178.lovable.app",
        "https://c5a7e71e-1325-42ec-bb47-cffe6e600178.lovableproject.com",
        "https://c5a7e71e-1325-42ec-bb47-cffe6e600178.lovable.app",
        "https://disbursement-request-check.lovable.app"
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.get("/")
def health():
    return {"status": "ok"}

TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
FIELDS = [
    "Buyer PAN", "Buyer Name", "Invoice ID", "Invoice date (YYYY-MM-DD)", 
    "Taxable Amount", "GST (amount)", "Total Invoice Amount"
]
FIELD_ALIASES = {
    "Invoice date (YYYY-MM-DD)": ["invoice date", "date"]
}

class OCRMatch(BaseModel):
    invoice_id: str
    matched_fields: Dict[str, Dict[str, str]]
    ocr_confidence: float

def tesseract_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

def normalize_number(value: str) -> str:
    try:
        return str(float(str(value).replace(",", "")))
    except:
        return str(value)

def llm_extract(text: str) -> Dict[str, str]:
    prompt = f"""
You are an AI that extracts invoice fields. Extract ONLY the following fields from the provided invoice text and return a raw JSON dictionary — no comments, no formatting, no markdown.
Wrap your JSON response between [[JSON]] and [[/JSON]]

Fields: {', '.join(FIELDS)}

Invoice Text:
{text}

Respond ONLY with valid JSON enclosed by markers.
If a value is not found, set it as "Not Extracted".
"""
    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 3072
    }
    response = requests.post(TOGETHER_URL, json=payload, headers=headers, verify=False)
    if response.status_code != 200:
        logger.error("LLM extraction failed: %s", response.text)
        raise HTTPException(status_code=500, detail="LLM extraction failed")
    extracted_text = response.json()['choices'][0]['message']['content'].strip()
    logger.info("🧠 LLM raw output:\n%s", extracted_text)
    match = re.search(r'\[\[JSON\]\](.*?)\[\[/JSON\]\]', extracted_text, re.DOTALL)
    if not match:
        raise HTTPException(status_code=500, detail="Invalid JSON block")
    try:
        raw_data = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error("⚠️ JSON decode failed: %s", e)
        raise HTTPException(status_code=500, detail="Malformed JSON")
    parsed = {field: raw_data.get(field, raw_data.get("Invoice date", "Not Extracted") if field == "Invoice date (YYYY-MM-DD)" else "Not Extracted") for field in FIELDS}
    return parsed

async def read_excel_resilient(excel_file: UploadFile) -> pd.DataFrame:
    contents = await excel_file.read()
    excel_io = BytesIO(contents)
    tried = []
    try:
        df = pd.read_excel(excel_io, engine="openpyxl")
        return df
    except Exception as e:
        tried.append(f"openpyxl: {e}")
    excel_io.seek(0)
    try:
        df = pd.read_excel(excel_io, engine="xlrd")
        return df
    except Exception as e:
        tried.append(f"xlrd: {e}")
    excel_io.seek(0)
    try:
        encoding = chardet.detect(contents)["encoding"] or "utf-8"
        df = pd.read_csv(BytesIO(contents), encoding=encoding)
        return df
    except Exception as e:
        tried.append(f"csv: {e}")
    raise HTTPException(status_code=400, detail="❌ Could not read uploaded file. Tried:\n" + "\n".join(tried))

@app.post("/ocr-and-match/")
async def ocr_and_match(excel_file: UploadFile = File(...), documents: List[UploadFile] = File(...)):
    df = await read_excel_resilient(excel_file)
    df.columns = [col.strip().lower() for col in df.columns]
    id_col = next((k for k in ["invoice id", "invoice_id", "invoice number", "invoice no", "invoice"] if k in df.columns), None)
    if not id_col:
        raise HTTPException(status_code=400, detail=f"'Invoice ID' column not found in Excel. Found: {df.columns.tolist()}")
    df = df.fillna("").set_index(id_col)
    excel_data = {str(k).strip().lower(): v for k, v in df.to_dict(orient="index").items()}
    results = []
    for doc in documents:
        file_bytes = await doc.read()
        ext = doc.filename.split('.')[-1].lower()
        if ext == 'pdf':
            images = convert_from_bytes(file_bytes, dpi=150, fmt='jpeg', thread_count=1, first_page=1, last_page=2)
            text = "\n".join([tesseract_ocr(img) for img in images])
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            image = Image.open(tmp_path)
            text = tesseract_ocr(image)
            os.unlink(tmp_path)
        extracted_fields = llm_extract(text)
        invoice_id = extracted_fields.get("Invoice ID", "").strip()
        if not invoice_id or invoice_id.lower() == "not extracted":
            match = re.search(r"(?:Invoice\s*(?:ID|No|#)?[:\-]?\s*)([A-Za-z0-9\-\/]+)", text, re.IGNORECASE)
            invoice_id = match.group(1).strip() if match else "Not Extracted"
        excel_row = excel_data.get(invoice_id.lower(), {})
        excel_row["invoice id"] = invoice_id
        matched_fields = {}
        for field in FIELDS:
            excel_val = excel_row.get(field.lower(), "Not in Excel")
            if field in FIELD_ALIASES and excel_val == "Not in Excel":
                for alias in FIELD_ALIASES[field]:
                    if alias in excel_row:
                        excel_val = excel_row[alias]
                        break
            ocr_val = extracted_fields.get(field, "Not Extracted")
            if pd.isna(excel_val):
                excel_val = ""
            elif isinstance(excel_val, pd.Timestamp):
                excel_val = excel_val.strftime("%Y-%m-%d")
            else:
                excel_val = str(excel_val).strip()
            ocr_val = "" if isinstance(ocr_val, dict) else str(ocr_val).strip()
            if any(x in field.lower() for x in ["amount", "gst", "total"]):
                norm_excel = normalize_number(excel_val)
                norm_ocr = normalize_number(ocr_val)
            else:
                norm_excel = excel_val
                norm_ocr = ocr_val
            match_status = "Match" if norm_excel == norm_ocr and ocr_val != "Not Extracted" else "No Match"
            matched_fields[field] = {
                "Excel Value": excel_val,
                "OCR Value": ocr_val,
                "Match Status": match_status
            }
        results.append(OCRMatch(
            invoice_id=invoice_id or "Not Extracted",
            matched_fields=matched_fields,
            ocr_confidence=85.0
        ).dict())
    return JSONResponse(content={"results": results})
