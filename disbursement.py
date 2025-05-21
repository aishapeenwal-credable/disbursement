import os
import uuid
import requests
import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pdf2image import convert_from_bytes
import pytesseract
from PIL import Image
import tempfile
from typing import List, Dict
from pydantic import BaseModel
import json
import re
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI()

# Add health check route
@app.get("/")
def health():
    return {"status": "ok"}

# Enable CORS
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
        "https://c5a7e71e-1325-42ec-bb47-cffe6e600178.lovable.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Together.ai API configuration
TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
LLM_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"
TOGETHER_URL = "https://api.together.xyz/v1/chat/completions"

# Static fields for extraction
FIELDS = [
    "Buyer PAN", "Buyer Name", "Invoice ID", "Invoice date (YYYY-MM-DD)", "Taxable Amount", "GST (amount)", "Total Invoice Amount"
]

# Helper: LLM extraction with Together.ai
def llm_extract(text: str) -> Dict[str, str]:
    prompt = f"""
You are an AI that extracts invoice fields. Extract ONLY the following fields from the provided invoice text and return a raw JSON dictionary — no comments, no formatting, no markdown.
Wrap your JSON response between [[JSON]] and [[/JSON]]

Fields: {', '.join(FIELDS)}

Example:
[[JSON]]
{{
    "Invoice ID":"2025-26/02",
    "Buyer PAN": "ABCDE1234F",
    "Buyer Name": "ANJALI SCRAP TRADERS",
    ...
}}
[[/JSON]]

Invoice Text:
{text}

Respond ONLY with valid JSON enclosed by markers.
If a value is not found, set it as "Not Extracted".
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 3072
    }

    headers = {
        "Authorization": f"Bearer {TOGETHER_API_KEY}",
        "Content-Type": "application/json"
    }

    response = requests.post(TOGETHER_URL, json=payload, headers=headers, verify=False)

    if response.status_code != 200:
        logger.error(f"LLM extraction failed: {response.text}")
        raise HTTPException(status_code=500, detail="LLM extraction failed")

    response_json = response.json()
    extracted_text = response_json['choices'][0]['message']['content'].strip()
    usage = response_json.get("usage", {})
    logger.info("Token usage: prompt=%s, completion=%s, total=%s", usage.get("prompt_tokens"), usage.get("completion_tokens"), usage.get("total_tokens"))
    logger.info("\U0001f9e0 LLM raw output:\n%s", extracted_text)

    match = re.search(r'\[\[JSON\]\](.*?)\[\[/JSON\]\]', extracted_text, re.DOTALL)
    if not match:
        logger.warning("\u26a0\ufe0f No [[JSON]] block found in LLM response")
        raise HTTPException(status_code=500, detail="Invalid JSON block")

    try:
        json_block = json.loads(match.group(1))
    except json.JSONDecodeError as e:
        logger.error("\u26a0\ufe0f JSON decode failed: %s", e)
        raise HTTPException(status_code=500, detail="Malformed JSON")

    parsed = {}
    for key in FIELDS:
        parsed[key] = json_block.get(key, json_block.get("Invoice date", "Not Extracted") if key == "Invoice date (YYYY-MM-DD)" else "Not Extracted")
    return parsed

# Normalize numeric strings for comparison
def normalize_number(value: str) -> str:
    try:
        return str(float(str(value).replace(",", "")))
    except:
        return str(value)

# OCR using pytesseract
def tesseract_ocr(image: Image.Image) -> str:
    return pytesseract.image_to_string(image)

# Response schema
class OCRMatch(BaseModel):
    invoice_id: str
    matched_fields: Dict[str, Dict[str, str]]
    ocr_confidence: float

# API: OCR-and-Match
@app.post("/ocr-and-match/")
async def ocr_and_match(
    excel_file: UploadFile = File(...),
    documents: List[UploadFile] = File(...)
):
    try:
        df = pd.read_excel(excel_file.file, engine="openpyxl")
        original_cols = df.columns.tolist()
        df.columns = [col.strip().lower() for col in df.columns]

        possible_keys = ["invoice id", "invoice_id", "invoice number", "invoice no", "invoice"]
        id_col = next((k for k in possible_keys if k in df.columns), None)

        if not id_col:
            raise HTTPException(
                status_code=400,
                detail=f"❌ 'Invoice ID' column not found in Excel. Found: {original_cols}"
            )

        df["invoice_id_temp_copy"] = df[id_col]  # Copy before indexing
        df = df.fillna("")
        df = df.set_index(id_col)
        excel_data = {str(k).strip().lower(): v for k, v in df.to_dict(orient="index").items()}
        print("📄 Available invoice IDs in Excel:", list(excel_data.keys())[:10])

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process Excel file: {e}")

    results = []

    for doc in documents:
        file_bytes = await doc.read()
        ext = doc.filename.split('.')[-1].lower()

        if ext == 'pdf':
            images = convert_from_bytes(file_bytes, dpi=150, fmt='jpeg', thread_count=1, first_page=1, last_page=2)
            texts = [tesseract_ocr(img) for img in images]
            text = "\n".join(texts)
            confidence = 85.0
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{ext}") as tmp:
                tmp.write(file_bytes)
                tmp_path = tmp.name
            image = Image.open(tmp_path)
            text = tesseract_ocr(image)
            confidence = 85.0
            os.unlink(tmp_path)

        extracted_fields = llm_extract(text)
        invoice_id = extracted_fields.get("Invoice ID") or extracted_fields.get("Invoice No", "")
        invoice_id = invoice_id.strip()

        if not invoice_id or invoice_id.lower() == "not extracted":
            logger.warning("\u26a0\ufe0f Invoice ID not extracted by LLM. Trying regex fallback...")
            match = re.search(r"(?:Invoice\s*(?:ID|No|#)?[:\-]?\s*)([A-Za-z0-9\-\/]+)", text, re.IGNORECASE)
            if match:
                invoice_id = match.group(1).strip()
                logger.info(f"✅ Recovered Invoice ID via regex: {invoice_id}")
            else:
                logger.warning("❌ Could not find Invoice ID via regex either.")

        print("🔎 Looking for Invoice ID:", invoice_id.lower())
        excel_row = excel_data.get(invoice_id.lower(), {})
        excel_row["invoice id"] = invoice_id  # Restore for matching

        matched_fields = {}
        for field in FIELDS:
            excel_val = excel_row.get(field.lower(), "Not in Excel")

            # Handle alternate field naming for known mismatches
            fallback_keys = {
    "Invoice date (YYYY-MM-DD)": ["invoice date", "date"]
}
            if field in fallback_keys and excel_val == "Not in Excel":
                for alt in fallback_keys[field]:
                    if alt in excel_row:
                        excel_val = excel_row[alt]
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

        result = OCRMatch(
            invoice_id=invoice_id or "Not Extracted",
            matched_fields=matched_fields,
            ocr_confidence=round(confidence, 2)
        )

        results.append(result.dict())

    os.makedirs("results", exist_ok=True)
    result_file = f"results/result_{uuid.uuid4()}.json"
    with open(result_file, "w") as f:
        pd.Series(results).to_json(f, orient="records", indent=4)

    return JSONResponse(content={"results": results})
