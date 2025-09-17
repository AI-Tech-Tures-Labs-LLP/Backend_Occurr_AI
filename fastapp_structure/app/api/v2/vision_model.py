# from fastapi import FastAPI, UploadFile, File, HTTPException, APIRouter
# from pydantic import BaseModel
# import aiohttp
# import os

# router = APIRouter()

# # You can set your Groq API key in an environment variable
# GROQ_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_GROQ_API_KEY_HERE")
# # Choose model: either "llama‑3.2‑11B‑vision‑preview" or "llama‑3.2‑90B‑vision‑preview"
# GROQ_MODEL = os.getenv("OPENAI_MODEL", "llama‑3.2‑90B‑vision‑preview")


# class OCRResult(BaseModel):
#     text: str
#     # you can add more fields as per Groq’s response, e.g. bounding boxes, structured JSON etc.


# @router.post("/ocr", response_model=OCRResult)
# async def perform_ocr(file: UploadFile = File(...)):
#     # Check file is a supported image
#     filename = file.filename.lower()
#     if not (filename.endswith(".jpg") or filename.endswith(".jpeg") or filename.endswith(".png") or filename.endswith(".pdf")):
#         raise HTTPException(status_code=400, detail="Unsupported file type. Use jpg, jpeg, png, or pdf.")

#     # Read file contents
#     content = await file.read()
    
#     # Send to Groq OCR API
#     url = "https://api.groq.com/openai/v1/vision/ocr"  # (Note: replace with actual endpoint if different)
#     headers = {
#         "Authorization": f"Bearer {GROQ_API_KEY}",
#         "Content-Type": "application/json"
#     }
#     # If the API expects multipart/form‑data instead of JSON with base64, adjust accordingly
#     # We'll send base64 encoded image in JSON here for example

#     import base64
#     image_b64 = base64.b64encode(content).decode("utf-8")

#     payload = {
#         "model": GROQ_MODEL,
#         "image": image_b64,
#         # Possibly additional instructions or options:
#         # "json_mode": True, "additional_instructions": "Preserve layout" etc.
#     }

#     async with aiohttp.ClientSession() as session:
#         async with session.post(url, json=payload, headers=headers) as resp:
#             if resp.status != 200:
#                 text = await resp.text()
#                 raise HTTPException(status_code=resp.status, detail=f"Error from Groq API: {text}")
#             resp_json = await resp.json()

#     # Extract text from response
#     # The structure depends on Groq's response format
#     # Assuming resp_json has something like: { "data": { "text": "..."} }
#     extracted_text = resp_json.get("data", {}).get("text")
#     if extracted_text is None:
#         # Fallback or different key, inspect resp_json
#         extracted_text = str(resp_json)

#     return OCRResult(text=extracted_text)





