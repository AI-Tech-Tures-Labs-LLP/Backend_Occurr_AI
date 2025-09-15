import os, uuid, mimetypes
from pathlib import Path
from fastapi import UploadFile, HTTPException

APP_BASE_URL = os.getenv("APP_BASE_URL", "http://127.0.0.1:8000")
UPLOAD_DIR = Path(os.getenv("LOCAL_UPLOAD_DIR", "uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

def save_image_locally(file: UploadFile, subdir: str = "journal/meals") -> str:
    if not file:
        raise ValueError("No file provided")

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Only image files are allowed")

    ext = Path(file.filename or "").suffix
    if not ext:
        ext = mimetypes.guess_extension(file.content_type) or ".jpg"

    rel_dir = UPLOAD_DIR / subdir
    rel_dir.mkdir(parents=True, exist_ok=True)

    fname = f"{uuid.uuid4().hex}{ext}"
    fpath = rel_dir / fname

    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")

    with open(fpath, "wb") as out:
        out.write(data)

    # Public URL (served by StaticFiles mount above)
    # Example: http://127.0.0.1:8000/uploads/journal/meals/<file>
    rel_url = f"/uploads/{subdir}/{fname}".replace("\\", "/")
    return f"{APP_BASE_URL}{rel_url}"