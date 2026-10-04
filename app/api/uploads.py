from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from PIL import Image
from app.services.image import  open_image,InvalidImageError, UnsupportedFormatError

def read_upload(file: UploadFile, max_size: int) -> bytes:
    data = file.file.read(max_size + 1)
    if len(data) > max_size:
        limit_mb = max_size / (1024 * 1024)
        raise HTTPException(status_code=413, detail=f"File size exceeds the maximum limit of {limit_mb:.2f} MB.")
    return data

def open_upload(data:bytes)->Image.Image:
    try:
        return open_image(data, filename="unknown")
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc