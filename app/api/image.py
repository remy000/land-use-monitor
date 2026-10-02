from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from app.schemas.images import ImageInfo
from app.services.image import (
    inspect_image,
    InvalidImageError,
    UnsupportedFormatError
)
from app.core.config import get_settings

router = APIRouter(tags=["Images"], prefix="/images")

# MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
settings = get_settings()
MAX_FILE_SIZE = settings.max_upload_size

@router.post("/inspect", response_model=ImageInfo)
def inspect_upload(file : UploadFile):
    data=file.file.read(MAX_FILE_SIZE + 1)
    if len(data) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail="File size exceeds the maximum limit.")

    try:
        return inspect_image(data, filename=file.filename or "unknown")
    except InvalidImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except UnsupportedFormatError as exc:
        raise HTTPException(status_code=415, detail=str(exc)) from exc