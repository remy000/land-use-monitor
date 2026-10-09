from fastapi import APIRouter, Request, HTTPException, UploadFile, File
from app.schemas.images import ImageInfo
from app.api.deps import SettingsDependency
from app.api.uploads import open_image, open_upload
from app.core.config import get_settings

router = APIRouter(tags=["Images"], prefix="/images")


@router.post("/inspect", response_model=ImageInfo)
def inspect_upload(file : UploadFile, settings:SettingsDependency):
    data=open_upload(file, settings.max_upload_size)
    image=open_image(data)
    return ImageInfo(
        filename=file.filename or "unknown",
        format=image.format,
        width=image.width,
        height=image.height,
        mode=image.mode,
        size_bytes=len(data),
    )