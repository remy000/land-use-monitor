from pydantic import BaseModel

class ImageInfo(BaseModel):
    filename: str
    format: str
    width: int
    height: int
    mode: str
    size_bytes: int