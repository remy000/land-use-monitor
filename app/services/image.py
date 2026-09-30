# app/services/images.py
import io

from PIL import Image, UnidentifiedImageError

from app.schemas.images import ImageInfo

ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}


class InvalidImageError(ValueError):
    """The bytes are not a readable image."""


class UnsupportedFormatError(ValueError):
    """The image is valid, but its format is not allowed."""


def inspect_image(data: bytes, filename: str) -> ImageInfo:
    try:
        img = Image.open(io.BytesIO(data))
    except UnidentifiedImageError as exc:
        raise InvalidImageError("File is not a readable image.") from exc

    with img:
        if img.format not in ALLOWED_FORMATS:
            allowed = ", ".join(sorted(ALLOWED_FORMATS))
            raise UnsupportedFormatError(
                f"Format {img.format} is not supported. Allowed: {allowed}."
            )

        try:
            img.load()
        except OSError as exc:
            raise InvalidImageError("Image file is corrupted or truncated.") from exc

        return ImageInfo(
            filename=filename,
            format=img.format,
            width=img.width,
            height=img.height,
            mode=img.mode,
            size_bytes=len(data),
        )