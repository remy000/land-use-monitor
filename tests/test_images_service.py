import pytest
from app.services.image import InvalidImageError, UnsupportedFormatError, open_image
from tests.helpers import make_image_bytes

def test_open_image_valid_png():
    image=open_image(make_image_bytes(size=(64,64), fmt="PNG"))
    assert image.size == (64,64)
    assert image.format == "PNG"