import io
from PIL import Image
from app.services.predictor import InvalidSizeError


def make_image_bytes(size=(64,64), fmt="PNG", color="green")-> bytes:
    """
        create image in memory and return it's encoded bytes
    """
    buffer=io.BytesIO()
    Image.new("RGB", size, color).save(buffer, format=fmt)
    return buffer.getvalue()

class FakePredictor:
    class_names=  ["AnnualCrop", "Forest", "Residential"]

    def predict(self, image, top_k=3):
        if image.size != (64,64):
            raise InvalidSizeError("Wrong size")
        ranked=[("Residential",0.7),("Forest", 0.2), ("AnnualCrop", 0.1)
        ]
        return ranked[:top_k]
