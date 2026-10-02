import json
from pathlib import Path
import numpy as np
import onnxruntime as ort
from PIL import Image

INPUT_SIZE = (64,64)

class InvalidSizeError(ValueError):
    """The image is not the size the model was trained on."""

def softmax(logits: np.ndarray) -> np.ndarray:
    """Compute softmax probabilities from logits."""
    shifted =  logits - logits.max()
    exp= np.exp(shifted)
    return exp / exp.sum()


class LandUsePredictor:
    def __init__(self, onx_path : Path, stats_path: Path, class_names_path: Path):
        self.session = ort.InferenceSession(str(onx_path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

        stats = json.loads(Path(stats_path).read_text())
        self.mean=np.array(stats["mean"], dtype=np.float32).reshape(3,1,1)
        self.std=np.array(stats["std"], dtype=np.float32).reshape(3,1,1)

        self.class_names=json.loads(Path(class_names_path).read_text())

        num_outputs=self.session.get_outputs()[0].shape[1]
        if num_outputs != len(self.class_names):
            raise ValueError(
                f"Model outputs {num_outputs} classes, "
                f"but {len(self.class_names)} class names were provided."
            )

    def preprocess(self, image: Image.Image) -> np.ndarray:
        if image.size != INPUT_SIZE:
            raise InvalidSizeError(f"Image must be {INPUT_SIZE}, but got {image.size}.")
        x = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
        x = x.transpose(2,0,1)  # HWC -> CHW
        x = (x - self.mean) / self.std
        return x[np.newaxis,...]

    def predict(self, image: Image.Image, top_k: int = 3) -> list[tuple[str, float]]:
        batch = self.preprocess(image)
        logits = self.session.run(None, {self.input_name: batch})[0][0]
        probs = softmax(logits)
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.class_names[i], float(probs[i])) for i in top_indices]
