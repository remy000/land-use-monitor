from pathlib import Path
import numpy as np
import pytest
from PIL import Image

from app.services.predictor import LandUsePredictor

MODEL_DIR= Path("models")
REQUIRED= ["model.onnx","eurostat_stats.json", "class_names.json"]

pytestmark=pytest.mark.skipif(
    not all((MODEL_DIR / name). exists() for name in REQUIRED
    ),
    reason="Model files not available"
)


def test_real_predictor_returns_valid_probabilities():
    predictor=LandUsePredictor(
        MODEL_DIR / "model.onnx",
        MODEL_DIR / "eurosat_stats.json",
        MODEL_DIR / "class_names.json",
    )

    image=Image.new("RGB",(64,64), "green")
    ranked=predictor.predict(image, top_k=10)


    assert len(ranked) == 10
    assert {name for name, _ in ranked} == set(predictor.class_names)
    probabilities = [p for _, p in ranked]
    assert np.isclose(sum(probabilities), 1.0, atol=1e-5)
    assert probabilities == sorted(probabilities, reverse=True)