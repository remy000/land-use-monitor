import pytest
from tests.conftest import TEST_UPLOAD_LIMIT
from tests.helpers import make_image_bytes


def test_health(client):
    response=client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status":"healthy"}


def test_predict_return_top(client):
    files={"file":("title.png", make_image_bytes(), "image/png")}
    response=client.post("/predict", files=files)