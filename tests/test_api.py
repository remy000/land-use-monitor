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

    assert response.status_code == 200
    body= response.json()

    assert body["label"] == "Residential"
    assert body["confidence"] == 0.7
    assert len(body["top_k"]) == 3
    assert body["filename"] == "title.png"


    @pytest.mark.parametrize(
        ("payload", "expected_status"),
        [
            (b"I am not an image", 400),
            (make_image_bytes(fmt="GIF"), 415)
        ],
    )
    def test_predict_reject_bad_uploads(client, payload, expected_status):
        files={"file":("upload", payload, "application/octet-stream")}

        response=client.post("/predict", files=files)
        assert response.status_code == expected_status