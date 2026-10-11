import pytest
from fastapi.testclient import TestClient
from app.api.deps import get_predictor
from app.core.config import Settings, get_settings
from app.main import app
from tests.helpers import FakePredictor

TEST_UPLOAD_LIMIT=10_000

@pytest.fixture
def client():
    app.dependency_overrides[get_predictor] = lambda:FakePredictor()
    app.dependency_overrides[get_settings] = lambda: Settings(max_upload_size=TEST_UPLOAD_LIMIT)
    yield TestClient(app)
    app.dependency_overrides.clear()