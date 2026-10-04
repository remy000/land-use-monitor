from typing import Annotated

from fastapi import Depends, Request
from app.core.config import get_settings, Settings
from app.services.predictor import LandUsePredictor

def get_predictor(request: Request) -> LandUsePredictor:
    return request.app.state.predictor

PredictorDependency = Annotated[LandUsePredictor, Depends(get_predictor)]
SettingsDependency = Annotated[Settings, Depends(get_settings)]