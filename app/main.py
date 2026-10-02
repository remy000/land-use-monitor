from contextlib import asynccontextmanager

from fastapi import FastAPI
from app.api.system import router as system_router
from app.api.image import router as image_router
from app.core.config import get_settings
from app.services.predictor import LandUsePredictor



@asynccontextmanager
async def lifespan(app:FastAPI):
    settings=get_settings()
    app.state.predictor=LandUsePredictor(
        settings.onnx_path, settings.stats_path, settings.class_names_path
    )
    print(f"Model Loaded: {settings.onnx_path} ({len(app.state.predictor.class_names)} classes)")
    yield

settings=get_settings()
app=FastAPI(title="Land Monitor API", description="API for monitoring land usage and changes", 
            version=settings.app_version, lifespan=lifespan)

app.include_router(system_router)
app.include_router(image_router)

