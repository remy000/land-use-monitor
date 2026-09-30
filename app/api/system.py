

from fastapi import APIRouter, Request
from app.schemas.system import HealthResponse, AppInfo
router = APIRouter(tags=["System"])

@router.get("/health", response_model=HealthResponse)
def health_check():
    return {"status": "healthy"}

@router.get("/", response_model=AppInfo)
def get_info(request:Request):
    return{
        AppInfo(title=request.app.title, version=request.app.version)
    }