from typing import Literal

from pydantic import BaseModel

class HealthResponse(BaseModel):
    status:Literal["healthy"] = "healthy"

class AppInfo(BaseModel):
    app_title:str
    version:str