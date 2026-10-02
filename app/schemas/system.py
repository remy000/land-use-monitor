from typing import Literal

from pydantic import BaseModel, ConfigDict

class HealthResponse(BaseModel):
    status:Literal["healthy"] = "healthy"

class AppInfo(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title:str
    version:str