from pydantic import BaseModel, Field

class ClassProbability(BaseModel):
    label:str
    probability:float=Field(ge=0, le=1)

class PredictionResponse(BaseModel):
    filename:str
    label:str
    confidence:float=Field(ge=0, le=1)
    top_k: list[ClassProbability]
    model_version:str
