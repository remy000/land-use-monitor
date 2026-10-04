from typing import Annotated
from fastapi import APIRouter, HTTPException, Query, UploadFile
from app.api.deps import PredictorDependency, SettingsDependency
from app.api.uploads import open_upload, read_upload
from app.schemas.prediction import ClassProbability, PredictionResponse
from app.services.predictor import InvalidSizeError


router=APIRouter(tags=["predictions"])

def predict(
        file:UploadFile,
        predictor:PredictorDependency,
        settings:SettingsDependency,
        top_k:Annotated[int, Query(ge=1, le=10)]=3
):
    data=read_upload(file, settings.max_upload_size)
    image=open_upload(data=data)

    try:
        ranked=predictor.predict(image, top_k=top_k)
    except InvalidSizeError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    label, confidence=ranked[0]

    return PredictionResponse(
        filename=file.filename or "unknown",
        label=label,
        confidence=confidence,
        top_k=[ClassProbability(label=name, probability=p) for name, p in ranked ],
        model_version=settings.model_version
    )
