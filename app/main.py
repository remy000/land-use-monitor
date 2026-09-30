from fastapi import FastAPI
from app.api import system as system_router

app=FastAPI(title="Land Monitor API", description="API for monitoring land usage and changes", version="0.1.0")

app.include_router(system_router.router)

