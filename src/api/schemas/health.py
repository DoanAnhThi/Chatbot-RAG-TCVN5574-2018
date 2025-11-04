from pydantic import BaseModel
from typing import Optional


class HealthResponse(BaseModel):
    status: str
    message: str
    timestamp: Optional[str] = None
