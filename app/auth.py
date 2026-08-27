from fastapi import Header, HTTPException
from app.config import API_KEY


def require_api_key(x_api_key: str = Header(default=None, alias="X-API-Key")) -> None:
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
