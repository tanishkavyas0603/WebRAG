from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from sqlalchemy.orm import Session
from app.core.config import settings
from app.core.database import get_db
from app.models.db import User

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

def get_current_user(db: Session = Depends(get_db), token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
        
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

import time
from collections import defaultdict
from fastapi import Request

# Simple process-local in-memory rate limiter
# Suitable for a single Uvicorn worker deployment.
_rate_limit_store = defaultdict(list)

def get_rate_limiter(requests: int = settings.RATE_LIMIT_REQUESTS, window_seconds: int = settings.RATE_LIMIT_WINDOW_SECONDS):
    def rate_limit_dependency(request: Request, current_user: User = Depends(get_current_user)):
        now = time.time()
        user_id = current_user.id
        
        # Clean up old timestamps
        _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if now - t < window_seconds]
        
        if len(_rate_limit_store[user_id]) >= requests:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Too many requests. Please try again later."
            )
            
        _rate_limit_store[user_id].append(now)
        return current_user
        
    return rate_limit_dependency
