from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from app.core.database import get_db
from app.core.security import get_password_hash, verify_password, create_access_token, _DUMMY_HASH
from app.core.logging import get_logger
from app.models.db import User
from app.models.schemas import UserCreate, UserResponse, Token

router = APIRouter()
logger = get_logger(__name__)

@router.post("/register", response_model=UserResponse)
def register(user: UserCreate, db: Session = Depends(get_db)):
    try:
        db_user = db.query(User).filter(User.email == user.email).first()
        if db_user:
            raise HTTPException(status_code=400, detail="Email already registered")
        
        hashed_password = get_password_hash(user.password)
        new_user = User(email=user.email, password_hash=hashed_password)
        db.add(new_user)
        db.commit()
        db.refresh(new_user)
        logger.info("User registered successfully: %s", user.email)
        return new_user
    except HTTPException:
        raise
    except SQLAlchemyError as e:
        db.rollback()
        logger.error("Database error during registration for %s: %s", user.email, type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please try again later.",
        )
    except Exception as e:
        db.rollback()
        logger.error("Unexpected error during registration for %s: %s", user.email, type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during registration.",
        )

@router.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        user = db.query(User).filter(User.email == form_data.username).first()
    except SQLAlchemyError as e:
        logger.error("Database error during user lookup for %s: %s", form_data.username, type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database service unavailable. Please try again later.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error("Unexpected error during user lookup for %s: %s", form_data.username, type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred. Please try again later.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        password_valid = verify_password(
            form_data.password, user.password_hash if user else _DUMMY_HASH
        )
    except Exception as e:
        logger.error("Password verification error for %s: %s", form_data.username, type(e).__name__)
        password_valid = False

    if not user or not password_valid:
        logger.warning("Failed login attempt for user: %s", form_data.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        access_token = create_access_token(data={"sub": user.email})
    except Exception as e:
        logger.error("Token creation failed for %s: %s", user.email, type(e).__name__)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    logger.info("User logged in successfully: %s", user.email)
    return {"access_token": access_token, "token_type": "bearer"}
