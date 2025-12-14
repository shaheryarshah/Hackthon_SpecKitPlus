import os
from datetime import datetime, timedelta
from typing import Optional
from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel

# Initialize password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Initialize security schemes
security = HTTPBearer()

# Secret key for JWT tokens (use a strong secret in production)
SECRET_KEY = os.getenv("SECRET_KEY", "fallback-secret-key-for-development")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30


class TokenData(BaseModel):
    session_id: str = None


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a plain password against its hashed version
    """
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """
    Generate a hash for a plain password
    """
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token with optional expiration
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[TokenData]:
    """
    Verify a JWT token and return the token data if valid
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        session_id: str = payload.get("sub")
        if session_id is None:
            return None
        token_data = TokenData(session_id=session_id)
    except JWTError:
        return None
    return token_data


async def get_current_session(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Get the current session from the token in the request
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    token_data = verify_token(credentials.credentials)
    if token_data is None:
        raise credentials_exception
    return token_data.session_id


def check_rate_limit(session_id: str) -> bool:
    """
    Check if the session has exceeded rate limits
    """
    from backend.src.utils.rate_limiter import RateLimiter
    return RateLimiter.is_allowed(session_id)