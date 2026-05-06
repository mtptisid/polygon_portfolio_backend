"""
JWT Authentication for Admin Panel

Provides JWT token generation, verification, and admin authentication.
"""

import os
import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from collections import defaultdict
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

logger = logging.getLogger(__name__)

# Security
security = HTTPBearer()

# Rate limiting storage (in-memory, simple implementation)
login_attempts = defaultdict(list)
MAX_LOGIN_ATTEMPTS = 5
RATE_LIMIT_WINDOW = 900  # 15 minutes in seconds


def get_secret_key() -> str:
    """
    Get or generate SECRET_KEY for JWT signing.
    
    Returns:
        SECRET_KEY string (minimum 32 characters)
    """
    secret_key = os.getenv("SECRET_KEY")
    
    if not secret_key:
        # Generate a random key and warn
        secret_key = secrets.token_urlsafe(32)
        logger.warning(
            "SECRET_KEY not set in environment variables. "
            "Generated temporary key. Set SECRET_KEY for production!"
        )
    
    if len(secret_key) < 32:
        logger.error("SECRET_KEY must be at least 32 characters long")
        raise ValueError("SECRET_KEY too short (minimum 32 characters)")
    
    return secret_key


def get_admin_password() -> Optional[str]:
    """
    Get ADMIN_PASSWORD from environment variables.
    
    Returns:
        Admin password or None if not set
    """
    password = os.getenv("ADMIN_PASSWORD")
    
    if not password:
        logger.critical(
            "ADMIN_PASSWORD not set in environment variables. "
            "Admin login will be disabled!"
        )
    
    return password


def get_jwt_expiration_hours() -> int:
    """
    Get JWT expiration time in hours.
    
    Returns:
        Expiration hours (default: 24)
    """
    try:
        hours = int(os.getenv("JWT_EXPIRATION_HOURS", "24"))
        return hours
    except ValueError:
        logger.warning("Invalid JWT_EXPIRATION_HOURS, using default: 24")
        return 24


def check_rate_limit(ip_address: str) -> bool:
    """
    Check if IP address has exceeded login rate limit.
    
    Args:
        ip_address: Client IP address
        
    Returns:
        True if within limit, False if exceeded
    """
    now = datetime.utcnow().timestamp()
    
    # Clean old attempts
    login_attempts[ip_address] = [
        attempt_time for attempt_time in login_attempts[ip_address]
        if now - attempt_time < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(login_attempts[ip_address]) >= MAX_LOGIN_ATTEMPTS:
        logger.warning(f"Rate limit exceeded for IP: {ip_address}")
        return False
    
    return True


def record_login_attempt(ip_address: str):
    """
    Record a login attempt for rate limiting.
    
    Args:
        ip_address: Client IP address
    """
    now = datetime.utcnow().timestamp()
    login_attempts[ip_address].append(now)


def verify_password(provided_password: str, ip_address: str = "unknown") -> bool:
    """
    Verify admin password with constant-time comparison.
    
    Args:
        provided_password: Password provided by user
        ip_address: Client IP address for rate limiting
        
    Returns:
        True if password matches, False otherwise
        
    Raises:
        HTTPException: If rate limit exceeded or password not configured
    """
    # Check rate limit
    if not check_rate_limit(ip_address):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many login attempts. Please try again in 15 minutes."
        )
    
    # Record attempt
    record_login_attempt(ip_address)
    
    # Get admin password
    admin_password = get_admin_password()
    
    if not admin_password:
        logger.error("Admin login attempted but ADMIN_PASSWORD not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin authentication not configured"
        )
    
    # Constant-time comparison to prevent timing attacks
    result = secrets.compare_digest(provided_password, admin_password)
    
    if result:
        logger.info(f"Successful admin login from IP: {ip_address}")
    else:
        logger.warning(f"Failed admin login attempt from IP: {ip_address}")
    
    return result


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.
    
    Args:
        data: Data to encode in the token
        expires_delta: Optional custom expiration time
        
    Returns:
        Encoded JWT token string
    """
    to_encode = data.copy()
    
    # Set expiration
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        hours = get_jwt_expiration_hours()
        expire = datetime.utcnow() + timedelta(hours=hours)
    
    to_encode.update({"exp": expire})
    
    # Encode token
    secret_key = get_secret_key()
    encoded_jwt = jwt.encode(to_encode, secret_key, algorithm="HS256")
    
    logger.debug(f"Access token created (expires: {expire})")
    
    return encoded_jwt


def verify_token(token: str) -> Dict:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string
        
    Returns:
        Decoded token payload
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        secret_key = get_secret_key()
        payload = jwt.decode(token, secret_key, algorithms=["HS256"])
        
        # Check if token has required claims
        if "role" not in payload:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token: missing role claim"
            )
        
        return payload
        
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )


async def get_current_admin(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Dict:
    """
    FastAPI dependency to get current admin user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        Admin user data from token
        
    Raises:
        HTTPException: If token is invalid or user is not admin
    """
    token = credentials.credentials
    payload = verify_token(token)
    
    # Check admin role
    if payload.get("role") != "admin":
        logger.warning(f"Non-admin role attempted admin access: {payload.get('role')}")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    
    return payload


def create_admin_token(ip_address: str = "unknown") -> tuple[str, datetime]:
    """
    Create an admin JWT token.
    
    Args:
        ip_address: Client IP address for logging
        
    Returns:
        Tuple of (token, expiration_datetime)
    """
    hours = get_jwt_expiration_hours()
    expires_delta = timedelta(hours=hours)
    expires_at = datetime.utcnow() + expires_delta
    
    token_data = {
        "role": "admin",
        "created_at": datetime.utcnow().isoformat(),
        "ip": ip_address
    }
    
    token = create_access_token(token_data, expires_delta)
    
    logger.info(f"Admin token created for IP: {ip_address} (expires: {expires_at})")
    
    return token, expires_at
