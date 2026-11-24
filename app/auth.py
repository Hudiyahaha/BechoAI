"""Seller logins (email + password -> JWT) and anonymous customer IDs (server-signed tokens).

Every seller endpoint gets store_id from the token, never from the request body or URL.
Every customer endpoint gets user_id from a token the server signed, so nobody can pick another person's ID.
"""
import hashlib
import hmac
import logging
import os
import secrets
import uuid
from datetime import datetime, timedelta, timezone

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

log = logging.getLogger("bechoai.auth")

JWT_SECRET = os.getenv("JWT_SECRET") or ""
if not JWT_SECRET:
    JWT_SECRET = secrets.token_urlsafe(32)
    log.warning("JWT_SECRET is not set: using a random one, so all logins reset when the server restarts.")
ALGORITHM = "HS256"
SELLER_TOKEN_HOURS = 24
CUSTOMER_TOKEN_DAYS = 365

bearer = HTTPBearer(auto_error=False)


# ── passwords (stdlib scrypt, salted) ──

def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1)
    return f"scrypt${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        _, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    digest = hashlib.scrypt(password.encode(), salt=bytes.fromhex(salt_hex), n=2**14, r=8, p=1)
    return hmac.compare_digest(digest.hex(), digest_hex)


# ── tokens ──

def _make_token(claims: dict, lifetime: timedelta) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode({**claims, "iat": now, "exp": now + lifetime}, JWT_SECRET, algorithm=ALGORITHM)


def _read_token(creds: HTTPAuthorizationCredentials | None, kind: str) -> dict:
    if creds is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Missing token.")
    try:
        claims = jwt.decode(creds.credentials, JWT_SECRET, algorithms=[ALGORITHM])
    except jwt.PyJWTError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid or expired token.")
    if claims.get("kind") != kind:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong kind of token.")
    return claims


def seller_token(seller_id: int, store_id: str) -> str:
    return _make_token({"kind": "seller", "sub": str(seller_id), "store_id": store_id},
                       timedelta(hours=SELLER_TOKEN_HOURS))


def new_customer_token() -> tuple[str, str]:
    customer_id = f"c-{uuid.uuid4().hex}"
    return customer_id, _make_token({"kind": "customer", "sub": customer_id}, timedelta(days=CUSTOMER_TOKEN_DAYS))


def current_store_id(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    """Dependency for seller endpoints: the store this seller owns."""
    return _read_token(creds, "seller")["store_id"]


def current_customer_id(creds: HTTPAuthorizationCredentials | None = Depends(bearer)) -> str:
    """Dependency for customer endpoints: the anonymous customer ID the server issued."""
    return _read_token(creds, "customer")["sub"]
