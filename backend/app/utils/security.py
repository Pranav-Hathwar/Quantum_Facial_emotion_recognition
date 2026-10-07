"""Password hashing (bcrypt) and JWT helpers. Plain-text passwords are never stored or logged."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import bcrypt
import jwt


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8")[:72], bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode("utf-8")[:72], password_hash.encode("utf-8"))
    except ValueError:
        return False


def create_access_token(subject: str, role: str, secret: str, algorithm: str, expire_minutes: int) -> str:
    exp = datetime.now(timezone.utc) + timedelta(minutes=expire_minutes)
    return jwt.encode({"sub": subject, "role": role, "exp": exp}, secret, algorithm=algorithm)


def decode_token(token: str, secret: str, algorithm: str) -> dict:
    return jwt.decode(token, secret, algorithms=[algorithm])
