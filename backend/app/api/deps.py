"""FastAPI dependencies: state access, DB session, authentication, role checks."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.database.session import db_session_dependency
from backend.app.models import Role, User
from backend.app.utils.security import decode_token

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)
ROLE_RANK = {Role.VIEWER.value: 1, Role.OPERATOR.value: 2, Role.ADMIN.value: 3}


def get_state(request: Request):
    return request.app.state


def get_db(request: Request) -> Iterator[Session]:
    yield from db_session_dependency(request.app.state.db)


@dataclass
class Principal:
    email: str
    name: str
    role: str


def principal_from_token(state, db: Session, token: str | None) -> Principal:
    if state.settings.auth_disabled:
        return Principal("demo@local", "Demo (auth disabled)", Role.ADMIN.value)
    if not token:
        raise HTTPException(401, "Not authenticated.", headers={"WWW-Authenticate": "Bearer"})
    try:
        payload = decode_token(token, state.settings.jwt_secret, state.settings.jwt_algorithm)
    except jwt.PyJWTError as exc:
        raise HTTPException(401, "Invalid or expired token.", headers={"WWW-Authenticate": "Bearer"}) from exc
    user = db.scalar(select(User).where(User.email == payload.get("sub")))
    if user is None:
        raise HTTPException(401, "User no longer exists.")
    return Principal(user.email, user.name, user.role)


def current_user(request: Request, token: str | None = Depends(oauth2), db: Session = Depends(get_db)) -> Principal:
    return principal_from_token(request.app.state, db, token)


def require_role(minimum: Role):
    def checker(user: Principal = Depends(current_user)) -> Principal:
        if ROLE_RANK.get(user.role, 0) < ROLE_RANK[minimum.value]:
            raise HTTPException(403, f"This action requires the {minimum.value} role or higher.")
        return user
    return checker
