from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.app.api.deps import Principal, current_user, get_db, get_state, require_role
from backend.app.models import Role, User
from backend.app.schemas.api import TokenOut, UserCreate, UserOut
from backend.app.utils.logging import get_logger
from backend.app.utils.security import create_access_token, hash_password, verify_password

router = APIRouter(prefix="/api/auth", tags=["auth"])
log = get_logger("auth")


@router.post("/login", response_model=TokenOut)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db), state=Depends(get_state)):
    user = db.scalar(select(User).where(User.email == form.username.strip().lower()))
    if user is None or not verify_password(form.password, user.password_hash):
        log.warning("failed login", extra={"email": form.username[:80]})
        raise HTTPException(401, "Incorrect email or password.")
    s = state.settings
    token = create_access_token(user.email, user.role, s.jwt_secret, s.jwt_algorithm, s.jwt_expire_minutes)
    return TokenOut(access_token=token, user=UserOut.model_validate(user))


@router.get("/me", response_model=UserOut | dict)
def me(user: Principal = Depends(current_user), db: Session = Depends(get_db)):
    row = db.scalar(select(User).where(User.email == user.email))
    return UserOut.model_validate(row) if row else {"email": user.email, "name": user.name, "role": user.role}


@router.get("/users", response_model=list[UserOut])
def list_users(_: Principal = Depends(require_role(Role.ADMIN)), db: Session = Depends(get_db)):
    return db.scalars(select(User).order_by(User.id)).all()


@router.post("/users", response_model=UserOut, status_code=201)
def create_user(body: UserCreate, _: Principal = Depends(require_role(Role.ADMIN)), db: Session = Depends(get_db)):
    if body.role not in {r.value for r in Role}:
        raise HTTPException(422, f"role must be one of {[r.value for r in Role]}")
    email = body.email.strip().lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(409, "A user with this email already exists.")
    user = User(name=body.name, email=email, password_hash=hash_password(body.password), role=body.role)
    db.add(user)
    db.commit()
    return user
