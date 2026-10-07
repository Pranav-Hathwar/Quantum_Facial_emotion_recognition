"""SQLAlchemy engine/session. Works with PostgreSQL (production) and SQLite (local dev / tests)."""
from __future__ import annotations

from collections.abc import Iterator

from fastapi import HTTPException
from sqlalchemy import create_engine
from sqlalchemy.exc import OperationalError, SQLAlchemyError
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker
from sqlalchemy.pool import StaticPool

from backend.app.utils.logging import get_logger

log = get_logger("db")


class Base(DeclarativeBase):
    pass


class Database:
    def __init__(self, url: str):
        kwargs: dict = {"pool_pre_ping": True}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if ":memory:" in url:
                kwargs["poolclass"] = StaticPool
        self.url = url
        self.engine = create_engine(url, **kwargs)
        self.SessionLocal = sessionmaker(bind=self.engine, autoflush=False, expire_on_commit=False)

    def create_all(self) -> None:
        from backend.app import models  # noqa: F401  (register tables)
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self.SessionLocal()


def db_session_dependency(db: Database) -> Iterator[Session]:
    try:
        session = db.session()
        session.connection()  # fail fast if the database is down
    except (OperationalError, SQLAlchemyError) as exc:
        log.error("database unavailable", extra={"error": str(exc)[:200]})
        raise HTTPException(503, "Database unavailable. Please try again shortly.") from exc
    try:
        yield session
    except SQLAlchemyError as exc:
        session.rollback()
        log.error("database error", extra={"error": str(exc)[:200]})
        raise HTTPException(503, "Database error.") from exc
    finally:
        session.close()
