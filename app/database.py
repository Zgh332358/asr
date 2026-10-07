"""Async SQLAlchemy engine and session management.

When DATABASE_URL is empty (default), all operations are no-ops — the service
runs in fully stateless mode for backward compatibility.
"""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import HTTPException
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings

# Module-level singletons (initialized at startup)
_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


async def init_database(
    url: str | None = None,
    pool_size: int | None = None,
    pool_overflow: int | None = None,
) -> None:
    """Initialize the async database engine and session factory.

    Must be called once during application startup. No-op if DATABASE_URL is empty.

    Args:
        url: PostgreSQL connection URL. Defaults to settings.database_url.
        pool_size: Connection pool size. Defaults to settings.database_pool_size.
        pool_overflow: Max overflow connections. Defaults to settings.database_pool_overflow.
    """
    global _engine, _session_factory

    _defaults = Settings.resolve(None)

    url = url or _defaults.database_url
    if not url.strip():
        return

    pool_size = pool_size or _defaults.database_pool_size
    pool_overflow = pool_overflow or _defaults.database_pool_overflow

    _engine = create_async_engine(
        url,
        pool_size=pool_size,
        max_overflow=pool_overflow,
        pool_pre_ping=True,
        echo=False,
    )

    _session_factory = async_sessionmaker(
        _engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )


async def close_database() -> None:
    """Dispose the database engine and release all connections.

    Must be called during application shutdown. No-op if database was never initialized.
    """
    global _engine, _session_factory

    if _engine is not None:
        await _engine.dispose()
        _engine = None
        _session_factory = None


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields an async database session.

    Usage:
        @router.get("/items")
        async def list_items(db: AsyncSession = Depends(get_db)):
            ...
    """
    if _session_factory is None:
        raise HTTPException(status_code=503, detail="Database not configured. Set DATABASE_URL to enable database features.")

    async with _session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


def get_session_factory() -> async_sessionmaker[AsyncSession] | None:
    """Return the session factory, or None if database is not configured."""
    return _session_factory


def is_database_ready() -> bool:
    """Return True if the database engine is initialized and ready."""
    return _engine is not None and _session_factory is not None
