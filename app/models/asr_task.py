"""AsrTask ORM model — ASR任务统一管理表.

Each row represents one ASR processing attempt for a corpus file.
Multiple tasks may exist for the same corpus (e.g., retries, different engines).
"""

from __future__ import annotations

import datetime
from decimal import Decimal
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.corpus import Corpus


class AsrTask(Base, TimestampMixin):
    """ASR processing task — status, engine, results, and timing."""

    __tablename__ = "asr_tasks"

    # Primary key
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)

    # Foreign key to corpus
    corpus_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("corpus.id", ondelete="CASCADE"),
        nullable=False,
        comment="关联语料ID",
    )

    # Task state
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING",
        nullable=False,
        comment="状态: PENDING, PROCESSING, SUCCESS, FAILED",
    )
    asr_engine: Mapped[str] = mapped_column(
        String(30),
        default="STEPFUN",
        nullable=False,
        comment="ASR引擎: STEPFUN, WHISPER, AZURE, ALIYUN, TENCENT, HUAWEI",
    )
    engine_config: Mapped[dict] = mapped_column(
        JSONB, default=dict, nullable=False, comment="引擎配置参数"
    )

    # Results
    result_text: Mapped[Optional[str]] = mapped_column(
        Text, nullable=True, comment="识别结果文本"
    )
    confidence: Mapped[Optional[Decimal]] = mapped_column(
        Numeric(5, 4), nullable=True, comment="整体置信度 0~1"
    )
    result_detail: Mapped[Optional[dict]] = mapped_column(
        JSONB, nullable=True, comment="详细结果（时间戳、词级置信度等）"
    )
    processing_time: Mapped[Optional[int]] = mapped_column(
        Integer, nullable=True, comment="处理耗时（毫秒）"
    )
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Timestamps (beyond the mixin)
    started_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="开始处理时间"
    )
    completed_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, comment="完成时间"
    )

    # Relationships
    corpus: Mapped["Corpus"] = relationship(
        "Corpus", back_populates="asr_tasks"
    )

    def __repr__(self) -> str:
        return f"<AsrTask id={self.id} corpus={self.corpus_id} status={self.status}>"
