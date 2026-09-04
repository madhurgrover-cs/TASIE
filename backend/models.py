import enum
from datetime import datetime, timezone
from sqlalchemy import Column, Integer, String, Enum, DateTime, Boolean, Text, Float
from backend.database import Base


class FeedbackLabel(enum.Enum):
    VALID_VULNERABILITY = "valid_vulnerability"
    FALSE_POSITIVE = "false_positive"
    NEEDS_REVIEW = "needs_review"


class Feedback(Base):
    """Stores per-scan findings and developer labels for incremental learning."""
    __tablename__ = "feedback"

    id = Column(Integer, primary_key=True, index=True)
    code_hash = Column(String(255), index=True, nullable=False)
    code_snippet = Column(Text, nullable=False)
    file_path = Column(String(512), nullable=True)
    vulnerability_type = Column(String(100), nullable=True)
    prediction = Column(String(50), nullable=False)   # "vulnerable" | "safe"
    confidence_score = Column(Float, nullable=True)    # 0.0 – 1.0
    developer_label = Column(Enum(FeedbackLabel), nullable=True)  # set after review
    timestamp = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    is_used_for_training = Column(Boolean, default=False)


class SmartMemory(Base):
    """Deterministic override rules — approved patterns bypass the ML model."""
    __tablename__ = "smart_memory"

    id = Column(Integer, primary_key=True, index=True)
    pattern = Column(Text, nullable=False)              # regex or literal substring
    pattern_type = Column(String(50), nullable=False)   # "safe_library" | "approved_snippet"
    description = Column(String(255), nullable=True)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
