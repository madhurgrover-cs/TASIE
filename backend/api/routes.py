from datetime import datetime, timezone
from fastapi import APIRouter, Depends, HTTPException, Body
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel
from backend.database import get_db
from backend.models import Feedback, FeedbackLabel, SmartMemory
from backend.scanner import git_utils, sast_core

router = APIRouter()


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────

class ScanRequest(BaseModel):
    repo_path: str
    mode: str = "diff"  # "diff" | "full"

class FeedbackRequest(BaseModel):
    code_hash: str
    developer_label: str  # "valid_vulnerability" | "false_positive" | "needs_review"

class SmartMemoryRequest(BaseModel):
    pattern: str
    pattern_type: str
    description: str | None = None


# ─── Scan ─────────────────────────────────────────────────────────────────────

@router.post("/api/scan")
def scan_repository(req: ScanRequest, db: Session = Depends(get_db)):
    """Scan a local git repository and return findings."""
    try:
        if req.mode == "diff":
            chunks = git_utils.get_diff_chunks(req.repo_path)
        else:
            chunks = git_utils.get_all_files(req.repo_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Git error: {str(e)}")

    findings = sast_core.scan_chunks(chunks, db)

    # Persist findings to feedback table (awaiting developer review)
    for finding in findings:
        existing = db.query(Feedback).filter(Feedback.code_hash == finding["code_hash"]).first()
        if not existing:
            fb = Feedback(
                code_hash=finding["code_hash"],
                code_snippet=finding["code_snippet"],
                file_path=finding.get("file_path"),
                vulnerability_type=finding.get("vulnerability_type"),
                prediction=finding["prediction"],
                confidence_score=finding.get("confidence_score"),
            )
            db.add(fb)
    db.commit()

    return {"findings": findings, "total": len(findings)}


# ─── Feedback ─────────────────────────────────────────────────────────────────

@router.post("/api/feedback")
def submit_feedback(req: FeedbackRequest, db: Session = Depends(get_db)):
    """Developer labels a finding as valid, false positive, or needs review."""
    try:
        label = FeedbackLabel(req.developer_label)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid developer_label value.")

    fb = db.query(Feedback).filter(Feedback.code_hash == req.code_hash).first()
    if not fb:
        raise HTTPException(status_code=404, detail="Finding not found.")

    fb.developer_label = label
    fb.timestamp = datetime.now(timezone.utc)
    db.commit()
    return {"status": "ok", "code_hash": req.code_hash, "label": label.value}


@router.get("/api/feedback")
def list_pending_feedback(db: Session = Depends(get_db)):
    """Returns findings that have not yet been labelled by a developer."""
    rows = db.query(Feedback).filter(Feedback.developer_label == None).order_by(Feedback.timestamp.desc()).limit(50).all()
    return [
        {
            "id": r.id,
            "code_hash": r.code_hash,
            "file_path": r.file_path,
            "vulnerability_type": r.vulnerability_type,
            "prediction": r.prediction,
            "confidence_score": r.confidence_score,
            "timestamp": r.timestamp.isoformat() if r.timestamp else None,
            "code_snippet": r.code_snippet,
        }
        for r in rows
    ]


# ─── Dashboard Analytics ──────────────────────────────────────────────────────

@router.get("/api/dashboard/metrics")
def get_dashboard_metrics(db: Session = Depends(get_db)):
    """Returns aggregated stats for the analytics dashboard."""
    total_scans = db.query(func.count(Feedback.id)).scalar()
    labelled = db.query(Feedback).filter(Feedback.developer_label.isnot(None)).all()
    total_labelled = len(labelled)

    fp_count = sum(1 for r in labelled if r.developer_label == FeedbackLabel.FALSE_POSITIVE)
    valid_count = sum(1 for r in labelled if r.developer_label == FeedbackLabel.VALID_VULNERABILITY)
    review_count = sum(1 for r in labelled if r.developer_label == FeedbackLabel.NEEDS_REVIEW)

    fp_rate = round(fp_count / total_labelled * 100, 1) if total_labelled else 0
    acceptance_rate = round(valid_count / total_labelled * 100, 1) if total_labelled else 0

    # Weekly trend: count findings per week
    from sqlalchemy import extract
    week_trends = (
        db.query(
            extract("year", Feedback.timestamp).label("year"),
            extract("week", Feedback.timestamp).label("week"),
            func.count(Feedback.id).label("count"),
        )
        .group_by("year", "week")
        .order_by("year", "week")
        .all()
    )

    # Vulnerability recurrence: how many times each vuln type appears
    vuln_counts = (
        db.query(Feedback.vulnerability_type, func.count(Feedback.id))
        .group_by(Feedback.vulnerability_type)
        .all()
    )

    # Model registry info
    from backend.ml.model_registry import get_all_versions
    versions = get_all_versions()

    return {
        "total_scans": total_scans,
        "total_labelled": total_labelled,
        "false_positive_count": fp_count,
        "valid_vulnerability_count": valid_count,
        "needs_review_count": review_count,
        "false_positive_rate_pct": fp_rate,
        "developer_acceptance_rate_pct": acceptance_rate,
        "weekly_trends": [{"year": int(w.year), "week": int(w.week), "count": w.count} for w in week_trends],
        "vulnerability_recurrence": [{"type": v[0], "count": v[1]} for v in vuln_counts],
        "model_versions": len(versions),
        "active_model": versions[-1]["version"] if versions else None,
    }


# ─── Smart Memory ─────────────────────────────────────────────────────────────

@router.post("/api/smart-memory")
def add_smart_memory(req: SmartMemoryRequest, db: Session = Depends(get_db)):
    """Add a safe pattern to Smart Memory (overrides ML predictions)."""
    entry = SmartMemory(
        pattern=req.pattern,
        pattern_type=req.pattern_type,
        description=req.description,
    )
    db.add(entry)
    db.commit()
    return {"status": "ok", "id": entry.id}


@router.get("/api/smart-memory")
def list_smart_memory(db: Session = Depends(get_db)):
    """List all Smart Memory patterns."""
    rows = db.query(SmartMemory).all()
    return [{"id": r.id, "pattern": r.pattern, "pattern_type": r.pattern_type, "description": r.description} for r in rows]


# ─── Model Versions ───────────────────────────────────────────────────────────

@router.get("/api/model/versions")
def get_model_versions():
    from backend.ml.model_registry import get_all_versions
    return get_all_versions()


@router.post("/api/model/retrain")
def trigger_retrain():
    """Manually trigger model retraining."""
    from backend.ml.retraining import train_model
    version = train_model()
    if version is None:
        return {"status": "skipped", "reason": "Not enough labelled samples, or only one label present."}
    return {"status": "ok", "version": version}
