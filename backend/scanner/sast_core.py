import hashlib
import re
from sqlalchemy.orm import Session
from backend.models import SmartMemory
from backend.ml.model_registry import load_active_model

# Baseline regex patterns (used before an ML model is trained)
BASELINE_PATTERNS = [
    {"pattern": r"eval\s*\(", "type": "Code Injection", "severity": "HIGH"},
    {"pattern": r"execute\s*\(.*%.*\)", "type": "SQL Injection", "severity": "HIGH"},
    {"pattern": r"subprocess\.(call|run|Popen)\s*\(.*shell\s*=\s*True", "type": "Command Injection", "severity": "HIGH"},
    {"pattern": r"pickle\.loads?\s*\(", "type": "Insecure Deserialization", "severity": "HIGH"},
    {"pattern": r"os\.system\s*\(", "type": "Command Injection", "severity": "MEDIUM"},
    {"pattern": r"md5\s*\(|MD5\s*\(", "type": "Weak Cryptography", "severity": "MEDIUM"},
    {"pattern": r"password\s*=\s*['\"][^'\"]{1,20}['\"]", "type": "Hardcoded Credential", "severity": "HIGH"},
    {"pattern": r"secret\s*=\s*['\"][^'\"]{1,40}['\"]", "type": "Hardcoded Secret", "severity": "HIGH"},
    {"pattern": r"innerHTML\s*=", "type": "XSS", "severity": "MEDIUM"},
    {"pattern": r"document\.write\s*\(", "type": "XSS", "severity": "MEDIUM"},
]


def hash_code(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()


def check_smart_memory(code: str, db: Session) -> bool:
    """Returns True if the code matches a SmartMemory safe pattern (override to safe)."""
    rules = db.query(SmartMemory).all()
    for rule in rules:
        try:
            if re.search(rule.pattern, code, re.IGNORECASE | re.MULTILINE):
                return True
        except re.error:
            if rule.pattern in code:
                return True
    return False


def scan_chunk(chunk: dict, db: Session) -> dict | None:
    """
    Scan a single code chunk.
    Returns a finding dict or None if no vulnerability detected.
    Pipeline: SmartMemory override -> ML Model -> Baseline Rules
    """
    code = chunk["code"]
    if not code.strip():
        return None

    code_hash = hash_code(code)

    # Step 1: Smart Memory deterministic override
    if check_smart_memory(code, db):
        return None

    # Step 2: Try ML model
    model_data = load_active_model()
    ml_flagged = False
    if model_data:
        vectorizer = model_data["vectorizer"]
        clf = model_data["classifier"]
        threshold = model_data.get("threshold", 0.5)
        vec = vectorizer.transform([code])
        proba = clf.predict_proba(vec)[0]
        vuln_idx = list(clf.classes_).index("valid_vulnerability") if "valid_vulnerability" in clf.classes_ else -1
        if vuln_idx >= 0 and proba[vuln_idx] >= threshold:
            ml_flagged = True
            return {
                "code_hash": code_hash,
                "code_snippet": code[:1000],
                "file_path": chunk.get("file_path"),
                "vulnerability_type": "ML-Detected",
                "prediction": "vulnerable",
                "confidence_score": round(float(proba[vuln_idx]), 3),
            }
        # ML said safe — still run baseline rules as safety net

    # Step 3: Baseline regex rules (always runs as safety net / when model untrained)
    for rule in BASELINE_PATTERNS:
        if re.search(rule["pattern"], code, re.IGNORECASE | re.MULTILINE):
            return {
                "code_hash": code_hash,
                "code_snippet": code[:1000],
                "file_path": chunk.get("file_path"),
                "vulnerability_type": rule["type"],
                "prediction": "vulnerable",
                "confidence_score": 0.8,
            }

    return None


def scan_chunks(chunks: list[dict], db: Session) -> list[dict]:
    """Scan all chunks and return findings."""
    findings = []
    for chunk in chunks:
        result = scan_chunk(chunk, db)
        if result:
            findings.append(result)
    return findings
