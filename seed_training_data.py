"""
Seed script — populates the feedback table with labelled training examples
so the ML model can be trained immediately.

Run from the SAST directory:
    python seed_training_data.py
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backend.database import SessionLocal, engine, Base
from backend.models import Feedback, FeedbackLabel
import hashlib
from datetime import datetime, timezone

Base.metadata.create_all(bind=engine)

SAMPLES = [
    # ── VALID VULNERABILITIES ──────────────────────────────────────────
    ("os.system(user_input)",                            "Command Injection",           FeedbackLabel.VALID_VULNERABILITY),
    ('db.execute("SELECT * FROM users WHERE id = " + x)',"SQL Injection",              FeedbackLabel.VALID_VULNERABILITY),
    ('password = "admin123"',                            "Hardcoded Credential",        FeedbackLabel.VALID_VULNERABILITY),
    ("pickle.loads(data)",                               "Insecure Deserialization",    FeedbackLabel.VALID_VULNERABILITY),
    ("eval(user_expression)",                            "Code Injection",              FeedbackLabel.VALID_VULNERABILITY),
    ('hashlib.md5(pwd.encode()).hexdigest()',             "Weak Cryptography",           FeedbackLabel.VALID_VULNERABILITY),
    ('document.getElementById("x").innerHTML = input',  "XSS",                         FeedbackLabel.VALID_VULNERABILITY),
    ('subprocess.call(cmd, shell=True)',                 "Command Injection",           FeedbackLabel.VALID_VULNERABILITY),
    ('secret = "my_super_secret_key"',                  "Hardcoded Secret",            FeedbackLabel.VALID_VULNERABILITY),
    ('db.execute(f"SELECT * FROM orders WHERE id={id}")',"SQL Injection",              FeedbackLabel.VALID_VULNERABILITY),
    ('document.write(user_content)',                     "XSS",                         FeedbackLabel.VALID_VULNERABILITY),
    ('api_key = "sk-prod-12345abcde"',                  "Hardcoded Secret",            FeedbackLabel.VALID_VULNERABILITY),

    # ── FALSE POSITIVES ────────────────────────────────────────────────
    ("import os\nimport sys",                            "Safe Import",                 FeedbackLabel.FALSE_POSITIVE),
    ("result = subprocess.run(['ls', '-la'])",           "Safe Subprocess",             FeedbackLabel.FALSE_POSITIVE),
    ("sha256 = hashlib.sha256(data).hexdigest()",       "Safe Crypto",                 FeedbackLabel.FALSE_POSITIVE),
    ("password = getpass.getpass('Enter password: ')",  "Safe Password Input",         FeedbackLabel.FALSE_POSITIVE),
    ("db.execute('SELECT * FROM users WHERE id = ?', (user_id,))", "Safe Parameterized Query", FeedbackLabel.FALSE_POSITIVE),
    ("os.environ.get('SECRET_KEY', '')",                "Safe Env Lookup",             FeedbackLabel.FALSE_POSITIVE),
    ("x = eval('2 + 2')  # safe: literal only",         "Safe Eval",                   FeedbackLabel.FALSE_POSITIVE),
    ("# This is just a comment: password = test",       "Comment Not Code",            FeedbackLabel.FALSE_POSITIVE),
    ("content = template.render(user_input=escape(val))","Safe Template Render",       FeedbackLabel.FALSE_POSITIVE),
    ("data = json.loads(request.body)",                 "Safe JSON Deserialization",   FeedbackLabel.FALSE_POSITIVE),
]

def seed():
    db = SessionLocal()
    added = 0
    skipped = 0
    for snippet, vuln_type, label in SAMPLES:
        code_hash = hashlib.sha256(snippet.encode()).hexdigest()
        existing = db.query(Feedback).filter(Feedback.code_hash == code_hash).first()
        if existing:
            skipped += 1
            continue
        fb = Feedback(
            code_hash=code_hash,
            code_snippet=snippet,
            file_path="seed_data.py",
            vulnerability_type=vuln_type,
            prediction="vulnerable" if label == FeedbackLabel.VALID_VULNERABILITY else "safe",
            confidence_score=0.85 if label == FeedbackLabel.VALID_VULNERABILITY else 0.3,
            developer_label=label,
            timestamp=datetime.now(timezone.utc),
            is_used_for_training=False,
        )
        db.add(fb)
        added += 1

    db.commit()
    db.close()
    print(f"[OK] Seeded {added} samples ({skipped} already existed).")
    print(f"   {sum(1 for _, _, l in SAMPLES if l == FeedbackLabel.VALID_VULNERABILITY)} valid vulnerabilities")
    print(f"   {sum(1 for _, _, l in SAMPLES if l == FeedbackLabel.FALSE_POSITIVE)} false positives")
    print("\nRun next: python -m backend.ml.retraining")

if __name__ == "__main__":
    seed()
