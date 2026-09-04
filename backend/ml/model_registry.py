import os
import json
import joblib
from datetime import datetime, timezone
from pathlib import Path

# Default: …/sast-iq-main/models (anchored to the project root, not the working
# dir). Override with the MODEL_DIR env var to point at a persistent volume.
_DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent.parent / "models"
MODEL_DIR = Path(os.getenv("MODEL_DIR") or _DEFAULT_MODEL_DIR)
REGISTRY_FILE = MODEL_DIR / "registry.json"


def _ensure_dir():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def save_model(vectorizer, classifier, threshold: float, metrics: dict) -> str:
    """
    Saves a new model version and updates the registry.
    Returns the version string.
    """
    _ensure_dir()
    version = datetime.now(timezone.utc).strftime("v%Y%m%d_%H%M%S")
    model_path = MODEL_DIR / f"model_{version}.joblib"

    joblib.dump({"vectorizer": vectorizer, "classifier": classifier, "threshold": threshold}, model_path)

    # Load existing registry
    registry = load_registry()
    registry["versions"].append({
        "version": version,
        "path": str(model_path),
        "metrics": metrics,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    registry["active_version"] = version

    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=2)

    return version


def load_registry() -> dict:
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE) as f:
            return json.load(f)
    return {"active_version": None, "versions": []}


def load_active_model() -> dict | None:
    """Loads and returns the active model dict {vectorizer, classifier, threshold} or None."""
    registry = load_registry()
    active = registry.get("active_version")
    if not active:
        return None
    for v in registry["versions"]:
        if v["version"] == active:
            path = Path(v["path"])
            if not path.is_absolute():
                # Older registries stored paths relative to the working dir;
                # resolve them against MODEL_DIR instead.
                path = MODEL_DIR / path.name
            if path.exists():
                return joblib.load(path)
    return None


def get_all_versions() -> list[dict]:
    return load_registry().get("versions", [])
