import os
import uuid
from datetime import datetime, timezone

from fastapi import WebSocket

import json
from pathlib import Path

from backend.emailer import send_alert_email

CONFIG_PATH = Path(os.getenv(
    "ALERT_CONFIG", Path(__file__).resolve().parent.parent / "artifacts" / "alert_config.json"))


def load_threshold() -> float:
    """Priority: env ALERT_THRESHOLD > artifacts/alert_config.json (written by
    src/evaluation/choose_threshold.py, chosen on validation data) > 0.85 (charter default)."""
    env = os.getenv("ALERT_THRESHOLD", "").strip()
    if env:
        return float(env)
    try:
        return float(json.loads(CONFIG_PATH.read_text())["threshold"])
    except Exception:
        return 0.85


ALERT_THRESHOLD = load_threshold()

FRACTIONS = ["10%", "25%", "50%", "75%"]

history: list[dict] = []


class ConnectionManager:
    """Keeps track of connected dashboards and broadcasts messages."""

    def __init__(self):
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict):
        for ws in list(self.active):
            try:
                await ws.send_json(message)
            except Exception:
                self.disconnect(ws)


manager = ConnectionManager()


def is_positive(probability: float) -> bool:
    return float(probability) > ALERT_THRESHOLD


def build_event(predictions: dict, execution_name: str | None = None) -> dict:

    scores = {
        pct: float(predictions[pct]["probability"])
        for pct in FRACTIONS
        if pct in predictions
    }
    max_score = max(scores.values()) if scores else 0.0

    detection_point = None
    for pct in FRACTIONS:
        if pct in scores and is_positive(scores[pct]):
            detection_point = pct
            break

    is_alert = detection_point is not None

    return {
        "type": "alert" if is_alert else "score_update",
        "id": str(uuid.uuid4()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "execution": execution_name,
        "scores": scores,
        "max_score": max_score,
        "threshold": ALERT_THRESHOLD,
        "detection_point": detection_point,
        "status": "pending" if is_alert else "normal",
    }


def find_event(event_id: str) -> dict | None:
    for event in history:
        if event["id"] == event_id:
            return event
    return None


def apply_action(event_id: str, action: str) -> dict | None:
    """Used by POST /alerts/{id}/action. Returns the updated event or None."""
    event = find_event(event_id)
    if event is None:
        return None
    if action == "isolate":
        event["status"] = "isolated"
    elif action == "dismiss":
        event["status"] = "dismissed"
    return event


def notify_email(event: dict) -> str:
    """Send the alert e-mail (blocking: call it from a thread). Stores the outcome on the
    event ('sent' / 'disabled' / 'failed: ...') so History can show it. Never raises."""
    status = send_alert_email(event)
    event["email_status"] = status
    return status
