import os
import uuid
from datetime import datetime, timezone

from fastapi import WebSocket

ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.85"))

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


def build_event(predictions: dict, execution_name: str | None = None) -> dict:

    scores = {
        pct: float(predictions[pct]["probability"])
        for pct in FRACTIONS
        if pct in predictions
    }
    max_score = max(scores.values()) if scores else 0.0

    detection_point = None
    for pct in FRACTIONS:
        if pct in scores and scores[pct] > ALERT_THRESHOLD:
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


def notify_email(event: dict) -> None:
    """Ayettttt hdhy tache mte3ekk"""
    pass