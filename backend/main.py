import sys
import shutil
import tempfile
from pathlib import Path

import uvicorn
from fastapi import (
    FastAPI,
    HTTPException,
    UploadFile,
    File,
    Form,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.inference import predict_execution
from backend.alerts import (
    manager,
    history,
    build_event,
    apply_action,
    notify_email,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
INFERENCE_OUTPUT_DIR = PROJECT_ROOT / "inference_output"

EXPECTED_FILES = {
    "ata_read.csv",
    "ata_write.csv",
    "mem_exec.csv",
    "mem_read.csv",
    "mem_readwrite.csv",
    "mem_write.csv",
}


app = FastAPI(
    title="RanSMAP API",
    description="Ransomware detection using LSTM models",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000","http://localhost:5173"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Basic routes ─────────────────────────────────────────────────────────────
@app.get("/")
def home():
    return {
        "message": "RanSMAP API is running",
        "status": "success",
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


# ── Prediction ───────────────────────────────────────────────────────────────
@app.post("/predict")
async def predict(
    ata_read: UploadFile = File(..., description="Upload ata_read.csv"),
    ata_write: UploadFile = File(..., description="Upload ata_write.csv"),
    mem_exec: UploadFile = File(..., description="Upload mem_exec.csv"),
    mem_read: UploadFile = File(..., description="Upload mem_read.csv"),
    mem_readwrite: UploadFile = File(..., description="Upload mem_readwrite.csv"),
    mem_write: UploadFile = File(..., description="Upload mem_write.csv"),
    # NEW: optional name shown on the dashboard (the replayer can send it)
    execution_name: str = Form(None),
):
    files = [
        ata_read,
        ata_write,
        mem_exec,
        mem_read,
        mem_readwrite,
        mem_write,
    ]

    received_files = {
        Path(file.filename).name
        for file in files
        if file.filename
    }

    if received_files != EXPECTED_FILES:
        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid CSV files.",
                "expected_files": sorted(EXPECTED_FILES),
                "received_files": sorted(received_files),
            },
        )

    execution_path = Path(tempfile.mkdtemp(prefix="ransmap_"))

    try:
        for file in files:
            filename = Path(file.filename).name
            destination = execution_path / filename

            with open(destination, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

        INFERENCE_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        results = predict_execution(
            execution_path=str(execution_path),
            out_root=str(INFERENCE_OUTPUT_DIR),
            artifacts_dir=str(ARTIFACTS_DIR),
        )

        predictions = {}

        for fraction, result in results.items():
            percentage = f"{int(fraction * 100)}%"

            predictions[percentage] = {
                "probability": result["probability"],
                "prediction": result["prediction"],
                "label": (
                    "Ransomware"
                    if result["prediction"] == 1
                    else "Benign"
                ),
            }

        # NEW: build the dashboard event, store it, broadcast it
        event = build_event(predictions, execution_name)
        history.append(event)
        await manager.broadcast(event)

        if event["type"] == "alert":
            notify_email(event)  # placeholder for Task 9 (Ayett)

        return {
            "status": "success",
            "predictions": predictions,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        for file in files:
            await file.close()

        shutil.rmtree(execution_path, ignore_errors=True)


# ── NEW: WebSocket endpoint (live scores for the dashboard) ──────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            # Keep the connection open; we don't expect messages from the client
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


# ── NEW: History + Isolate / Dismiss actions ─────────────────────────────────
class ActionBody(BaseModel):
    action: str  # "isolate" or "dismiss"


@app.get("/history")
def get_history():
    return history


@app.post("/alerts/{event_id}/action")
async def alert_action(event_id: str, body: ActionBody):
    if body.action not in ("isolate", "dismiss"):
        raise HTTPException(status_code=400, detail="action must be isolate or dismiss")

    event = apply_action(event_id, body.action)

    if event is None:
        raise HTTPException(status_code=404, detail="Event not found")

    await manager.broadcast({**event, "type": "alert_update"})
    return event


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )