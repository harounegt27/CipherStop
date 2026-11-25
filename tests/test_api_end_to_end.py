"""End-to-end test of the FastAPI app with a stubbed model (no TensorFlow needed):
POST /predict -> WebSocket broadcast -> /history -> e-mail through a local SMTP server.
Run: pytest tests/test_api_end_to_end.py -q   (needs httpx, aiosmtpd)"""
import importlib
import io
import json
import sys
import time
import types

import pytest

pytest.importorskip("httpx")
aiosmtpd = pytest.importorskip("aiosmtpd.controller")
from fastapi.testclient import TestClient  # noqa: E402

FILES = ["ata_read", "ata_write", "mem_exec", "mem_read", "mem_readwrite", "mem_write"]


def make_client(monkeypatch, tmp_path, probs, smtp_port):
    (tmp_path / "alert_config.json").write_text(json.dumps({"threshold": 0.4}))
    monkeypatch.setenv("ALERT_CONFIG", str(tmp_path / "alert_config.json"))
    monkeypatch.delenv("ALERT_THRESHOLD", raising=False)
    for k, v in dict(SMTP_HOST="127.0.0.1", SMTP_PORT=str(smtp_port), SMTP_USER="", SMTP_FROM="cs@test.local",
                     ALERT_EMAIL="analyst@test.local").items():
        monkeypatch.setenv(k, v)
    stub = types.ModuleType("backend.inference")
    stub.predict_execution = lambda **kw: {f: {"probability": p, "prediction": int(p >= 0.5)}
                                           for f, p in zip([0.10, 0.25, 0.50, 0.75], probs)}
    monkeypatch.setitem(sys.modules, "backend.inference", stub)
    import backend.alerts as alerts
    importlib.reload(alerts)
    import backend.main as main
    importlib.reload(main)
    return TestClient(main.app)


def post(client, name):
    files = [(f, (f"{f}.csv", io.BytesIO(b"1,2,3\n"), "text/csv")) for f in FILES]
    return client.post("/predict", files=files, data={"execution_name": name})


def test_alert_flow(monkeypatch, tmp_path):
    got = []

    class H:
        async def handle_DATA(self, server, session, envelope):
            got.append(envelope); return "250 OK"

    ctl = aiosmtpd.Controller(H(), hostname="127.0.0.1", port=8027)
    ctl.start()
    try:
        c = make_client(monkeypatch, tmp_path, [0.1, 0.45, 0.9, 0.9], 8027)
        assert c.get("/config").json() == {"threshold": 0.4}
        with c.websocket_connect("/ws") as ws:
            r = post(c, "Ryuk-demo")
            assert r.status_code == 200
            preds = r.json()["predictions"]
            # /predict labels use the SAME threshold (0.4), not 0.5
            assert preds["25%"]["label"] == "Ransomware" and preds["10%"]["label"] == "Benign"
            msg = ws.receive_json()
            assert msg["type"] == "alert" and msg["detection_point"] == "25%" and msg["threshold"] == 0.4
        for _ in range(30):
            if got:
                break
            time.sleep(0.1)
        assert len(got) == 1 and b"Ryuk-demo" in got[0].content
        hist = c.get("/history").json()
        assert hist[0]["status"] == "pending"
        for _ in range(20):                              # e-mail runs in a thread: status appears shortly after
            if c.get("/history").json()[0].get("email_status"):
                break
            time.sleep(0.1)
        assert c.get("/history").json()[0]["email_status"] == "sent"
        ev = c.post(f"/alerts/{hist[0]['id']}/action", json={"action": "isolate"}).json()
        assert ev["status"] == "isolated"
    finally:
        ctl.stop()


def test_benign_sends_no_email_and_smtp_down_does_not_break_predict(monkeypatch, tmp_path):
    c = make_client(monkeypatch, tmp_path, [0.01, 0.02, 0.01, 0.01], 1)      # nothing listens on port 1
    assert post(c, "Firefox-demo").json()["status"] == "success"
    assert c.get("/history").json()[0]["type"] == "score_update"
    c2 = make_client(monkeypatch, tmp_path, [0.9, 0.9, 0.9, 0.9], 1)         # alert, but SMTP unreachable
    assert post(c2, "Ryuk-demo").status_code == 200                           # prediction still succeeds
