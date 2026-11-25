"""Tests for the alert threshold loading and the SMTP notification.
Run from the repo root:  pytest tests/test_alerts_email.py -q
(needs `pip install aiosmtpd` for the end-to-end SMTP test; it is skipped otherwise)"""
import importlib
import json
import time

import pytest


def fresh_alerts(monkeypatch, tmp_path, cfg=None, env=None):
    monkeypatch.delenv("ALERT_THRESHOLD", raising=False)
    path = tmp_path / "alert_config.json"
    if cfg is not None:
        path.write_text(json.dumps(cfg))
    monkeypatch.setenv("ALERT_CONFIG", str(path))
    for k, v in (env or {}).items():
        monkeypatch.setenv(k, v)
    import backend.alerts as alerts
    return importlib.reload(alerts)


def preds(p10, p25, p50, p75):
    return {k: {"probability": v} for k, v in zip(["10%", "25%", "50%", "75%"], [p10, p25, p50, p75])}


def test_threshold_default_when_no_config(monkeypatch, tmp_path):
    assert fresh_alerts(monkeypatch, tmp_path).ALERT_THRESHOLD == 0.85


def test_threshold_from_config_file(monkeypatch, tmp_path):
    assert fresh_alerts(monkeypatch, tmp_path, cfg={"threshold": 0.37}).ALERT_THRESHOLD == 0.37


def test_env_overrides_config(monkeypatch, tmp_path):
    a = fresh_alerts(monkeypatch, tmp_path, cfg={"threshold": 0.37}, env={"ALERT_THRESHOLD": "0.6"})
    assert a.ALERT_THRESHOLD == 0.6


def test_broken_config_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("ALERT_CONFIG", str(tmp_path / "x.json"))
    (tmp_path / "x.json").write_text("{not json")
    monkeypatch.delenv("ALERT_THRESHOLD", raising=False)
    import backend.alerts as alerts
    assert importlib.reload(alerts).ALERT_THRESHOLD == 0.85


def test_detection_point_is_first_cut_above_threshold(monkeypatch, tmp_path):
    a = fresh_alerts(monkeypatch, tmp_path, cfg={"threshold": 0.4})
    ev = a.build_event(preds(0.1, 0.45, 0.9, 0.2), "x")
    assert ev["type"] == "alert" and ev["detection_point"] == "25%" and ev["threshold"] == 0.4
    ev = a.build_event(preds(0.1, 0.4, 0.3, 0.2), "x")          # exactly the threshold -> not an alert (strict >)
    assert ev["type"] == "score_update" and ev["detection_point"] is None


def test_notify_disabled_without_smtp(monkeypatch, tmp_path):
    a = fresh_alerts(monkeypatch, tmp_path, env={"SMTP_HOST": "", "ALERT_EMAIL": ""})
    ev = {"id": "1", "scores": {}, "max_score": 0.9}
    assert a.notify_email(ev) == "disabled" and ev["email_status"] == "disabled"


def test_notify_never_raises_when_server_down(monkeypatch, tmp_path):
    a = fresh_alerts(monkeypatch, tmp_path, env={"SMTP_HOST": "127.0.0.1", "SMTP_PORT": "1", "ALERT_EMAIL": "a@b.c"})
    ev = {"id": "1", "scores": {"10%": 0.9}, "max_score": 0.9}
    assert a.notify_email(ev).startswith("failed")


def test_email_end_to_end_with_local_smtp(monkeypatch, tmp_path):
    aiosmtpd = pytest.importorskip("aiosmtpd.controller")
    got = []

    class H:
        async def handle_DATA(self, server, session, envelope):
            got.append(envelope)
            return "250 OK"

    ctl = aiosmtpd.Controller(H(), hostname="127.0.0.1", port=8026)
    ctl.start()
    try:
        a = fresh_alerts(monkeypatch, tmp_path, env={
            "SMTP_HOST": "127.0.0.1", "SMTP_PORT": "8026", "SMTP_USER": "", "SMTP_FROM": "cipherstop@test.local",
            "ALERT_EMAIL": "analyst@test.local, boss@test.local", "DASHBOARD_URL": "http://dash:5173"})
        ev = a.build_event(preds(0.2, 0.95, 0.99, 0.99), "Ryuk-2024-demo")
        assert a.notify_email(ev) == "sent"
        for _ in range(20):
            if got:
                break
            time.sleep(0.1)
    finally:
        ctl.stop()
    assert len(got) == 1
    env = got[0]
    assert env.rcpt_tos == ["analyst@test.local", "boss@test.local"]
    text = env.content.decode()
    assert "Ryuk-2024-demo" in text and "25%" in text and f"http://dash:5173/alert/{ev['id']}" in text
    assert "Subject: [CipherStop] RANSOMWARE ALERT" in text
