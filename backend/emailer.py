"""SMTP e-mail notification for CipherStop alerts (standard library only).

Environment variables (all read at send time, so a restart is enough to change them):
    SMTP_HOST       e.g. smtp.gmail.com      (empty -> e-mail disabled, nothing is sent)
    SMTP_PORT       587 (STARTTLS, default) or 465 (implicit SSL)
    SMTP_USER       login (usually the sender address)
    SMTP_PASSWORD   password / app password
    SMTP_FROM       sender address (default: SMTP_USER)
    ALERT_EMAIL     recipient(s), comma separated
    DASHBOARD_URL   base URL of the dashboard (default http://localhost:5173)
"""
import logging
import os
import smtplib
import ssl
from email.message import EmailMessage

log = logging.getLogger("cipherstop.email")


def smtp_settings() -> dict | None:
    host = os.getenv("SMTP_HOST", "").strip()
    to = [x.strip() for x in os.getenv("ALERT_EMAIL", "").split(",") if x.strip()]
    if not host or not to:
        return None
    user = os.getenv("SMTP_USER", "").strip()
    return dict(host=host, port=int(os.getenv("SMTP_PORT", "587") or 587), user=user,
                password=os.getenv("SMTP_PASSWORD", ""), sender=os.getenv("SMTP_FROM", "").strip() or user,
                to=to, dashboard=os.getenv("DASHBOARD_URL", "http://localhost:5173").rstrip("/"))


def build_message(event: dict, cfg: dict) -> EmailMessage:
    name = event.get("execution") or "unknown execution"
    point = event.get("detection_point") or "?"
    scores = event.get("scores", {})
    lines = [f"  {pct:>4}  score = {s:.3f}" + ("   <-- ALERT" if pct == point else "")
             for pct, s in scores.items()]
    body = (
        f"CipherStop detected ransomware-like behaviour.\n\n"
        f"Execution       : {name}\n"
        f"Detected at     : {point} of the observed behaviour\n"
        f"Max score       : {event.get('max_score', 0):.3f} (alert threshold {event.get('threshold')})\n"
        f"Time (UTC)      : {event.get('timestamp')}\n\n"
        f"Score per observation point:\n" + "\n".join(lines) + "\n\n"
        f"Review and act (Isolate / Mark as false positive):\n"
        f"  {cfg['dashboard']}/alert/{event.get('id')}\n"
    )
    msg = EmailMessage()
    msg["Subject"] = f"[CipherStop] RANSOMWARE ALERT - {name} (detected at {point})"
    msg["From"] = cfg["sender"]
    msg["To"] = ", ".join(cfg["to"])
    msg.set_content(body)
    return msg


def send_alert_email(event: dict, timeout: float = 10.0) -> str:
    """Send the alert e-mail. Returns 'sent', 'disabled' or 'failed: <reason>'. Never raises."""
    cfg = smtp_settings()
    if cfg is None:
        log.warning("e-mail disabled (SMTP_HOST / ALERT_EMAIL not set)")
        return "disabled"
    try:
        msg = build_message(event, cfg)
        if cfg["port"] == 465:
            with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=timeout,
                                  context=ssl.create_default_context()) as s:
                if cfg["user"]:
                    s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        else:
            with smtplib.SMTP(cfg["host"], cfg["port"], timeout=timeout) as s:
                s.ehlo()
                if s.has_extn("starttls"):
                    s.starttls(context=ssl.create_default_context())
                    s.ehlo()
                if cfg["user"]:
                    s.login(cfg["user"], cfg["password"])
                s.send_message(msg)
        log.info("alert e-mail sent to %s", cfg["to"])
        return "sent"
    except Exception as e:  # a mail problem must never break detection
        log.error("alert e-mail failed: %s", e)
        return f"failed: {str(e)[:150]}"
