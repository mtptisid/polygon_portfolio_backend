"""
Standalone Gmail SMTP test script.

Run this BEFORE wiring SMTP into the app, just to confirm your
Gmail App Password + "send as" alias actually work.

Usage:
    export SMTP_USER="youraddress@gmail.com"        # the Gmail account the App Password belongs to
    export SMTP_PASSWORD="xxxx xxxx xxxx xxxx"       # the 16-char App Password (spaces optional)
    export SMTP_FROM_EMAIL="me@siddharamayya.in"     # optional, must be a verified "Send mail as" alias
    export TEST_TO_EMAIL="you@example.com"           # where to send the test email
    python backend/app/scripts/test_smtp_gmail.py
"""

import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", SMTP_USER)
TO_EMAIL = os.environ.get("TEST_TO_EMAIL", SMTP_USER)


def main():
    if not SMTP_USER or not SMTP_PASSWORD:
        raise SystemExit("Set SMTP_USER and SMTP_PASSWORD env vars first (see file header).")

    msg = MIMEMultipart("alternative")
    msg["Subject"] = "SMTP test email"
    msg["From"] = FROM_EMAIL
    msg["To"] = TO_EMAIL
    msg.attach(MIMEText("This is a plain-text test.", "plain"))
    msg.attach(MIMEText("<p>This is an <b>HTML</b> test.</p>", "html"))

    print(f"Connecting to {SMTP_HOST}:{SMTP_PORT} as {SMTP_USER} ...")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
        server.set_debuglevel(1)  # print the full SMTP conversation
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())

    print(f"\nSent OK: from={FROM_EMAIL} to={TO_EMAIL}")


if __name__ == "__main__":
    main()
