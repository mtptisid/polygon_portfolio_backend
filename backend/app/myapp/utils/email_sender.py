"""
SMTP email sender (Gmail SMTP relay, sending from the siddharamayya.in domain).

Required environment variables (set in Cloud Run):
    SMTP_HOST        default smtp.gmail.com
    SMTP_PORT        default 587
    SMTP_USER        Gmail login address (the mailbox the App Password belongs to)
    SMTP_PASSWORD    Gmail App Password (not the account password)
Optional:
    SMTP_FROM_EMAIL  "from" address, e.g. me@siddharamayya.in (must be an alias/send-as on SMTP_USER)
    SMTP_FROM_NAME   default "from" display name
"""

import smtplib
import os
import logging
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from typing import Optional, List, TypedDict

logger = logging.getLogger(__name__)

SMTP_HOST = os.environ.get("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_USER = os.environ.get("SMTP_USER")
SMTP_PASSWORD = os.environ.get("SMTP_PASSWORD")
DEFAULT_FROM_EMAIL = os.environ.get("SMTP_FROM_EMAIL", "me@siddharamayya.in")
DEFAULT_FROM_NAME = os.environ.get("SMTP_FROM_NAME", "Siddharamayya Mathapati")


class EmailAttachment(TypedDict):
    filename: str
    content: bytes
    mimetype: str


class EmailSendError(Exception):
    """Raised when the SMTP server fails to accept/send the message."""


def send_email(
    to_email: str,
    subject: str,
    html_content: str,
    plain_content: Optional[str] = None,
    from_email: Optional[str] = None,
    from_name: Optional[str] = None,
    cc: Optional[str] = None,
    bcc: Optional[str] = None,
    attachments: Optional[List[EmailAttachment]] = None,
) -> None:
    """Send a single email via Gmail SMTP. Raises EmailSendError on failure."""
    if not SMTP_USER or not SMTP_PASSWORD:
        raise EmailSendError("SMTP_USER / SMTP_PASSWORD not configured")

    sender_email = from_email or DEFAULT_FROM_EMAIL
    sender_name = from_name or DEFAULT_FROM_NAME

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = f"{sender_name} <{sender_email}>"
    msg["To"] = to_email

    recipients = [to_email]
    if cc:
        msg["Cc"] = cc
        recipients.append(cc)
    if bcc:
        recipients.append(bcc)

    body = MIMEMultipart("alternative")
    if plain_content:
        body.attach(MIMEText(plain_content, "plain"))
    body.attach(MIMEText(html_content, "html"))
    msg.attach(body)

    for att in attachments or []:
        part = MIMEApplication(att["content"], Name=att["filename"])
        part["Content-Disposition"] = f'attachment; filename="{att["filename"]}"'
        msg.attach(part)

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
            server.starttls()
            server.login(SMTP_USER, SMTP_PASSWORD)
            # sender_email must be a verified "Send mail as" alias on the SMTP_USER Gmail account
            server.sendmail(sender_email, recipients, msg.as_string())
    except smtplib.SMTPException as e:
        logger.error(f"SMTP error sending to {to_email}: {e}")
        raise EmailSendError(str(e)) from e
    except OSError as e:
        logger.error(f"SMTP connection error sending to {to_email}: {e}")
        raise EmailSendError(str(e)) from e
