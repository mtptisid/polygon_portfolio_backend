from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
import smtplib
from email.mime.text import MIMEText
import os
from fastapi.middleware.cors import CORS

router = APIRouter(
    prefix="/api/email",
    tags=['Contact']
)

# Enable CORS for GitHub Pages
CORS(router, allow_origins=["https://mtptisid.github.io"], allow_methods=["POST"], allow_headers=["*"])

# Pydantic model for form data validation
class ContactForm(BaseModel):
    name: str = "Unknown"
    email: EmailStr
    subject: str = "Contact Form Submission"
    message: str

@router.post("/contact")
async def contact(form: ContactForm):
    try:
        # Extract form data
        name = form.name
        email = form.email
        subject = form.subject
        message = form.message

        # Validate required fields
        if not email or not message:
            raise HTTPException(status_code=400, detail="Email and message are required")

        # Set up SMTP
        smtp_server = "smtp.gmail.com"
        smtp_port = 587
        sender_email = "msidrm455@gmail.com"
        password = os.environ.get("GMAIL_APP_PASSWORD")

        if not password:
            raise HTTPException(status_code=500, detail="GMAIL_APP_PASSWORD not configured")

        # Email to you
        msg_to_you = MIMEText(
            f"Name: {name}\n"
            f"Email: {email}\n"
            f"Subject: {subject}\n"
            f"Message:\n{message}"
        )
        msg_to_you["Subject"] = f"New Contact Form Submission: {subject}"
        msg_to_you["From"] = sender_email
        msg_to_you["To"] = sender_email

        # Acknowledgment email to user
        ack_subject = subject if subject != "Contact Form Submission" else "Thank You for Contacting Me"
        ack_message = (
            f"Dear {name},\n\n"
            f"Thank you for reaching out! I have received your message:\n\n"
            f"Subject: {subject}\n"
            f"Message:\n{message}\n\n"
            f"I will get back to you soon.\n\n"
            f"Best regards,\n"
            f"Siddharamayya M"
        ) if subject != "Contact Form Submission" else (
            f"Dear {name},\n\n"
            f"Thank you for contacting me through my portfolio. I appreciate your interest and will respond to your inquiry soon.\n\n"
            f"Best regards,\n"
            f"Siddharamayya M"
        )
        msg_to_user = MIMEText(ack_message)
        msg_to_user["Subject"] = ack_subject
        msg_to_user["From"] = sender_email
        msg_to_user["To"] = email

        # Send emails
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, password)
            server.sendmail(sender_email, [sender_email], msg_to_you.as_string())
            server.sendmail(sender_email, [email], msg_to_user.as_string())

        return JSONResponse(content={"message": "Emails sent successfully"}, status_code=200)

    except smtplib.SMTPAuthenticationError:
        raise HTTPException(status_code=500, detail="Failed to authenticate with Gmail SMTP")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error sending emails: {str(e)}")
