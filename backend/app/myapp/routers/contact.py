from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import os
import logging


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()


# Pydantic model for form data validation
class ContactForm(BaseModel):
    name: str = "Unknown"
    email: EmailStr
    subject: str = "Contact Form Submission"
    message: str
    honeypot: str | None = None  # Optional honeypot field for spam prevention

@router.post("/contact")
async def contact(form: ContactForm, request: Request):
    try:
        # Check honeypot field for spam
        if form.honeypot:
            logger.warning("Spam detected: Honeypot field filled")
            raise HTTPException(status_code=400, detail="Spam detected")

        # Extract and sanitize form data
        name = form.name.strip()
        email = form.email
        subject = form.subject.strip()
        message = form.message.strip()

        # Validate required fields
        if not email or not message:
            logger.warning("Validation failed: Email or message missing")
            raise HTTPException(status_code=400, detail="Email and message are required")

        # Initialize SendGrid client
        api_key = os.environ.get("SENDGRID_API_KEY")
        if not api_key:
            logger.error("SENDGRID_API_KEY not configured")
            raise HTTPException(status_code=500, detail="Server configuration error: SENDGRID_API_KEY missing")

        sg = sendgrid.SendGridAPIClient(api_key=api_key)

        # Email to you
        mail_to_you = Mail(
            from_email=Email("msidrm455@siddharamayya.in", "Siddharamayya Portfolio"),
            to_emails=To("msidrm455@gmail.com"),
            subject=f"New Contact Form Submission: {subject}",
            plain_text_content=(
                f"Name: {name}\n"
                f"Email: {email}\n"
                f"Subject: {subject}\n"
                f"Message:\n{message}"
            )
        )

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
        mail_to_user = Mail(
            from_email=Email("siddharamayya@siddharamayya.in", "Siddharamayya Mathapati"),
            to_emails=To(email),
            subject=ack_subject,
            plain_text_content=ack_message
        )

        # Send emails
        logger.info(f"Sending emails to {email} and msidrm455@gmail.com")
        response_to_you = sg.send(mail_to_you)
        response_to_user = sg.send(mail_to_user)

        if response_to_you.status_code != 202 or response_to_user.status_code != 202:
            logger.error(f"SendGrid failed: to_you={response_to_you.status_code}, to_user={response_to_user.status_code}")
            raise HTTPException(status_code=500, detail="Failed to send one or more emails")

        logger.info("Emails sent successfully")
        return JSONResponse(content={"message": "Emails sent successfully"}, status_code=200)

    except sendgrid.SendGridException as sg_error:
        logger.error(f"SendGrid error: {str(sg_error)}")
        raise HTTPException(status_code=500, detail=f"SendGrid error: {str(sg_error)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending emails: {str(e)}")
