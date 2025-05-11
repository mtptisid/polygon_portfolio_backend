```python
from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To
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

# HTML email template for the email sent to you
def get_email_to_you_html(name: str, email: str, subject: str, message: str) -> str:
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New Contact Form Submission</title>
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f7fafc;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 600px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .navbar {
                background-color: #404347;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .navbar a {
                color: #edf2f7;
                text-decoration: none;
                font-size: 18px;
                font-weight: 600;
            }
            .navbar a:hover {
                color: #63b3ed;
            }
            .content {
                padding: 20px;
            }
            .header {
                background-color: #07b1d0;
                color: #ffffff;
                padding: 15px 20px;
                font-size: 20px;
                font-weight: 600;
                text-align: center;
            }
            .field {
                margin-bottom: 15px;
            }
            .field-label {
                font-weight: 600;
                color: #1a202c;
                margin-bottom: 5px;
            }
            .field-value {
                color: #4a5568;
                word-wrap: break-word;
            }
            .footer {
                background-color: #daebdd;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }
            .footer p {
                margin: 5px 0;
                color: #000000;
                font-size: 14px;
            }
            .social-links {
                margin-top: 10px;
            }
            .social-links a {
                margin: 0 10px;
                color: #07b1d0;
                text-decoration: none;
                font-size: 24px;
            }
            .social-links a:hover {
                color: #0cd2e8;
            }
            @media (max-width: 600px) {
                .container {
                    margin: 10px;
                }
                .navbar a {
                    font-size: 16px;
                }
                .header {
                    font-size: 18px;
                }
                .content {
                    padding: 15px;
                }
                .footer {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="navbar">
                <a href="https://siddharamayya.in">Siddharamayya</a>
            </div>
            <div class="header">
                New Contact Form Submission
            </div>
            <div class="content">
                <div class="field">
                    <div class="field-label">Name</div>
                    <div class="field-value">{name}</div>
                </div>
                <div class="field">
                    <div class="field-label">Email</div>
                    <div class="field-value">{email}</div>
                </div>
                <div class="field">
                    <div class="field-label">Subject</div>
                    <div class="field-value">{subject}</div>
                </div>
                <div class="field">
                    <div class="field-label">Message</div>
                    <div class="field-value">{message}</div>
                </div>
            </div>
            <div class="footer">
                <p>Siddharamayya M</p>
                <p>Email: msidrm455@gmail.com</p>
                <p>Phone: +91 97406 71620</p>
                <p>Instagram: its_5iD</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/siddharamayya-mathapati" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
                    <a href="https://medium.com/@msidrm455" title="Medium"><i class="fab fa-medium"></i></a>
                    <a href="https://github.com/mtptisid" title="GitHub"><i class="fab fa-github"></i></a>
                    <a href="https://www.instagram.com/its_5iD" title="Instagram"><i class="fab fa-instagram"></i></a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """.format(name=name, email=email, subject=subject, message=message)

# HTML email template for the acknowledgment email to the user
def get_ack_email_html(name: str, subject: str, message: str) -> str:
    ack_content = (
        f"<p>Dear {name},</p>"
        f"<p>Thank you for reaching out! I have received your message:</p>"
        f"<blockquote style='border-left: 4px solid #07b1d0; padding-left: 15px; margin: 15px 0; color: #4a5568;'>"
        f"{message}"
        f"</blockquote>"
        f"<p>Your inquiry is important to me; I will respond at the earliest opportunity.</p>"
        f"<p>Best regards,<br>Siddharamayya M</p>"
    ) if subject != "Contact Form Submission" else (
        f"<p>Dear {name},</p>"
        f"<p>Thank you for contacting me through my portfolio. I appreciate your interest and will respond to your inquiry soon.</p>"
        f"<p>Best regards,<br>Siddharamayya M</p>"
    )
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        <style>
            body {
                font-family: 'Poppins', sans-serif;
                line-height: 1.6;
                color: #333;
                background-color: #f7fafc;
                margin: 0;
                padding: 0;
            }
            .container {
                max-width: 600px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .navbar {
                background-color: #404347;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }
            .navbar a {
                color: #edf2f7;
                text-decoration: none;
                font-size: 18px;
                font-weight: 600;
            }
            .navbar a:hover {
                color: #63b3ed;
            }
            .content {
                padding: 20px;
            }
            .header {
                background-color: #07b1d0;
                color: #ffffff;
                padding: 15px 20px;
                font-size: 20px;
                font-weight: 600;
                text-align: center;
            }
            .footer {
                background-color: #daebdd;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }
            .footer p {
                margin: 5px 0;
                color: #000000;
                font-size: 14px;
            }
            .social-links {
                margin-top: 10px;
            }
            .social-links a {
                margin: 0 10px;
                color: #07b1d0;
                text-decoration: none;
                font-size: 24px;
            }
            .social-links a:hover {
                color: #0cd2e8;
            }
            @media (max-width: 600px) {
                .container {
                    margin: 10px;
                }
                .navbar a {
                    font-size: 16px;
                }
                .header {
                    font-size: 18px;
                }
                .content {
                    padding: 15px;
                }
                .footer {
                    padding: 15px;
                }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="navbar">
                <a href="https://siddharamayya.in">Siddharamayya</a>
            </div>
            <div class="header">
                {subject}
            </div>
            <div class="content">
                {ack_content}
            </div>
            <div class="footer">
                <p>Siddharamayya M</p>
                <p>Email: msidrm455@gmail.com</p>
                <p>Phone: +91 97406 71620</p>
                <p>Instagram: its_5iD</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/siddharamayya-mathapati" title="LinkedIn"><i class="fab fa-linkedin"></i></a>
                    <a href="https://medium.com/@msidrm455" title="Medium"><i class="fab fa-medium"></i></a>
                    <a href="https://github.com/mtptisid" title="GitHub"><i class="fab fa-github"></i></a>
                    <a href="https://www.instagram.com/its_5iD" title="Instagram"><i class="fab fa-instagram"></i></a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """.format(subject=subject, ack_content=ack_content)

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
            from_email=Email("siddharamayya@siddharamayya.in", "Siddharamayya Mathapati"),
            to_emails=To("msidrm455@gmail.com"),
            subject=f"New Contact Form Submission: {subject}",
            html_content=get_email_to_you_html(name, email, subject, message)
        )

        # Acknowledgment email to user
        ack_subject = subject if subject != "Contact Form Submission" else "Thank You for Contacting Me"
        mail_to_user = Mail(
            from_email=Email("siddharamayya@siddharamayya.in", "Siddharamayya Mathapati"),
            to_emails=To(email),
            subject=ack_subject,
            html_content=get_ack_email_html(name, ack_subject, message)
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
```
