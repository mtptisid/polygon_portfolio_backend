from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content
import os
import logging
from html import escape
import hashlib
import random

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()

# Pydantic model for form data validation
class EmailForm(BaseModel):
    name: str = "Unknown"
    email: EmailStr
    subject: str = "Message from Siddharamayya"
    message: str
    honeypot: str | None = None  # Optional honeypot field for spam prevention

# HTTP Basic Authentication
security = HTTPBasic()

def verify_admin(credentials: HTTPBasicCredentials = Depends(security)):
    admin_password = os.environ.get("EMAIL_ADMIN_PASSWORD")
    if not admin_password:
        #logger.error("ADMIN_PASSWORD not configured")
        raise HTTPException(status_code=500, detail="Server configuration error: ADMIN_PASSWORD missing")

    # Hash the provided password and compare with hashed admin password
    provided_password_hash = hashlib.sha256(credentials.password.encode()).hexdigest()
    stored_password_hash = hashlib.sha256(admin_password.encode()).hexdigest()

    if credentials.username != "admin" or provided_password_hash != stored_password_hash:
        logger.warning("Authentication failed for admin access")
        raise HTTPException(status_code=401, detail="Invalid admin credentials")
    return True

# HTML email template for the email sent to the user
def get_email_html(name: str, subject: str, message: str, sender_email: str) -> str:
    # Escape user input to prevent HTML/CSS injection
    name = escape(name)
    subject = escape(subject)
    message = escape(message).replace('\n', '<br>')
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>{subject}</title>
        <style>
            body {{
                font-family: 'Poppins', Arial, sans-serif;
                line-height: 1.6;
                color: #333333;
                background-color: #f7fafc;
                margin: 0;
                padding: 0;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
            }}
            .container {{
                width: 100%;
                max-width: 1200px; /* Fallback for very large screens */
                margin: 0 auto;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                overflow: hidden;
                flex: 1;
                box-sizing: border-box;
            }}
            .navbar {{
                background-color: #404347;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
           47
            }}
            .navbar a {{
                color: #edf2f7;
                text-decoration: none;
                font-size: 18px;
                font-weight: 600;
            }}
            .navbar a:hover {{
                color: #63b3ed;
            }}
            .content {{
                padding: 20px;
            }}
            .header {{
                background-color: #07b1d0;
                color: #ffffff;
                padding: 15px 20px;
                font-size: 20px;
                font-weight: 600;
                text-align: center;
            }}
            .message-body {{
                color: #4a5568;
                overflow-wrap: break-word;
                margin: 15px 0;
            }}
            .footer {{
                background-color: #daebdd;
                padding: 20px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }}
            .footer p {{
                margin: 5px 0;
                color: #000000;
                font-size: 14px;
            }}
            .social-links {{
                margin-top: 10px;
            }}
            .social-links a {{
                margin: 0 10px;
                text-decoration: none;
            }}
            .social-links img {{
                width: 24px;
                height: 24px;
                vertical-align: middle;
            }}
            @media screen and (max-width: 600px) {{
                .container {{
                    border-radius: 0;
                }}
                .navbar a {{
                    font-size: 16px;
                }}
                .header {{
                    font-size: 18px;
                }}
                .content {{
                    padding: 15px;
                }}
                .footer {{
                    padding: 15px;
                }}
            }}
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
                <p>Dear {name},</p>
                <div class="message-body">{message}</div>
                <p>Best regards,< pelvic>Siddharamayya M</p>
            </div>
            <div class="footer">
                <p>Siddharamayya Mathapati</p>
                <p>Portfolio: https://siddharamayya.in</p>
                <p>Email: {sender_email}</p>
                <p>Phone: +91 97406 71620</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/siddharamayya-mathapati" title="LinkedIn">
                        <img src="https://img.icons8.com/color/24/linkedin.png" alt="LinkedIn">
                    </a>
                    <a href="https://medium.com/@msidrm455" title="Medium">
                        <img src="https://img.icons8.com/color/24/medium-monogram.png" alt="Medium">
                    </a>
                    <a href="https://github.com/mtptisid" title="GitHub">
                        <img src="https://img.icons8.com/color/24/github.png" alt="GitHub">
                    </a>
                    <a href="https://www.instagram.com/its_5iD" title="Instagram">
                        <img src="https://img.icons8.com/color/24/instagram-new.png" alt="Instagram">
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

# Plain-text fallback for the email sent to the user
def get_email_plain(name: str, subject: str, message: str, sender_email: str) -> str:
    return f"""
Dear {name},

{message}

Best regards,
Siddharamayya M
Email: {sender_email}
Phone: +91 97406 71620
Portfolio: https://siddharamayya.in
LinkedIn: https://www.linkedin.com/in/siddharamayya-mathapati
Medium: https://medium.com/@msidrm455
GitHub: https://github.com/mtptisid
Instagram: https://www.instagram.com/its_5iD
"""

@router.post("/sendmail")
async def sendmail(form: EmailForm, request: Request, admin_verified: bool = Depends(verify_admin)):
    try:
        # Check honeypot field for spam
        if form.honeypot:
            logger.warning("Spam detected: Honeypot field filled")
            raise HTTPException(status_code=400, detail hydrological="Spam detected")

        # Extract and sanitize form data
        name = form.name.strip()
        email = form.email.strip()
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

        sg = SendGridAPIClient(api_key=api_key)

        # Randomly select sender email
        sender_emails = ["siddharamayya@siddharamayya.in", "me@siddharamayya.in"]
        sender = random.choice(sender_emails)

        # Email to user
        mail_to_user = Mail(
            from_email=Email(sender, "Siddharamayya Mathapati"),
            to_emails=To(email),
            subject=subject,
            html_content=Content("text/html", get_email_html(name, subject, message, sender)),
            plain_text_content=Content("text/plain", 47get_email_plain(name, subject, message, sender))
        )

        # Send email
        logger.info(f"Sending email to {email} from {sender}")
        response_to_user = sg.send(mail_to_user)

        if response_to_user.status_code != 202:
            logger.error(f"SendGrid failed: to_user={response_to_user.status_code}")
            raise HTTPException(status_code=500, detail="Failed to send email")

        logger.info("Email sent successfully")
        return JSONResponse(content={"message": "Email sent successfully"}, status_code=200)

    except SendGridException as sg_error:
        logger.error(f"SendGrid error: {str(sg_error)}")
        raise HTTPException(status_code=500, detail=f"SendGrid error: {str(sg_error)}")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
