from fastapi import APIRouter, HTTPException, Request, Depends
from pydantic import BaseModel, EmailStr, ValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Email, To, Content, Cc, Bcc
from jose import JWTError, jwt
import os
import logging
from html import escape
import random
from datetime import datetime, timedelta
from typing import Optional

# Set up logging (commented out for production)
#logging.basicConfig(level=logging.INFO)
#logger = logging.getLogger(__name__)

router = APIRouter()

# JWT configuration
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "302dc08633d5abeff51ac92c2fbcf5a6ea5a5286f82aa9c121da081a56c24706")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Pydantic model for form data validation
class EmailForm(BaseModel):
    name: str = "Unknown"
    email: EmailStr
    subject: str = "Message from Siddharamayya"
    message: str
    honeypot: str | None = None
    cc: str | None = None
    bcc: str | None = None

# Pydantic model for login
class LoginForm(BaseModel):
    password: str

# JWT token dependency
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        token = credentials.credentials
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username != "admin":
            #logger.warning("Invalid JWT: Incorrect username")
            raise HTTPException(status_code=401, detail="Invalid token")
        return True
    except JWTError as e:
        #logger.error(f"JWT error: {str(e)}")
        raise HTTPException(status_code=401, detail="Invalid token")

# Create JWT token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

# HTML email template (unchanged)
def get_email_html(name: str, subject: str, message: str, sender_email: str) -> str:
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
                justify-content: center;
                align-items: center;
            }}
            .container {{
                width: 100%;
                max-width: 800px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 12px;
                box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
                overflow: hidden;
                flex: 0 1 auto;
                box-sizing: border-box;
            }}
            .navbar {{
                background-color: #404347;
                padding: 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
            }}
            .navbar a {{
                color: #edf2f7;
                text-decoration: none;
                font-size: 20px;
                font-weight: 600;
                transition: color 0.3s ease;
            }}
            .navbar a:hover {{
                color: #63b3ed;
            }}
            .header {{
                background-color: #07b1d0;
                color: #ffffff;
                padding: 20px;
                font-size: 24px;
                font-weight: 700;
                text-align: center;
                border-bottom: 2px solid #06a2c0;
            }}
            .content {{
                padding: 30px;
                font-size: 16px;
            }}
            .message-body {{
                color: #4a5568;
                overflow-wrap: break-word;
                margin: 20px 0;
                font-size: 16px;
                line-height: 1.8;
            }}
            .footer {{
                background-color: #daebdd;
                padding: 25px;
                text-align: center;
                border-top: 1px solid #e2e8f0;
            }}
            .footer p {{
                margin: 8px 0;
                color: #000000;
                font-size: 14px;
                font-weight: 500;
            }}
            .social-links {{
                margin-top: 15px;
                display: flex;
                justify-content: center;
                gap: 15px;
            }}
            .social-links a {{
                text-decoration: none;
            }}
            .social-links img {{
                width: 28px;
                height: 28px;
                transition: transform 0.3s ease;
            }}
            .social-links img:hover {{
                transform: scale(1.1);
            }}
            @media screen and (max-width: 600px) {{
                .container {{
                    margin: 10px;
                    border-radius: 8px;
                }}
                .navbar {{
                    padding: 15px;
                }}
                .navbar a {{
                    font-size: 16px;
                }}
                .header {{
                    font-size: 20px;
                    padding: 15px;
                }}
                .content {{
                    padding: 20px;
                    font-size: 14px;
                }}
                .message-body {{
                    font-size: 14px;
                }}
                .footer {{
                    padding: 20px;
                }}
                .social-links img {{
                    width: 24px;
                    height: 24px;
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
                <p>Best regards,<br>Siddharamayya M</p>
            </div>
            <div class="footer">
                <p>Siddharamayya Mathapati</p>
                <p>Portfolio: <a href="https://siddharamayya.in">https://siddharamayya.in</a></p>
                <p>Email: {sender_email}</p>
                <p>Phone: +91 97406 71620</p>
                <div class="social-links">
                    <a href="https://www.linkedin.com/in/siddharamayya-mathapati" title="LinkedIn">
                        <img src="https://img.icons8.com/color/28/linkedin.png" alt="LinkedIn">
                    </a>
                    <a href="https://medium.com/@msidrm455" title="Medium">
                        <img src="https://img.icons8.com/color/28/medium-monogram.png" alt="Medium">
                    </a>
                    <a href="https://github.com/mtptisid" title="GitHub">
                        <img src="https://img.icons8.com/color/28/github.png" alt="GitHub">
                    </a>
                    <a href="https://www.instagram.com/its_5iD" title="Instagram">
                        <img src="https://img.icons8.com/color/28/instagram-new.png" alt="Instagram">
                    </a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

# Plain-text fallback (unchanged)
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

# Login endpoint
@router.post("/sendmail/login")
async def login(form: LoginForm):
    admin_password = os.environ.get("EMAIL_ADMIN_PASSWORD")
    if not admin_password:
        #logger.error("ADMIN_PASSWORD not configured")
        raise HTTPException(status_code=500, detail="Server configuration error: ADMIN_PASSWORD missing")

    # Verify password (direct comparison for simplicity)
    if form.password != admin_password:
        #logger.warning("Invalid admin password")
        raise HTTPException(status_code=401, detail="Invalid password")

    # Create JWT
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": "admin"}, expires_delta=access_token_expires
    )
    return {"access_token": access_token, "token_type": "bearer"}

@router.post("/sendmail")
async def sendmail(form: EmailForm, request: Request, token_verified: bool = Depends(verify_token)):
    try:
        # Check honeypot field for spam
        if form.honeypot:
            #logger.warning("Spam detected: Honeypot field filled")
            raise HTTPException(status_code=400, detail="Spam detected")

        # Extract and sanitize form data
        name = form.name.strip() if form.name else "Unknown"
        email = form.email.strip()
        subject = form.subject.strip() if form.subject else "Message from Siddharamayya"
        message = form.message.strip()
        cc = form.cc.strip() if form.cc else None
        bcc = form.bcc.strip() if form.bcc else None

        # Validate required fields
        if not email or not message:
            #logger.warning("Validation failed: Email or message missing")
            raise HTTPException(status_code=400, detail="Email and message are required")

        # Validate cc and bcc email formats if provided
        from email_validator import validate_email, EmailNotValidError
        if cc:
            try:
                validate_email(cc, check_deliverability=False)
            except EmailNotValidError as e:
                #logger.warning(f"Invalid CC email: {cc}")
                raise HTTPException(status_code=400, detail=f"Invalid CC email: {str(e)}")
        if bcc:
            try:
                validate_email(bcc, check_deliverability=False)
            except EmailNotValidError as e:
                #logger.warning(f"Invalid BCC email: {bcc}")
                raise HTTPException(status_code=400, detail=f"Invalid BCC email: {str(e)}")

        # Initialize SendGrid client
        api_key = os.environ.get("SENDGRID_API_KEY")
        if not api_key:
            #logger.error("SENDGRID_API_KEY not configured")
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
            plain_text_content=Content("text/plain", get_email_plain(name, subject, message, sender))
        )

        # Add CC and BCC if provided
        if cc:
            mail_to_user.add_cc(Cc(cc))
        if bcc:
            mail_to_user.add_bcc(Bcc(bcc))

        # Send email
        #logger.info(f"Sending email to {email} from {sender}")
        response_to_user = sg.send(mail_to_user)

        if response_to_user.status_code != 202:
            #logger.error(f"SendGrid failed: to_user={response_to_user.status_code}")
            raise HTTPException(status_code=500, detail="Failed to send email")

        #logger.info("Email sent successfully")
        return JSONResponse(content={"message": "Email sent successfully"}, status_code=200)

    except ValidationError as ve:
        #logger.error(f"Validation error: {str(ve)}")
        raise HTTPException(status_code=422, detail=f"Invalid input: {str(ve)}")
    except Exception as e:
        #logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error sending email: {str(e)}")
