from fastapi import APIRouter, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel, EmailStr
from fastapi.responses import JSONResponse
from myapp.utils.email_sender import send_email, EmailSendError
from myapp.services.ai import ai_manager
import os
import logging
from html import escape

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
    # Escape user input to prevent HTML/CSS injection
    name = escape(name)
    email = escape(email)
    subject = escape(subject)
    message = escape(message).replace('\n', '<br>')
    
    return f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>New Contact Form Submission</title>
        <style>
            body {{
                font-family: 'Poppins', Arial, sans-serif;
                line-height: 1.6;
                color: #333333;
                background-color: #f7fafc;
                margin: 0;
                padding: 0;
            }}
            .container {{
                max-width: 600px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .navbar {{
                background-color: #404347;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
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
            .field {{
                margin-bottom: 15px;
            }}
            .field-label {{
                font-weight: 600;
                color: #1a202c;
                margin-bottom: 5px;
            }}
            .field-value {{
                color: #4a5568;
                overflow-wrap: break-word;
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
                    margin: 10px;
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
                <a href="siddharamayya.in">Siddharamayya</a>
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
                <p>Siddharamayya Mathapati</p>
                <p>Portfolio: siddharamayya.in</p>
                <p>Email: me@siddharamayya.in</p>
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

# Plain-text fallback for the email sent to you
def get_email_to_you_plain(name: str, email: str, subject: str, message: str) -> str:
    return f"""
    New Contact Form Submission

    Name: {name}
    Email: {email}
    Subject: {subject}
    Message:
    {message}

    ---
    Siddharamayya M
    Email: me@siddharamayya.in
    Phone: +91 97406 71620
    Portfolio: siddharamayya.in
    LinkedIn: https://www.linkedin.com/in/siddharamayya-mathapati
    Medium: https://medium.com/@msidrm455
    GitHub: https://github.com/mtptisid
    Instagram: https://www.instagram.com/its_5iD
    """

# HTML email template for the acknowledgment email to the user
def get_ack_email_html(name: str, subject: str, message: str, ai_reply: str | None = None) -> str:
    # Escape user input to prevent HTML/CSS injection
    name = escape(name)
    subject = escape(subject)
    message = escape(message).replace('\n', '<br>')

    if ai_reply:
        reply_html = "".join(
            f"<p>{escape(p.strip()).replace(chr(10), '<br>')}</p>"
            for p in ai_reply.strip().split("\n\n") if p.strip()
        )
        ack_content = (
            f"<p>Dear {name},</p>"
            f"<p>Thank you for reaching out!</p>"
            f"{reply_html}"
            f"<p>Best regards,<br>Siddharamayya M</p>"
        )
    else:
        ack_content = (
            f"<p>Dear {name},</p>"
            f"<p>Thank you for reaching out! I have received your message:</p>"
            f"<blockquote style='border-left: 4px solid #07b1d0; padding-left: 15px; margin: 15px 0; color: #4a5568;'>{message}</blockquote>"
            f"<p>Your inquiry is important to me; I will respond at the earliest opportunity.</p>"
            f"<p>Best regards,<br>Siddharamayya M</p>"
        ) if subject != "Contact Form Submission" else (
            f"<p>Dear {name},</p>"
            f"<p>Thank you for contacting me through my portfolio. I appreciate your interest and will respond to your inquiry soon.</p>"
            f"<p>Best regards,<br>Siddharamayya M</p>"
        )
    
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
            }}
            .container {{
                max-width: 600px;
                margin: 20px auto;
                background: #ffffff;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                overflow: hidden;
            }}
            .navbar {{
                background-color: #404347;
                padding: 15px 20px;
                display: flex;
                justify-content: space-between;
                align-items: center;
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
                    margin: 10px;
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
                <a href="siddharamayya.in">Siddharamayya</a>
            </div>
            <div class="header">
                {subject}
            </div>
            <div class="content">
                {ack_content}
            </div>
            <div class="footer">
                <p>Siddharamayya Mathapati</p>
                <p>Portfolio: siddharamayya.in</p>
                <p>Email: me@siddharamayya.in</p>
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

# Plain-text fallback for the acknowledgment email to the user
def get_ack_email_plain(name: str, subject: str, message: str, ai_reply: str | None = None) -> str:
    if ai_reply:
        return f"""
        Dear {name},

        Thank you for reaching out!

        {ai_reply.strip()}

        Best regards,
        Siddharamayya M
        Email: me@siddharamayya.in
        Phone: +91 97406 71620
        Portfolio: siddharamayya.in
        LinkedIn: https://www.linkedin.com/in/siddharamayya-mathapati
        Medium: https://medium.com/@msidrm455
        GitHub: https://github.com/mtptisid
        Instagram: https://www.instagram.com/its_5iD
        """
    if subject != "Contact Form Submission":
        return f"""
        Dear {name},

        Thank you for reaching out! I have received your message:

        ---
        {message}
        ---

        Your inquiry is important to me; I will respond at the earliest opportunity.

        Best regards,
        Siddharamayya M
        Email: me@siddharamayya.in
        Phone: +91 97406 71620
        Portfolio: siddharamayya.in
        LinkedIn: https://www.linkedin.com/in/siddharamayya-mathapati
        Medium: https://medium.com/@msidrm455
        GitHub: https://github.com/mtptisid
        Instagram: https://www.instagram.com/its_5iD
        """
    return f"""
    Dear {name},

    Thank you for contacting me through my portfolio. I appreciate your interest and will respond to your inquiry soon.

    Best regards,
    Siddharamayya M
    Email: me@siddharamayya.in
    Phone: +91 97406 71620
    Portfolio: siddharamayya.in
    LinkedIn: https://www.linkedin.com/in/siddharamayya-mathapati
    Medium: https://medium.com/@msidrm455
    GitHub: https://github.com/mtptisid
    Instagram: https://www.instagram.com/its_5iD
    """

async def generate_ai_reply(name: str, message: str) -> str | None:
    """Ask Gemini to draft a personal reply to the sender's message. Returns None on failure so callers fall back to the canned reply."""
    prompt = [
        {
            "role": "system",
            "content": (
                "You are Siddharamayya M, replying personally by email to someone who messaged you "
                "through the contact form on your portfolio website (siddharamayya.in). Write a warm, "
                "concise, professional reply that directly addresses what they wrote — answer any "
                "questions, or acknowledge their point and say you'll follow up with more detail soon. "
                "Do not include a greeting, subject line, or sign-off — those are added separately. "
                "Do not quote the entire original message back. Plain text only, no markdown, "
                "separate paragraphs with a blank line."
            ),
        },
        {"role": "user", "content": f"Sender name: {name}\nMessage:\n{message}"},
    ]
    try:
        reply = await ai_manager.get_response("gemini", prompt)
        reply = (reply or "").strip()
        return reply or None
    except Exception as e:
        logger.error(f"AI reply generation failed, falling back to canned reply: {str(e)}")
        return None


async def send_contact_emails(name: str, email: str, subject: str, message: str, ack_subject: str) -> None:
    """Runs in the background: notifies the site owner, then drafts + sends the AI-personalized ack email."""
    try:
        send_email(
            to_email="msidrm455@gmail.com",
            subject=f"New Contact Form Submission: {subject}",
            html_content=get_email_to_you_html(name, email, subject, message),
            plain_content=get_email_to_you_plain(name, email, subject, message),
        )
    except EmailSendError as e:
        logger.error(f"Failed to send owner notification email: {str(e)}")

    ai_reply = await generate_ai_reply(name, message)

    try:
        send_email(
            to_email=email,
            subject=ack_subject,
            html_content=get_ack_email_html(name, ack_subject, message, ai_reply),
            plain_content=get_ack_email_plain(name, ack_subject, message, ai_reply),
            bcc="msidrm455@gmail.com",
        )
    except EmailSendError as e:
        logger.error(f"Failed to send acknowledgment email: {str(e)}")


@router.post("/contact")
async def contact(form: ContactForm, request: Request, background_tasks: BackgroundTasks):
    try:
        # Check honeypot field for spam
        if form.honeypot:
            logger.warning("Spam detected: Honeypot field filled")
            raise HTTPException(status_code=400, detail="Spam detected")

        # Extract and sanitize form data
        name = form.name.strip()
        email = form.email.strip()
        subject = form.subject.strip()
        message = form.message.strip()

        # Validate required fields
        if not email or not message:
            logger.warning("Validation failed: Email or message missing")
            raise HTTPException(status_code=400, detail="Email and message are required")

        ack_subject = subject if subject != "Contact Form Submission" else "Thank You for Contacting Me"

        # Emails (including the Gemini-drafted reply) are sent after the response,
        # so the frontend gets an immediate success without waiting on AI + SMTP.
        background_tasks.add_task(send_contact_emails, name, email, subject, message, ack_subject)

        return JSONResponse(content={"message": "Message received successfully"}, status_code=200)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")
