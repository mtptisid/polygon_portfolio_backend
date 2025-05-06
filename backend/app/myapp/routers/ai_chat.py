from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from ..services.ai import ai_manager
from langchain.tools import DuckDuckGoSearchRun
import asyncio
import logging
import re
from fastapi.responses import JSONResponse

# Set up #logging
logging.basicConfig(level=logging.INFO)
logger = logging.get#logger(__name__)

router = APIRouter(
    prefix="/api/ai_chat",
    tags=["Chat"]
)

# Define Pydantic models
class MessageCreate(BaseModel):
    content: str
    model: str
    session_id: Optional[int] = None
    tool: Optional[str] = None

class MessageResponse(BaseModel):
    content: str
    is_bot: bool
    session_id: int
    timestamp: datetime
    tool_used: Optional[str] = None

class SessionMessage(BaseModel):
    content: str
    is_bot: bool
    timestamp: datetime
    tool_used: Optional[str] = None

class Session(BaseModel):
    session_id: int
    messages: List[SessionMessage]
    created_at: datetime

# Initialize DuckDuckGo search tool
search = DuckDuckGoSearchRun()

# In-memory session store (replace with database for production)
SESSIONS = {}

async def search_web(query: str, model: str = "groq") -> str:
    """Perform a web search using DuckDuckGo via LangChain with fixed site-specific query."""
    sites = ["github.com", "linkedin.com"]  # Fixed sites
    
    site_queries = [f"site:{site}" for site in sites]
    full_query = f"{query} {' OR '.join(site_queries)}"
    #logger.info(f"Performing search with query: {full_query}")
    
    try:
        # Run synchronous search in a separate thread to avoid blocking
        search_results = await asyncio.to_thread(search.run, full_query)
        if not search_results:
            return "Web Search Results: No results found."
        # Format search results as plain text
        formatted_results = "Web Search Results:\n"
        result_lines = search_results.split("\n")
        for line in result_lines:
            if line.strip():
                formatted_results += f"{line.strip()}\n"
        formatted_results += "\nNote: Format these URLs in Markdown as `[name](link)` in the final response."
        #logger.info(f"Formatted search results: {formatted_results}")
        return formatted_results
    except Exception as e:
        #logger.error(f"Search failed: {str(e)}")
        return f"Web Search Results: Search failed: {str(e)}"

def clean_text(text: str) -> str:
    """Clean text by removing excessive newlines."""
    # Replace 3+ newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    return text

@router.post("/request", response_model=MessageResponse)
async def send_message(request: Request, message: MessageCreate):
    """Send a message to the selected AI model, optionally using tools."""
    # Log raw request payload for debugging
    raw_body = await request.body()
    #logger.info(f"Raw request payload: {raw_body.decode('utf-8')}")
    
    if not message.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    # Assign session_id if not provided (use UUID)
    session_id = message.session_id or str(uuid4())

    # Initialize session if new
    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.utcnow()
        }

    # Store user message
    user_message = SessionMessage(
        content=message.content,
        is_bot=False,
        timestamp=datetime.utcnow(),
        tool_used=None
    )
    SESSIONS[session_id]["messages"].append(user_message)

    # Structured profile data about Siddharamayya
    PROFILE_DATA = {
        "about": {
            "full_name": "Siddharamayya M",
            "dob": "April 14, 1997",
            "languages": ["English", "Kannada", "Hindi", "Telugu", "Tamil", "Marathi", "Malayalam"],
            "email": "msidrm455@gmail.com",
            "phone": "+91 97406 71620",
            "location": "Yadur, Chikodi, Belagavi, Karnataka, India",
            "description": "Siddharamayya is passionate about coding and cutting-edge technologies. He thrives on creating innovative, scalable solutions, always seeking to push boundaries in cloud, automation, and AI.",
            "availability": "Immediately available for new opportunities as of May 2025."
        },
        "skills": [
            {"name": "Python", "proficiency": "90%"},
            {"name": "Deep Learning", "proficiency": "85%"},
            {"name": "Machine Learning/AI", "proficiency": "85%"},
            {"name": "AI Development", "proficiency": "70%"},
            {"name": "Data Science", "proficiency": "85%"},
            {"name": "DBMS & SQL", "proficiency": "80%"},
            {"name": "Web Development", "proficiency": "95%"},
            {"name": "App Development", "proficiency": "80%"},
            {"name": "DevOps", "tools": ["Docker", "Kubernetes", "Ansible", "GitHub Actions"]}
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Capgemini Technology Service Limited India",
                "duration": "May 2021 - April 2025",
                "description": "Over 4 years of experience, specializing in AI, machine learning, and modern web development."
            }
        ],
        "education": [
            {
                "degree": "MCA - Master of Computer Applications",
                "institution": "Acharya Institutes of Technology, Bangalore",
                "duration": "July 2018 - September 2020",
                "score": "CGPA: 7.82/10"
            },
            {
                "degree": "BCA - Bachelor of Computer Applications",
                "institution": "KLE’s Basavprabhu Kore College, Chikodi",
                "duration": "July 2015 - June 2018",
                "score": "68%"
            },
            {
                "degree": "PUC - Intermediate",
                "institution": "KLE’s Basavprabhu Kore PU College, Chikodi",
                "duration": "May 2013 - May 2015",
                "score": "58%"
            }
        ],
        "projects": [
            {
                "category": "Web Development",
                "description": "Developed responsive, user-friendly web applications using React, Flask, and Streamlit."
            },
            {
                "category": "Deep Learning",
                "description": "Built intelligent systems using advanced neural networks to analyze and predict from large datasets."
            },
            {
                "category": "Machine Learning",
                "description": "Created predictive models for real-time data analysis and decision-making."
            },
            {
                "category": "Generative AI",
                "description": "Explored innovative content creation with algorithms generating realistic text, images, and data."
            },
            {
                "category": "AI-Powered Finance",
                "description": "Developed models for stock market predictions using ML and LLMs."
            },
            {
                "category": "NLP and Chatbots",
                "description": "Built autonomous AI agents for natural language processing."
            }
        ],
        "contact": {
            "email": "msidrm455@gmail.com",
            "phone": "+91 97406 71620",
            "address": "#372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India"
        },
        "online_presence": {
            "github": "https://github.com/mtptisid",
            "portfolio": "https://mtptisid.github.io"
        },
        "interests": [
            "Generative AI & LLM Development",
            "AI-powered Stock Market Predictions",
            "NLP, Chatbots, and Autonomous AI Agents",
            "Cloud & DevOps for AI Deployment"
        ],
        "tech_stack": [
            "Python", "LangChain", "Hugging Face", "TensorFlow", "PyTorch",
            "Streamlit", "Flask", "FastAPI", "React",
            "Docker", "Kubernetes", "GitHub Actions",
            "RHEL", "NFS", "Ansible"
        ]
    }

    # Few-shot examples with Markdown-formatted answers, including [name](link) for URLs
    FEW_SHOT_EXAMPLES = [
        {
            "content": "What is the total experience of Siddharamayya?",
            "response": "**Siddharamayya Mathapati** has over **4 years** of professional experience, showcasing expertise in advanced technical domains.\n\n- **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025): Specialized in developing **AI** and **machine learning** models for predictive analytics, building responsive web applications using modern frameworks, and implementing **DevOps** practices with tools like **Docker** and **Kubernetes** to streamline deployment processes."
        },
        {
            "content": "What is Siddharamayya great at technically?",
            "response": "**Siddharamayya Mathapati** is a highly skilled technologist with proficiency across multiple domains, enabling him to deliver innovative and scalable solutions.\n\n- **Python** (90%): Writes efficient, maintainable code for **AI**, web, and automation projects, leveraging libraries like **TensorFlow** and **LangChain**.\n- **Web Development** (95%): Builds responsive, user-friendly applications using **React**, **Flask**, and **Streamlit**, ensuring seamless user experiences.\n- **Deep Learning** & **ML/AI** (85%): Designs and trains neural networks for complex data analysis and predictive modeling, with applications in finance and **NLP**.\n- **DevOps**: Utilizes **Docker**, **Kubernetes**, and **Ansible** to automate and optimize **CI/CD** pipelines, enhancing deployment reliability.\n\n```python\ndef predict_stock(data):\n    model = load_model('stock_predictor.h5')\n    return model.predict(data)\n```"
        },
        {
            "content": "What are Siddharamayya's projects?",
            "response": "**Siddharamayya Mathapati** has worked on a diverse range of projects that showcase his expertise in **AI**, **web development**, and innovative technologies.\n\n- **Web Development**: Created responsive, user-friendly applications using **React**, **Flask**, and **Streamlit**, focusing on intuitive interfaces and scalability.\n- **Deep Learning**: Developed intelligent systems with neural networks to analyze and predict from large datasets, applied in real-world scenarios.\n- **AI-Powered Finance**: Built **machine learning** models for stock market predictions, integrating **large language models** for enhanced accuracy.\n- **NLP and Chatbots**: Designed autonomous **AI agents** for **natural language processing**, enabling efficient human-machine interactions.\n- **Generative AI**: Explored algorithms for generating realistic text, images, and data, pushing the boundaries of creative **AI** applications.\n\n- [GitHub](https://github.com/mtptisid)\n- [Portfolio](https://mtptisid.github.io)"
        },
        {
            "content": "What is Siddharamayya's GitHub page?",
            "response": "**Siddharamayya Mathapati**’s **GitHub** profile is a comprehensive showcase of his technical expertise and project contributions.\n\n- [GitHub](https://github.com/mtptisid)\n- **Projects**: Features repositories on **AI/ML** (predictive models for data analysis), **Web Development** (**React**-based applications with **Flask** backends), and **DevOps** (**CI/CD** pipelines using **GitHub Actions**, **Docker**, and **Kubernetes**), reflecting his ability to build end-to-end solutions."
        },
        {
            "content": "How can I contact Siddharamayya?",
            "response": "**Siddharamayya Mathapati** is accessible through multiple channels for professional and collaborative opportunities.\n\n- **[Email](mailto:msidrm455@gmail.com)**: Preferred for formal communication and project inquiries.\n- **Phone**: +91 97406 71620 – Available for direct calls or messages to discuss technical or professional matters.\n- **Address**: #372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India – Suitable for official correspondence or in-person meetings."
        },
        {
            "content": "What languages does Siddharamayya speak?",
            "response": "**Siddharamayya Mathapati** is a multilingual individual who speaks **seven languages**, enabling effective communication across diverse regions in **India**.\n\n- **Kannada**: Native language; fluent in speaking, reading, and writing.\n- **English**: Professional fluency; proficient in both verbal and written communication, ideal for technical, academic, and workplace settings.\n- **Hindi**: Fluent; skilled in reading, writing, and speaking.\n- **Telugu**: Conversational fluency; comfortable in everyday communication.\n- **Tamil**: Conversational fluency; able to understand and speak in most day-to-day scenarios.\n- **Marathi**: Basic to intermediate fluency; capable of effective communication in common interactions.\n- **Malayalam**: Basic understanding; can comprehend and engage in simple conversations."
        },
        {
            "content": "When will Siddharamayya be available to join?",
            "response": "**Siddharamayya Mathapati** is immediately available to join new opportunities as of **May 2025**, following the completion of his tenure at **Capgemini**.\n\n- **Availability**: Ready to take on roles in **AI**, **machine learning**, **web development**, or **DevOps**, bringing over **4 years** of experience from his work at **Capgemini Technology Service Limited India** (May 2021 - April 2025), where he specialized in innovative, scalable solutions."
        }
    ]

    # Prepare system prompt with Markdown emphasis and [name](link) for URLs
    system_prompt = (
        "You are an AI assistant with detailed knowledge about Siddharamayya Mathapati, an AI/ML enthusiast and Senior Software Engineer. "
        "Your responses must be in Markdown format, using bold, code blocks, lists, and [name](link) for URLs as specified. "
        "Answer questions about Siddharamayya using the profile data and few-shot examples. For unrelated questions, use web search results or general knowledge. "
        "Include chat history for context if provided.\n"
        "**Profile Data**:\n"
        f"**Full Name**: {PROFILE_DATA['about']['full_name']}\n"
        f"**DOB**: {PROFILE_DATA['about']['dob']}\n"
        f"**Languages**: {', '.join(PROFILE_DATA['about']['languages'])}\n"
        f"**[Email](mailto:{PROFILE_DATA['about']['email']})**\n"
        f"**Phone**: {PROFILE_DATA['about']['phone']}\n"
        f"**Location**: {PROFILE_DATA['about']['location']}\n"
        f"**Description**: {PROFILE_DATA['about']['description']}\n"
        f"**Availability**: {PROFILE_DATA['about']['availability']}\n\n"
        "**Skills**:\n" +
        "\n".join([f"- {s['name']} ({s.get('proficiency', 'Proficient')}): {s.get('tools', '')}" for s in PROFILE_DATA['skills']]) + "\n\n"
        "**Experience**:\n" +
        "\n".join([f"- {e['title']} at {e['company']} ({e['duration']}): {e['description']}" for e in PROFILE_DATA['experience']]) + "\n\n"
        "**Education**:\n" +
        "\n".join([f"- {e['degree']} at {e['institution']} ({e['duration']}): {e['score']}" for e in PROFILE_DATA['education']]) + "\n\n"
        "**Projects**:\n" +
        "\n".join([f"- {p['category']}: {p['description']}" for p in PROFILE_DATA['projects']]) + "\n\n"
        "**Contact**:\n" +
        f"- **[Email](mailto:{PROFILE_DATA['contact']['email']})**\n"
        f"- **Phone**: {PROFILE_DATA['contact']['phone']}\n"
        f"- **Address**: {PROFILE_DATA['contact']['address']}\n\n"
        "**Online Presence**:\n" +
        f"- [GitHub]({PROFILE_DATA['online_presence']['github']})\n"
        f"- [Portfolio]({PROFILE_DATA['online_presence']['portfolio']})\n\n"
        "**Interests**:\n" +
        "\n".join([f"- {i}" for i in PROFILE_DATA['interests']]) + "\n\n"
        "**Tech Stack**:\n" +
        "\n".join([f"- {t}" for t in PROFILE_DATA['tech_stack']]) + "\n\n"
        "**Few-Shot Examples**:\n" +
        "\n".join([f"**Q**: {ex['content']}\n**A**: {ex['response']}" for ex in FEW_SHOT_EXAMPLES]) + "\n\n"
        "**Instructions**:\n"
        "- Always respond in Markdown format.\n"
        "- Use **bold** for key terms like names, titles, or important concepts.\n"
        "- Wrap code snippets in triple backticks (```python ... ```).\n"
        "- Use `-` for bulleted lists, with one item per line and a blank line before and after.\n"
        "- For URLs, use Markdown link syntax `[name](link)`, where `name` is a descriptive label (e.g., `[GitHub](https://github.com/mtptisid)` instead of `https://github.com/mtptisid`).\n"
        "- Use proper spacing: one blank line between paragraphs, lists, and code blocks.\n"
        "- Ensure readability with clear section breaks (e.g., `---` for separators if needed).\n"
        "- Even if the question is unrelated to Siddharamayya, format the response in Markdown using lists, bold terms, code blocks, and `[name](link)` for URLs as appropriate.\n\n"
        "**Chat History**:\n" +
        "\n".join([f"**{'Bot' if msg.is_bot else 'User'}**: {msg.content}" for msg in SESSIONS.get(session_id, {"messages": []})["messages"]]) + "\n\n"
        "**Question**:\n" +
        message.content
    )

    # Prepare chat history for AI
    chat_history = [
        {"role": "system", "content": system_prompt}
    ]

    # Handle tool usage
    response_content = ""
    tool_used = None

    if message.tool == "SearchWeb":
        tool_used = "SearchWeb"
        search_result = await search_web(message.content, message.model)
        augmented_prompt = (
            f"{system_prompt}\n"
            f"**Web Search Results**:\n{search_result}\n"
            "Provide a detailed response in Markdown, using `[name](link)` for all URLs in the search results or elsewhere."
        )
        chat_history = [
            {"role": "system", "content": augmented_prompt}
        ]

    # Get AI response
    try:
        response_content = await ai_manager.get_response(message.model, chat_history)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")

    # Clean up response
    response_content = clean_text(response_content)

    # Store bot response
    bot_message = SessionMessage(
        content=response_content,
        is_bot=True,
        timestamp=datetime.utcnow(),
        tool_used=tool_used
    )
    SESSIONS[session_id]["messages"].append(bot_message)

    # Prepare response
    return MessageResponse(
        content=response_content,
        is_bot=True,
        session_id=session_id,
        timestamp=datetime.utcnow(),
        tool_used=tool_used
    )

@router.get("/history", response_model=List[Session])
async def get_session_history():
    """Retrieve all session histories."""
    return [Session(**session) for session in SESSIONS.values()]
