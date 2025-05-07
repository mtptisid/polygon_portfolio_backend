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

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    sites = ["github.com", "linkedin.com"]
    site_queries = [f"site:{site}" for site in sites]
    full_query = f"{query} {' OR '.join(site_queries)}"
    logger.info(f"Performing search with query: {full_query}")
    
    try:
        search_results = await asyncio.to_thread(search.run, full_query)
        if not search_results:
            return "Web Search Results: No results found."
        formatted_results = "Web Search Results:\n"
        result_lines = search_results.split("\n")
        for line in result_lines:
            if line.strip():
                formatted_results += f"{line.strip()}\n"
        formatted_results += "\nNote: Format these URLs in Markdown as `[name](link)` in the final response."
        logger.info(f"Formatted search results: {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return f"Web Search Results: Search failed: {str(e)}"

def clean_text(text: str) -> str:
    """Clean text by removing excessive newlines and unwanted characters."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[\r\t]', '', text)
    text = text.strip()
    return text

@router.post("/request", response_model=MessageResponse)
async def send_message(request: Request, message: MessageCreate):
    """Send a message to the selected AI model, optionally using tools."""
    raw_body = await request.body()
    logger.info(f"Raw request payload: {raw_body.decode('utf-8')}")
    
    if not message.content.strip():
        raise HTTPException(status_code=400, detail="Message content cannot be empty")

    session_id = message.session_id or str(uuid4())

    if session_id not in SESSIONS:
        SESSIONS[session_id] = {
            "session_id": session_id,
            "messages": [],
            "created_at": datetime.utcnow()
        }

    user_message = SessionMessage(
        content=message.content,
        is_bot=False,
        timestamp=datetime.utcnow(),
        tool_used=None
    )
    SESSIONS[session_id]["messages"].append(user_message)

    PROFILE_DATA = {
        "about": {
            "full_name": "Siddharamayya Mathapati",
            "dob": "April 14, 1997",
            "languages": ["English", "Kannada", "Hindi", "Telugu", "Tamil", "Marathi", "Malayalam"],
            "email": "msidrm455@gmail.com",
            "phone": "+91 97406 71620",
            "location": "Yadur, Chikodi, Belagavi, Karnataka, India",
            "description": (
                "**Siddharamayya Mathapati** is a seasoned **AI/ML Engineer** and **Senior Software Engineer** with over **4 years** of experience in **AI**, **MLOps**, **DevOps**, and **web development**. "
                "He excels in **large language models (LLMs)**, **generative AI**, and **cloud-native solutions**, delivering scalable systems for industries like **finance**, **healthcare**, and **cybersecurity**. "
                "At **Capgemini**, he led **MLOps pipelines**, optimized **LLM fine-tuning**, and mentored teams. His **Udemy certifications** and diverse project portfolio, including **IoT** and **NLP** solutions, showcase his commitment to innovation."
            ),
            "availability": "Immediately available for new opportunities as of May 2025."
        },
        "skills": [
            {"name": "Python", "proficiency": "90%", "description": "Expert in **AI/ML**, **web development**, and **automation** with **TensorFlow**, **PyTorch**, **LangChain**, and **FastAPI**."},
            {"name": "Deep Learning", "proficiency": "85%", "description": "Designs **neural networks** for **NLP**, **computer vision**, and **predictive modeling** using **TensorFlow** and **PyTorch**."},
            {"name": "Machine Learning/AI", "proficiency": "85%", "description": "Builds **LLMs**, **RAG applications**, and **predictive models** with **QLoRA** and **RLHF**."},
            {"name": "Data Science", "proficiency": "85%", "description": "Skilled in **data preprocessing**, **visualization**, and **analysis** with **Pandas**, **NumPy**, and **Matplotlib**."},
            {"name": "DevOps", "tools": ["Docker", "Kubernetes", "Ansible", "Jenkins", "GitHub Actions"], "description": "Automates **CI/CD pipelines** and manages **cloud infrastructure**."},
            {"name": "Web Development", "proficiency": "95%", "description": "Develops responsive applications with **React**, **Flask**, **Django**, and **Streamlit**."},
            {"name": "DBMS & SQL", "proficiency": "80%", "description": "Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB)."},
            {"name": "Cloud Engineering", "proficiency": "85%", "description": "Optimizes **AWS**, **GCP**, and **Azure** for cost and performance."},
            {"name": "IoT", "proficiency": "80%", "description": "Builds real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**."}
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Capgemini Technology Service Limited India",
                "location": "Navi Mumbai, India",
                "duration": "May 2021 - April 2025",
                "description": (
                    "Spearheaded **AI/ML** projects, fine-tuning **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing memory usage by **30%**. "
                    "Designed **RAG applications** with **FAISS** and **ChromaDB**, improving response accuracy by **25%**. "
                    "Automated **MLOps pipelines** using **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**. "
                    "Developed **real-time inference pipelines** with **TensorRT** and **ONNX** for low-latency applications. "
                    "Migrated legacy ML workloads to **Spark** and **Kubernetes**, enhancing scalability by **40%**. "
                    "Built **model retraining pipelines** with **drift detection**, reducing errors by **30%**. "
                    "Mentored junior engineers and led research internships."
                )
            },
            {
                "title": "Project Intern",
                "company": "X-Cencia Technology Solution Limited India",
                "location": "Bengaluru, India",
                "duration": "February 2020 - April 2021",
                "description": (
                    "Developed **computer vision models** with **TensorFlow** for real-time applications. "
                    "Tuned **Word2Vec** models for **recommendation systems**. "
                    "Built **IoT solutions**, including a **smart school bus tracking system** with **RFID** and **GPS**, and a **car security system** with **facial recognition**. "
                    "Designed **automated data pipelines** for **sensor data** processing."
                )
            }
        ],
        "education": [
            {
                "degree": "MCA - Master of Computer Applications",
                "institution": "Acharya Institute of Technology, Bangalore",
                "duration": "July 2018 - September 2020",
                "score": "CGPA: 7.2/10"
            },
            {
                "degree": "BCA - Bachelor of Computer Applications",
                "institution": "B K College, Chikodi",
                "duration": "July 2015 - June 2018",
                "score": "CGPA: 6.8/10"
            }
        ],
        "projects": [
            {
                "category": "LinkedIn Activity Scraper",
                "description": (
                    "A **Python-based tool** using **Selenium** and **BeautifulSoup** to extract **LinkedIn** activity data (posts, likes, shares, comments). "
                    "Features **dynamic content handling** and **ethical scraping**, supporting **social media analytics**."
                ),
                "link": "[LinkedIn Comment Poster](https://github.com/mtptisid/linkedin-comment-poster)"
            },
            {
                "category": "Recruitment Automation",
                "description": (
                    "An **AI-driven solution** for extracting job data from career pages and generating personalized **cold emails**. "
                    "Uses **NLP** and **web scraping**, reducing manual effort by **80%**."
                ),
                "link": "[Recruitment Solution](https://github.com/mtptisid/recruitment-solution)"
            },
            {
                "category": "Stock Price Prediction",
                "description": (
                    "An **AI-powered finance agent** using **LLMs** to predict stock prices, integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy."
                ),
                "link": "[Stock Price Prediction](https://github.com/mtptisid/stock-price-prediction)"
            },
            {
                "category": "Credit Risk Analysis",
                "description": (
                    "A **machine learning model** with **CatBoostClassifier** to predict credit card defaults, achieving **90% accuracy**."
                ),
                "link": "[Credit Risk Analysis](https://github.com/mtptisid/credit-risk-analysis)"
            },
            {
                "category": "Medical Assistant AI",
                "description": (
                    "A **web application** using **Google Generative AI** for **medical image analysis** with a **chat interface** for diagnostic insights."
                ),
                "link": "[Medical Assistant AI](https://github.com/mtptisid/medical-assistant-ai)"
            },
            {
                "category": "SQL Query Generator",
                "description": (
                    "A **Jupyter Notebook-based tool** integrating **Google Gemini LLM** with **MySQL** using **LangChain** for natural language **SQL query** generation."
                ),
                "link": "[Simple SQL with Gemini](https://github.com/mtptisid/simple-sql-gemini)"
            },
            {
                "category": "Student Study Assistant",
                "description": (
                    "A **chatbot** using **LangChain**, **Gemini LLM**, and **PDF processing** to answer student queries from uploaded documents."
                ),
                "link": "[Student Study Assistant](https://github.com/mtptisid/student-study-assistant)"
            },
            {
                "category": "Smart School Bus Tracking",
                "description": (
                    "An **IoT-based system** using **RFID**, **GPS**, and **Raspberry Pi** for real-time student location monitoring."
                ),
                "link": "[Smart Bus Tracking](https://github.com/mtptisid/smart-bus-tracking)"
            },
            {
                "category": "Car Security System",
                "description": (
                    "A **real-time AI-based system** using **IoT**, **facial recognition**, and **RFID** for vehicle security."
                ),
                "link": "[Car Security System](https://github.com/mtptisid/car-security-system)"
            }
        ],
        "contact": {
            "email": "msidrm455@gmail.com",
            "phone": "+91 97406 71620",
            "address": "#372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India"
        },
        "online_presence": {
            "github": "https://github.com/mtptisid",
            "portfolio": "https://mtptisid.github.io",
            "linkedin": "https://linkedin.com/in/siddharamayya"
        },
        "interests": [
            "Generative AI & LLM Development",
            "AI-Powered Financial Modeling",
            "NLP & Chatbots",
            "Cloud-Native MLOps & DevOps",
            "IoT & Real-Time Systems"
        ],
        "tech_stack": [
            "Python", "C++", "Ruby", "JavaScript", "Golang", "Bash",
            "TensorFlow", "PyTorch", "LangChain", "Hugging Face", "PySpark",
            "Flask", "Django", "FastAPI", "React", "Streamlit",
            "Docker", "Kubernetes", "Ansible", "Jenkins", "GitHub Actions",
            "AWS (SageMaker, EMR, Lambda)", "GCP", "Azure",
            "MySQL", "PostgreSQL", "MongoDB", "ChromaDB", "FAISS"
        ],
        "certifications": [
            "LLM Engineering: Master AI, Large Language Models & Agents (Udemy)",
            "MLOps Bootcamp: Mastering AI Operations for Success (Udemy)",
            "Python for Data Analysis & Visualization (Udemy)",
            "Deep Learning Masterclass with TensorFlow 2 (Udemy)"
        ]
    }

    FEW_SHOT_EXAMPLES = [
        {
            "content": "What is the total experience of Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** has over **4 years** of professional experience as a **Senior Software Engineer** and **AI/ML Engineer**, delivering cutting-edge solutions in **AI**, **MLOps**, **DevOps**, and **IoT**.\n\n"
                "- **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025):\n"
                "  - Led **AI/ML** initiatives, fine-tuning **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing computational overhead by **30%**.\n"
                "  - Designed **RAG applications** using **FAISS** and **ChromaDB**, improving response accuracy by **25%**.\n"
                "  - Automated **MLOps pipelines** with **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**.\n"
                "  - Developed **real-time inference pipelines** with **TensorRT** and **ONNX** for low-latency applications.\n"
                "  - Optimized legacy ML workloads with **Spark** and **Kubernetes**, enhancing scalability by **40%**.\n"
                "  - Mentored junior engineers and led research internships.\n"
                "- **Project Intern** at **X-Cencia Technology Solution Limited India** (February 2020 - April 2021):\n"
                "  - Built **computer vision models** with **TensorFlow** for real-time applications.\n"
                "  - Developed **IoT solutions**, including a **smart school bus tracking system** with **RFID** and **GPS**, and a **car security system** with **facial recognition**.\n"
                "  - Tuned **Word2Vec** models for **recommendation systems**.\n\n"
                "His extensive experience underscores his ability to deliver **scalable**, **innovative** solutions across diverse domains."
            )
        },
        {
            "content": "When will Siddharamayya be available to join?",
            "response": (
                "**Siddharamayya Mathapati** is **immediately available** to join new opportunities starting **May 2025**, following his tenure at **Capgemini Technology Service Limited India**.\n\n"
                "- **Availability Details**:\n"
                "  - Ready for roles in **AI/ML**, **MLOps**, **DevOps**, or **web development**.\n"
                "  - Brings **4+ years** of experience, including **LLM fine-tuning**, **RAG applications**, **cloud-native deployments**, and **IoT systems**.\n"
                "  - Proven expertise at **Capgemini** (May 2021 - April 2025), where he led **AI-driven projects**, automated **MLOps pipelines**, and mentored teams.\n"
                "- **Ideal Roles**:\n"
                "  - **AI/ML Engineer**: Developing **generative AI** or **predictive models**.\n"
                "  - **MLOps Engineer**: Automating **CI/CD pipelines** for ML deployments.\n"
                "  - **DevOps Engineer**: Managing **Kubernetes** and **AWS**-based infrastructure.\n"
                "  - **IoT Developer**: Building real-time systems with **Raspberry Pi** and **RFID**.\n\n"
                "Contact him via **[Email](mailto:msidrm455@gmail.com)** or **Phone** (+91 97406 71620) to discuss opportunities."
            )
        },
        {
            "content": "What is Siddharamayya great at technically?",
            "response": (
                "**Siddharamayya Mathapati** is a versatile technologist with exceptional expertise in **AI/ML**, **DevOps**, **web development**, **cloud engineering**, and **IoT**, enabling him to build end-to-end solutions.\n\n"
                "- **Python** (90%):\n"
                "  - Writes clean, modular code for **AI/ML**, **web development**, and **automation**.\n"
                "  - Leverages **TensorFlow**, **PyTorch**, **LangChain**, and **FastAPI**.\n"
                "  - Example:\n"
                "    ```python\n"
                "    from langchain import LLMChain\n"
                "    from langchain.prompts import PromptTemplate\n"
                "    def create_rag_chain(model, vector_db):\n"
                "        prompt = PromptTemplate(input_variables=['query', 'context'])\n"
                "        return LLMChain(llm=model, prompt=prompt, retriever=vector_db)\n"
                "    ```\n"
                "- **Deep Learning** & **Machine Learning/AI** (85%):\n"
                "  - Designs **neural networks** for **NLP**, **computer vision**, and **predictive modeling**.\n"
                "  - Fine-tunes **LLMs** with **QLoRA** and **RLHF** for domain-specific tasks.\n"
                "- **Web Development** (95%):\n"
                "  - Builds responsive applications with **React**, **Flask**, **Django**, and **Streamlit**.\n"
                "  - Example:\n"
                "    ```python\n"
                "    from flask import Flask, render_template\n"
                "    app = Flask(__name__)\n"
                "    @app.route('/')\n"
                "    def home():\n"
                "        return render_template('index.html')\n"
                "    ```\n"
                "- **DevOps**:\n"
                "  - Automates **CI/CD pipelines** with **Docker**, **Kubernetes**, **Ansible**, and **GitHub Actions**.\n"
                "  - Manages **AWS**, **GCP**, and **Azure** infrastructure.\n"
                "- **DBMS & SQL** (80%):\n"
                "  - Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB).\n"
                "- **IoT** (80%):\n"
                "  - Develops real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**.\n\n"
                "His **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning** keep him at the forefront of technology."
            )
        },
        {
            "content": "How good is Siddharamayya in AI/ML?",
            "response": (
                "**Siddharamayya Mathapati** is highly proficient in **AI/ML** (85%), with extensive experience in **large language models (LLMs)**, **deep learning**, **generative AI**, and **MLOps**, making him a standout in the field.\n\n"
                "- **LLM Expertise**:\n"
                "  - Fine-tuned **LLMs** using **QLoRA** and **LoRA** at **Capgemini**, optimizing efficiency for **finance**, **healthcare**, and **cybersecurity**.\n"
                "  - Implemented **RLHF** to align models with human feedback, improving contextual accuracy by **20%**.\n"
                "  - Example: Developed **Stock Price Prediction** using **LLMs** with **Yahoo Finance** and **NewsAPI**.\n"
                "- **Deep Learning**:\n"
                "  - Designed **neural networks** for **NLP**, **computer vision**, and **predictive modeling** using **TensorFlow** and **PyTorch**.\n"
                "  - Built **Medical Assistant AI** for **medical image analysis** with **Google Generative AI**.\n"
                "- **RAG Applications**:\n"
                "  - Deployed **Retrieval-Augmented Generation** systems with **FAISS** and **ChromaDB**, enhancing response accuracy by **25%**.\n"
                "  - Example: **Student Study Assistant** chatbot using **LangChain** and **Gemini LLM**.\n"
                "- **MLOps**:\n"
                "  - Automated **model training**, **deployment**, and **monitoring** with **Kubernetes**, **Docker**, and **AWS SageMaker**.\n"
                "  - Built **retraining pipelines** with **drift detection**, reducing errors by **30%**.\n"
                "- **Certifications**:\n"
                "  - **LLM Engineering** and **Deep Learning Masterclass** (Udemy) validate his expertise.\n\n"
                "His work on projects like **Credit Risk Analysis** (90% accuracy with **CatBoostClassifier**) and **SQL Query Generator** demonstrates his ability to apply **AI/ML** to real-world problems."
            )
        },
        {
            "content": "What are Siddharamayya's projects?",
            "response": (
                "**Siddharamayya Mathapati** has a diverse portfolio of projects showcasing his expertise in **AI/ML**, **web development**, **NLP**, **IoT**, and **DevOps**. Below is a comprehensive overview:\n\n"
                "- **LinkedIn Activity Scraper** ([GitHub](https://github.com/mtptisid/linkedin-comment-poster)):\n"
                "  - A **Python-based tool** using **Selenium** and **BeautifulSoup** to extract **LinkedIn** activity (posts, likes, shares, comments).\n"
                "  - Features **dynamic content handling** and **ethical scraping** for **social media analytics**.\n"
                "- **Recruitment Automation** ([GitHub](https://github.com/mtptisid/recruitment-solution)):\n"
                "  - An **AI-driven solution** for extracting job data and generating **cold emails** using **NLP** and **web scraping**, reducing effort by **80%**.\n"
                "- **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)):\n"
                "  - An **AI-powered finance agent** using **LLMs**, integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy.\n"
                "- **Credit Risk Analysis** ([GitHub](https://github.com/mtptisid/credit-risk-analysis)):\n"
                "  - A **machine learning model** with **CatBoostClassifier** for credit card default prediction, achieving **90% accuracy**.\n"
                "- **Medical Assistant AI** ([GitHub](https://github.com/mtptisid/medical-assistant-ai)):\n"
                "  - A **web application** using **Google Generative AI** for **medical image analysis** with a **chat interface**.\n"
                "- **SQL Query Generator** ([GitHub](https://github.com/mtptisid/simple-sql-gemini)):\n"
                "  - A **Jupyter Notebook-based tool** using **Google Gemini LLM** and **LangChain** for natural language **SQL query** generation.\n"
                "- **Student Study Assistant** ([GitHub](https://github.com/mtptisid/student-study-assistant)):\n"
                "  - A **chatbot** with **LangChain**, **Gemini LLM**, and **PDF processing** for student queries.\n"
                "- **Smart School Bus Tracking** ([GitHub](https://github.com/mtptisid/smart-bus-tracking)):\n"
                "  - An **IoT system** using **RFID**, **GPS**, and **Raspberry Pi** for real-time student tracking.\n"
                "- **Car Security System** ([GitHub](https://github.com/mtptisid/car-security-system)):\n"
                "  - A **real-time AI system** with **IoT**, **facial recognition**, and **RFID** for vehicle security.\n\n"
                "Explore more on his **[GitHub](https://github.com/mtptisid)** and **[Portfolio](https://mtptisid.github.io)**."
            )
        },
        {
            "content": "What is Siddharamayya's GitHub page?",
            "response": (
                "**Siddharamayya Mathapati**’s **GitHub** profile showcases his technical expertise in **AI/ML**, **web development**, **DevOps**, and **IoT**.\n\n"
                "- **[GitHub](https://github.com/mtptisid)**:\n"
                "  - Features projects like **Stock Price Prediction**, **Medical Assistant AI**, **Smart School Bus Tracking**, and **LinkedIn Activity Scraper**.\n"
                "  - Demonstrates proficiency in **Python**, **LLMs**, **IoT**, and **web scraping**.\n"
                "  - Includes **DevOps** workflows with **Docker**, **Kubernetes**, and **GitHub Actions**.\n"
                "  - Reflects **open-source contributions** and collaborative development.\n\n"
                "Visit his **[Portfolio](https://mtptisid.github.io)** for a curated overview."
            )
        },
        {
            "content": "What is Siddharamayya's portfolio page?",
            "response": (
                "**Siddharamayya Mathapati**’s **portfolio** is a professional showcase of his work in **AI/ML**, **web development**, **DevOps**, and **IoT**.\n\n"
                "- **[Portfolio](https://mtptisid.github.io)**:\n"
                "  - Highlights projects like **Stock Price Prediction**, **Medical Assistant AI**, and **Smart School Bus Tracking**.\n"
                "  - Showcases **technical skills** in **Python**, **React**, **Kubernetes**, and **AWS**.\n"
                "  - Provides insights into his **Udemy certifications** and **4+ years** of experience.\n"
                "- **Complementary Links**:\n"
                "  - **[GitHub](https://github.com/mtptisid)**: Detailed project repositories.\n"
                "  - **[LinkedIn](https://linkedin.com/in/siddharamayya)**: Professional achievements and endorsements.\n\n"
                "The portfolio is an excellent resource for exploring his **innovative solutions** and **technical expertise**."
            )
        },
        {
            "content": "What are Siddharamayya's key skills?",
            "response": (
                "**Siddharamayya Mathapati** possesses a robust skill set across **AI/ML**, **DevOps**, **web development**, **cloud engineering**, and **IoT**, enabling him to deliver **scalable**, **innovative** solutions.\n\n"
                "- **Python** (90%):\n"
                "  - Expert in **AI/ML**, **web development**, and **automation** with **TensorFlow**, **PyTorch**, **LangChain**, and **FastAPI**.\n"
                "- **Deep Learning** & **Machine Learning/AI** (85%):\n"
                "  - Designs **neural networks** and **LLMs** for **NLP**, **computer vision**, and **predictive modeling**.\n"
                "- **Web Development** (95%):\n"
                "  - Builds responsive applications with **React**, **Flask**, **Django**, and **Streamlit**.\n"
                "- **DevOps**:\n"
                "  - Automates **CI/CD pipelines** with **Docker**, **Kubernetes**, **Ansible**, and **GitHub Actions**.\n"
                "- **Cloud Engineering** (85%):\n"
                "  - Optimizes **AWS** (SageMaker, EMR, Lambda), **GCP**, and **Azure** for performance.\n"
                "- **DBMS & SQL** (80%):\n"
                "  - Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB).\n"
                "- **IoT** (80%):\n"
                "  - Develops real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**.\n"
                "- **Other Languages**:\n"
                "  - Proficient in **C++**, **Ruby**, **JavaScript**, **Golang**, and **Bash**.\n\n"
                "His skills are validated by **Udemy certifications** and applied in projects like **Medical Assistant AI** and **Stock Price Prediction**."
            )
        },
        {
            "content": "What is Siddharamayya’s educational background?",
            "response": (
                "**Siddharamayya Mathapati** has a solid academic foundation in **computer applications**, equipping him for **AI/ML**, **software development**, and **DevOps**.\n\n"
                "- **MCA - Master of Computer Applications**:\n"
                "  - **Institution**: Acharya Institute of Technology, Bangalore\n"
                "  - **Duration**: July 2018 - September 2020\n"
                "  - **Score**: CGPA 7.2/10\n"
                "  - Focused on **advanced programming**, **database systems**, and **software engineering**.\n"
                "- **BCA - Bachelor of Computer Applications**:\n"
                "  - **Institution**: B K College, Chikodi\n"
                "  - **Duration**: July 2015 - June 2018\n"
                "  - **Score**: CGPA 6.8/10\n"
                "  - Covered **programming fundamentals**, **web development**, and **database management**.\n\n"
                "His education, combined with **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning**, supports his technical expertise."
            )
        },
        {
            "content": "What is Siddharamayya’s current role at Capgemini?",
            "response": (
                "**Siddharamayya Mathapati** serves as a **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025), leading **AI/ML** and **MLOps** initiatives.\n\n"
                "- **Responsibilities**:\n"
                "  - Fine-tunes **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing memory usage by **30%**.\n"
                "  - Designs **RAG applications** with **FAISS** and **ChromaDB**, improving accuracy by **25%**.\n"
                "  - Automates **MLOps pipelines** using **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**.\n"
                "  - Develops **real-time inference pipelines** with **TensorRT** and **ONNX** for low-latency applications.\n"
                "  - Migrates legacy ML workloads to **Spark** and **Kubernetes**, enhancing scalability by **40%**.\n"
                "  - Builds **retraining pipelines** with **drift detection**, reducing errors by **30%**.\n"
                "  - Mentors junior engineers and supervises research internships.\n\n"
                "His role showcases his ability to deliver **high-impact**, **scalable** solutions."
            )
        },
        {
            "content": "What languages does Siddharamayya speak?",
            "response": (
                "**Siddharamayya Mathapati** is fluent in **seven languages**, enhancing his ability to collaborate across diverse teams in **India** and beyond.\n\n"
                "- **Kannada**:\n"
                "  - Native language; fluent in speaking, reading, and writing, used in **Karnataka**.\n"
                "- **English**:\n"
                "  - Professional fluency; excels in technical writing, presentations, and global collaboration.\n"
                "- **Hindi**:\n"
                "  - Fluent; skilled in verbal and written communication for cross-regional interactions.\n"
                "- **Telugu**:\n"
                "  - Conversational fluency; comfortable in professional and casual discussions.\n"
                "- **Tamil**:\n"
                "  - Conversational fluency; adept in everyday and technical contexts.\n"
                "- **Marathi**:\n"
                "  - Intermediate fluency; effective in professional and social settings in **Maharashtra**.\n"
                "- **Malayalam**:\n"
                "  - Basic proficiency; can engage in simple conversations in **Kerala**.\n\n"
                "His linguistic versatility supports his work in industries like **finance**, **healthcare**, and **education**."
            )
        },
        {
            "content": "How can I contact Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** is accessible for professional inquiries and collaborations.\n\n"
                "- **[Email](mailto:msidrm455@gmail.com)**:\n"
                "  - Preferred for formal communication, project proposals, or job opportunities.\n"
                "- **Phone**: +91 97406 71620\n"
                "  - Available for calls or messages to discuss **AI/ML**, **DevOps**, or **IoT** projects.\n"
                "- **Address**: #372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India\n"
                "  - Suitable for official correspondence or in-person meetings (by appointment).\n"
                "- **[LinkedIn](https://linkedin.com/in/siddharamayya)**:\n"
                "  - Ideal for networking and viewing professional updates.\n\n"
                "He is responsive and open to discussing **innovative** opportunities."
            )
        },
        {
            "content": "What are Siddharamayya’s interests in AI?",
            "response": (
                "**Siddharamayya Mathapati** is deeply passionate about **AI** and its transformative potential, with specific interests that drive his professional work.\n\n"
                "- **Generative AI & LLM Development**:\n"
                "  - Focuses on building and fine-tuning **LLMs** for applications like **text generation** and **image analysis** (e.g., **Medical Assistant AI**).\n"
                "- **AI-Powered Financial Modeling**:\n"
                "  - Develops **predictive models** for stock market analysis, as seen in **Stock Price Prediction** with **LLMs**.\n"
                "- **NLP & Chatbots**:\n"
                "  - Creates autonomous **AI agents** for natural language understanding, like the **Student Study Assistant** chatbot.\n"
                "- **Cloud-Native MLOps**:\n"
                "  - Integrates **AI** with **cloud platforms** (AWS, GCP, Azure) for scalable deployments, automating **MLOps pipelines**.\n"
                "- **IoT & Real-Time AI**:\n"
                "  - Combines **AI** with **IoT** for systems like **Smart School Bus Tracking** and **Car Security System**.\n\n"
                "His interests align with his **4+ years** of experience and **Udemy certifications**, positioning him as a leader in **AI innovation**."
            )
        },
        {
            "content": "What generative AI projects has Siddharamayya worked on?",
            "response": (
                "**Siddharamayya Mathapati** has worked on several **generative AI** projects, leveraging **LLMs** and **deep learning** to create innovative solutions.\n\n"
                "- **Medical Assistant AI** ([GitHub](https://github.com/mtptisid/medical-assistant-ai)):\n"
                "  - A **web application** using **Google Generative AI** to analyze **medical images** and provide diagnostic insights via a **chat interface**.\n"
                "  - Enhances support for medical professionals with real-time analysis.\n"
                "- **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)):\n"
                "  - An **AI-powered finance agent** using **LLMs** to generate stock price predictions and market insights.\n"
                "  - Integrates **Yahoo Finance** and **NewsAPI**, improving accuracy by **25%**.\n"
                "- **Student Study Assistant** ([GitHub](https://github.com/mtptisid/student-study-assistant)):\n"
                "  - A **chatbot** built with **LangChain** and **Gemini LLM** to generate context-aware responses from **PDF documents**.\n"
                "  - Supports students with study-related queries.\n"
                "- **SQL Query Generator** ([GitHub](https://github.com/mtptisid/simple-sql-gemini)):\n"
                "  - A **Jupyter Notebook-based tool** using **Google Gemini LLM** to generate **SQL queries** from natural language inputs.\n"
                "  - Employs **few-shot learning** for accuracy.\n\n"
                "These projects highlight his expertise in **generative AI** and **NLP**, supported by his **Udemy certification** in **LLM Engineering**."
            )
        },
        {
            "content": "What DevOps tools does Siddharamayya use?",
            "response": (
                "**Siddharamayya Mathapati** is proficient in a wide range of **DevOps tools**, enabling him to automate **CI/CD pipelines**, manage **cloud infrastructure**, and optimize deployments.\n\n"
                "- **Docker**:\n"
                "  - Creates **containerized applications** for portability and scalability.\n"
                "  - Used at **Capgemini** to deploy **ML models**.\n"
                "- **Kubernetes**:\n"
                "  - Orchestrates **containerized workloads** for high availability.\n"
                "  - Migrated legacy ML models to **Kubernetes** at **Capgemini**, improving scalability by **40%**.\n"
                "- **Ansible**:\n"
                "  - Automates **configuration management** and **server provisioning**.\n"
                "  - Streamlined **infrastructure setup** for **AI workloads**.\n"
                "- **Jenkins**:\n"
                "  - Builds **CI/CD pipelines** for automated testing and deployment.\n"
                "  - Integrated with **GitHub Actions** for **ML model** deployments.\n"
                "- **GitHub Actions**:\n"
                "  - Automates **workflows** for building, testing, and deploying code.\n"
                "  - Used in projects like **Stock Price Prediction**.\n"
                "- **AWS SageMaker**:\n"
                "  - Deploys and manages **ML models** in the cloud.\n"
                "  - Automated **MLOps pipelines** at **Capgemini**.\n"
                "- **MLflow**:\n"
                "  - Tracks **ML experiments** and manages model lifecycles.\n"
                "- **Airflow**:\n"
                "  - Schedules and monitors **data pipelines** for **ML workflows**.\n\n"
                "His **Udemy certification** in **MLOps** enhances his ability to leverage these tools effectively."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with LLMs?",
            "response": (
                "**Siddharamayya Mathapati** has extensive experience with **large language models (LLMs)**, focusing on **fine-tuning**, **deployment**, and **application development**.\n\n"
                "- **Fine-Tuning**:\n"
                "  - At **Capgemini**, fine-tuned **LLMs** using **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing memory usage by **30%**.\n"
                "  - Applied **RLHF** to align models with human feedback, improving accuracy by **20%**.\n"
                "- **RAG Applications**:\n"
                "  - Designed **Retrieval-Augmented Generation** systems with **FAISS** and **ChromaDB** for domain-specific tasks, enhancing response accuracy by **25%**.\n"
                "  - Example: **Student Study Assistant** chatbot.\n"
                "- **Project Highlights**:\n"
                "  - **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)): Used **LLMs** for market trend analysis.\n"
                "  - **SQL Query Generator** ([GitHub](https://github.com/mtptisid/simple-sql-gemini)): Integrated **Google Gemini LLM** for **SQL query** generation.\n"
                "  - **Medical Assistant AI** ([GitHub](https://github.com/mtptisid/medical-assistant-ai)): Leveraged **Google Generative AI** for diagnostic insights.\n"
                "- **Tools & Frameworks**:\n"
                "  - Proficient in **LangChain**, **Hugging Face**, and **PyTorch** for **LLM** development.\n"
                "  - Uses **AWS SageMaker** and **Kubernetes** for **LLM** deployments.\n"
                "- **Certifications**:\n"
                "  - **LLM Engineering** (Udemy) validates his expertise in **LLM** workflows.\n\n"
                "His work demonstrates a deep understanding of **LLM** optimization and real-world applications."
            )
        },
        {
            "content": "What AI-powered finance projects has Siddharamayya built?",
            "response": (
                "**Siddharamayya Mathapati** has developed impactful **AI-powered finance projects**, leveraging **LLMs** and **machine learning** for financial decision-making.\n\n"
                "- **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)):\n"
                "  - An **AI-powered finance agent** using **LLMs** to predict stock prices and analyze market trends.\n"
                "  - Integrates **Yahoo Finance** for stock data and **NewsAPI** for sentiment analysis, improving accuracy by **25%**.\n"
                "  - Supports investors with data-driven insights.\n"
                "- **Credit Risk Analysis** ([GitHub](https://github.com/mtptisid/credit-risk-analysis)):\n"
                "  - A **machine learning model** using **CatBoostClassifier** to predict credit card defaults.\n"
                "  - Achieves **90% accuracy**, aiding financial institutions in **risk management**.\n"
                "  - Features **data preprocessing** and **feature engineering** for robust predictions.\n"
                "- **Capgemini Finance Projects**:\n"
                "  - Fine-tuned **LLMs** for **financial analytics**, optimizing models for **contextual accuracy** in budgeting and forecasting.\n"
                "  - Deployed **RAG applications** with **vector databases** for financial document retrieval, enhancing efficiency by **30%**.\n\n"
                "These projects highlight his ability to apply **AI/ML** to solve complex financial challenges."
            )
        },
        {
            "content": "What is Siddharamayya’s tech stack?",
            "response": (
                "**Siddharamayya Mathapati**’s **tech stack** is comprehensive, covering **AI/ML**, **web development**, **DevOps**, **cloud engineering**, and **IoT**.\n\n"
                "- **Programming Languages**:\n"
                "  - **Python**, **C++**, **Ruby**, **JavaScript**, **Golang**, **Bash**.\n"
                "- **AI/ML Frameworks**:\n"
                "  - **TensorFlow**, **PyTorch**, **LangChain**, **Hugging Face**, **PySpark**.\n"
                "- **Web Development**:\n"
                "  - **Flask**, **Django**, **FastAPI**, **React**, **Streamlit**.\n"
                "- **DevOps Tools**:\n"
                "  - **Docker**, **Kubernetes**, **Ansible**, **Jenkins**, **GitHub Actions**, **MLflow**, **Airflow**.\n"
                "- **Cloud Platforms**:\n"
                "  - **AWS** (SageMaker, EMR, Lambda), **GCP**, **Azure**.\n"
                "- **Databases**:\n"
                "  - **MySQL**, **PostgreSQL**, **MongoDB**, **ChromaDB**, **FAISS**.\n"
                "- **IoT Technologies**:\n"
                "  - **Raspberry Pi**, **ESP32**, **RFID**, **GPS**.\n"
                "- **Other Tools**:\n"
                "  - **Git**, **VS Code**, **RabbitMQ**, **Postman**, **LangGraph**, **Firebase**.\n\n"
                "His **tech stack** supports his work on projects like **Medical Assistant AI**, **Stock Price Prediction**, and **Smart School Bus Tracking**."
            )
        },
        {
            "content": "How does Siddharamayya use Docker and Kubernetes?",
            "response": (
                "**Siddharamayya Mathapati** leverages **Docker** and **Kubernetes** to automate **CI/CD pipelines**, deploy **ML models**, and manage **cloud-native** applications.\n\n"
                "- **Docker**:\n"
                "  - Creates **containerized environments** for **AI/ML** and **web applications**, ensuring portability.\n"
                "  - Packages **ML models** with dependencies for consistent deployments.\n"
                "  - Example at **Capgemini**: Containerized **LLM inference pipelines** for **low-latency** applications.\n"
                "- **Kubernetes**:\n"
                "  - Orchestrates **containerized workloads** for scalability and high availability.\n"
                "  - Migrated legacy ML models to **Kubernetes** at **Capgemini**, improving scalability by **40%**.\n"
                "  - Manages **MLOps pipelines**, automating **model training** and **deployment**.\n"
                "  - Example: Deployed **RAG applications** with **vector databases** on **Kubernetes** clusters.\n"
                "- **Integration**:\n"
                "  - Uses **Docker** to build images and **Kubernetes** to orchestrate them in **AWS** and **GCP**.\n"
                "  - Integrates with **Jenkins** and **GitHub Actions** for **CI/CD** automation.\n"
                "- **Monitoring**:\n"
                "  - Implements **Prometheus** and **Grafana** for **real-time monitoring** of **Kubernetes** clusters.\n\n"
                "His **Udemy certification** in **MLOps** enhances his expertise in **Docker** and **Kubernetes**."
            )
        },
        {
            "content": "What are Siddharamayya’s contributions to web development?",
            "response": (
                "**Siddharamayya Mathapati** has made significant contributions to **web development** (95% proficiency), building responsive, user-friendly applications.\n\n"
                "- **Projects**:\n"
                "  - **Medical Assistant AI** ([GitHub](https://github.com/mtptisid/medical-assistant-ai)):\n"
                "    - A **web application** with **React** and **Flask** for **medical image analysis** and **chat-based** diagnostics.\n"
                "  - **Student Study Assistant** ([GitHub](https://github.com/mtptisid/student-study-assistant)):\n"
                "    - A **chatbot interface** using **Streamlit** for student queries from **PDF documents**.\n"
                "- **Technologies**:\n"
                "  - **Frontend**: **React**, **Bootstrap**, **HTML**, **CSS** for responsive UIs.\n"
                "  - **Backend**: **Flask**, **Django**, **FastAPI** for robust APIs.\n"
                "  - **Deployment**: Uses **Docker** and **Kubernetes** for scalable web apps.\n"
                "- **Capgemini Contributions**:\n"
                "  - Developed **web-based dashboards** for **ML model** monitoring using **Streamlit**.\n"
                "  - Automated **web automation scripts** with **Selenium** for testing and data extraction.\n"
                "- **X-Cencia Internship**:\n"
                "  - Built **web interfaces** for **IoT systems**, integrating **real-time data** from **Raspberry Pi**.\n\n"
                "His work combines **UX design**, **backend development**, and **DevOps** for seamless web solutions."
            )
        },
        {
            "content": "Visit Sid's portfolio",
            "response": (
                "To explore **Siddharamayya Mathapati**’s professional work and projects, visit his **portfolio**:\n\n"
                "- **[Portfolio](https://mtptisid.github.io)**:\n"
                "  - Showcases projects like **Stock Price Prediction**, **Medical Assistant AI**, and **Smart School Bus Tracking**.\n"
                "  - Highlights **AI/ML**, **web development**, **DevOps**, and **IoT** expertise.\n"
                "  - Includes details on **Udemy certifications** and **4+ years** of experience.\n"
                "- **Additional Links**:\n"
                "  - **[GitHub](https://github.com/mtptisid)**: View project repositories.\n"
                "  - **[LinkedIn](https://linkedin.com/in/siddharamayya)**: Connect for professional updates.\n\n"
                "His **portfolio** is a comprehensive resource for his **innovative** contributions."
            )
        },
        {
            "content": "Download Resume",
            "response": (
                "**Siddharamayya Mathapati**’s resume details his **4+ years** of experience in **AI/ML**, **MLOps**, **DevOps**, and **web development**. For a downloadable version, contact him directly:\n\n"
                "- **[Email](mailto:msidrm455@gmail.com)**:\n"
                "  - Request a PDF copy of his resume, tailored to **AI/ML**, **MLOps**, or **DevOps** roles.\n"
                "- **Phone**: +91 97406 71620\n"
                "  - Discuss resume details or specific role requirements.\n"
                "- **[LinkedIn](https://linkedin.com/in/siddharamayya)**:\n"
                "  - View his professional summary and endorsements.\n\n"
                "Alternatively, explore his **[Portfolio](https://mtptisid.github.io)** or **[GitHub](https://github.com/mtptisid)** for project details and skills."
            )
        },
        {
            "content": "What certifications does Siddharamayya have?",
            "response": (
                "**Siddharamayya Mathapati** has earned **Udemy certifications** that validate his expertise in **AI/ML**, **MLOps**, and **data science**.\n\n"
                "- **LLM Engineering: Master AI, Large Language Models & Agents**:\n"
                "  - Covers **LLM fine-tuning**, **prompt engineering**, and **AI agents**.\n"
                "  - Applied in **Stock Price Prediction** and **Medical Assistant AI**.\n"
                "- **MLOps Bootcamp: Mastering AI Operations for Success**:\n"
                "  - Focuses on **CI/CD pipelines**, **model monitoring**, and **cloud deployments**.\n"
                "  - Used at **Capgemini** for **MLOps** automation.\n"
                "- **Python for Data Analysis & Visualization**:\n"
                "  - Covers **Pandas**, **NumPy**, and **Matplotlib** for data processing.\n"
                "  - Applied in **Credit Risk Analysis**.\n"
                "- **Deep Learning Masterclass with TensorFlow 2**:\n"
                "  - Explores **neural network design** and **computer vision**.\n"
                "  - Used in **Medical Assistant AI** for **image analysis**.\n\n"
                "These certifications complement his **4+ years** of experience and project portfolio."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with IoT?",
            "response": (
                "**Siddharamayya Mathapati** has significant experience in **IoT** (80% proficiency), developing real-time systems for **smart applications**.\n\n"
                "- **X-Cencia Internship** (February 2020 - April 2021):\n"
                "  - Developed a **Smart School Bus Tracking System** using **RFID**, **GPS**, and **Raspberry Pi** for real-time student monitoring.\n"
                "  - Built a **Car Security System** with **IoT**, **facial recognition**, and **RFID** for vehicle authentication.\n"
                "  - Designed **automated data pipelines** for **sensor data** processing from **ESP32** devices.\n"
                "- **Projects**:\n"
                "  - **Smart School Bus Tracking** ([GitHub](https://github.com/mtptisid/smart-bus-tracking)):\n"
                "    - Ensures **real-time location tracking** with a **web interface** for parents and schools.\n"
                "  - **Car Security System** ([GitHub](https://github.com/mtptisid/car-security-system)):\n"
                "    - Integrates **AI-based facial recognition** with **IoT** for enhanced security.\n"
                "- **Technologies**:\n"
                "  - Proficient in **Raspberry Pi**, **ESP32**, **RFID**, **GPS**, and **MQTT** for **IoT** communication.\n"
                "  - Uses **Python** and **Flask** for **IoT** web interfaces.\n"
                "- **Capgemini Contributions**:\n"
                "  - Integrated **IoT data** with **ML models** for real-time analytics in **smart systems**.\n\n"
                "His **IoT** expertise enhances his **AI/ML** and **web development** capabilities."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with MLOps?",
            "response": (
                "**Siddharamayya Mathapati** has extensive **MLOps** experience, automating **machine learning** workflows for scalability and efficiency.\n\n"
                "- **Capgemini** (May 2021 - April 2025):\n"
                "  - Automated **MLOps pipelines** with **Kubernetes**, **Docker**, and **AWS SageMaker**, reducing deployment time by **50%**.\n"
                "  - Built **model retraining pipelines** with **drift detection**, cutting prediction errors by **30%**.\n"
                "  - Developed **real-time monitoring** systems with **Prometheus** and **Grafana** for **model performance**.\n"
                "  - Migrated legacy ML workloads to **Spark** and **Kubernetes**, improving scalability by **40%**.\n"
                "- **Tools**:\n"
                "  - **MLflow**: Tracks **ML experiments** and manages model lifecycles.\n"
                "  - **Airflow**: Schedules **data pipelines** for **ML workflows**.\n"
                "  - **Jenkins** & **GitHub Actions**: Automates **CI/CD** for **ML deployments**.\n"
                "- **Projects**:\n"
                "  - **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)):\n"
                "    - Deployed **LLM-based models** with **MLOps** automation.\n"
                "  - **Credit Risk Analysis** ([GitHub](https://github.com/mtptisid/credit-risk-analysis)):\n"
                "    - Managed model lifecycle with **MLflow**.\n"
                "- **Certification**:\n"
                "  - **MLOps Bootcamp** (Udemy) validates his expertise in **MLOps** workflows.\n\n"
                "His **MLOps** skills ensure robust, scalable **ML** deployments."
            )
        }
    ]

    system_prompt = (
        "You are an AI assistant with comprehensive knowledge about **Siddharamayya Mathapati**, a highly skilled **AI/ML Engineer** and **Senior Software Engineer**. "
        "Your responses must be in **Markdown format**, using **bold** for emphasis, **code blocks** for snippets, **bulleted lists** for clarity, and **[name](link)** for URLs. "
        "Provide detailed, accurate answers about Siddharamayya using the **profile data** and **few-shot examples**. For unrelated questions, leverage **web search results** or **general knowledge**, maintaining the same formatting standards. "
        "Incorporate **chat history** for context-aware responses.\n\n"
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
        "\n".join([f"- **{s['name']}** ({s.get('proficiency', 'Proficient')}): {s.get('description', '')}" for s in PROFILE_DATA['skills']]) + "\n\n"
        "**Experience**:\n" +
        "\n".join([f"- **{e['title']}** at **{e['company']}** ({e['duration']}):\n  - {e['description']}" for e in PROFILE_DATA['experience']]) + "\n\n"
        "**Education**:\n" +
        "\n".join([f"- **{e['degree']}** at **{e['institution']}** ({e['duration']}): {e['score']}" for e in PROFILE_DATA['education']]) + "\n\n"
        "**Projects**:\n" +
        "\n".join([f"- **{p['category']}**: {p['description']} ([Link]({p['link']}))" for p in PROFILE_DATA['projects']]) + "\n\n"
        "**Certifications**:\n" +
        "\n".join([f"- **{c}**" for c in PROFILE_DATA['certifications']]) + "\n\n"
        "**Contact**:\n" +
        f"- **[Email](mailto:{PROFILE_DATA['contact']['email']})**\n"
        f"- **Phone**: {PROFILE_DATA['contact']['phone']}\n"
        f"- **Address**: {PROFILE_DATA['contact']['address']}\n\n"
        "**Online Presence**:\n" +
        f"- **[GitHub]({PROFILE_DATA['online_presence']['github']})**\n"
        f"- **[Portfolio]({PROFILE_DATA['online_presence']['portfolio']})**\n"
        f"- **[LinkedIn]({PROFILE_DATA['online_presence']['linkedin']})**\n\n"
        "**Interests**:\n" +
        "\n".join([f"- {i}" for i in PROFILE_DATA['interests']]) + "\n\n"
        "**Tech Stack**:\n" +
        "\n".join([f"- {t}" for t in PROFILE_DATA['tech_stack']]) + "\n\n"
        "**Few-Shot Examples**:\n" +
        "\n".join([f"**Q**: {ex['content']}\n**A**: {ex['response']}" for ex in FEW_SHOT_EXAMPLES]) + "\n\n"
        "**Instructions**:\n"
        "- Always respond in **Markdown format** with proper spacing and structure.\n"
        "- Use **bold** for key terms (e.g., names, roles, technologies).\n"
        "- Wrap code snippets in triple backticks (e.g., ```python ... ```).\n"
        "- Use `-` for bulleted lists, with one item per line and blank lines before/after.\n"
        "- Format URLs as **[name](link)** (e.g., **[GitHub](https://github.com/mtptisid)**).\n"
        "- Ensure readability with clear section breaks (e.g., `---` for separators).\n"
        "- Provide detailed, context-aware responses, leveraging **profile data** and **chat history**.\n"
        "- For unrelated questions, use **web search results** or **general knowledge**, maintaining the same formatting.\n\n"
        "**Chat History**:\n" +
        "\n".join([f"**{'Bot' if msg.is_bot else 'User'}**: {msg.content}" for msg in SESSIONS.get(session_id, {"messages": []})["messages"]]) + "\n\n"
        "**Question**:\n" +
        message.content
    )

    chat_history = [
        {"role": "system", "content": system_prompt}
    ]

    response_content = ""
    tool_used = None

    if message.tool == "SearchWeb":
        tool_used = "SearchWeb"
        search_result = await search_web(message.content, message.model)
        augmented_prompt = (
            f"{system_prompt}\n"
            f"**Web Search Results**:\n{search_result}\n"
            "Provide a detailed response in **Markdown**, using **[name](link)** for all URLs in the search results or elsewhere."
        )
        chat_history = [
            {"role": "system", "content": augmented_prompt}
        ]

    try:
        response_content = await ai_manager.get_response(message.model, chat_history)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"AI service error: {str(e)}")

    response_content = clean_text(response_content)

    bot_message = SessionMessage(
        content=response_content,
        is_bot=True,
        timestamp=datetime.utcnow(),
        tool_used=tool_used
    )
    SESSIONS[session_id]["messages"].append(bot_message)

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
