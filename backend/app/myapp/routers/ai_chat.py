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
                "**Siddharamayya Mathapati** is a seasoned **AI/ML Engineer** and **Senior Software Engineer** with over **4 years** of experience in **AI**, **MLOps**, **DevOps**, **web development**, and **system automation**. "
                "He specializes in **large language models (LLMs)**, **generative AI**, **RAG applications**, and **cloud-native solutions**, delivering scalable systems for **finance**, **healthcare**, **agriculture**, and **IT infrastructure**. "
                "At **Capgemini**, he led **MLOps pipelines**, optimized **LLM fine-tuning**, and automated **infrastructure management**. His diverse portfolio includes **IoT**, **Ansible automation**, and **bash scripting**, complemented by **Udemy certifications**."
            ),
            "availability": "Immediately available for new opportunities as of May 2025."
        },
        "skills": [
            {"name": "Python", "proficiency": "90%", "description": "Expert in **AI/ML**, **web development**, **automation**, and **system scripting** with **TensorFlow**, **PyTorch**, **LangChain**, **FastAPI**, and **Scrapy**."},
            {"name": "Deep Learning", "proficiency": "85%", "description": "Designs **neural networks** for **NLP**, **computer vision**, and **predictive modeling** using **TensorFlow** and **PyTorch**."},
            {"name": "Machine Learning/AI", "proficiency": "85%", "description": "Builds **LLMs**, **RAG applications**, and **predictive models** with **QLoRA**, **RLHF**, and **Random Forest**."},
            {"name": "Data Science", "proficiency": "85%", "description": "Skilled in **data preprocessing**, **visualization**, and **analysis** with **Pandas**, **NumPy**, and **Matplotlib**."},
            {"name": "DevOps", "tools": ["Docker", "Kubernetes", "Ansible", "Jenkins", "GitHub Actions"], "description": "Automates **CI/CD pipelines**, **cloud infrastructure**, and **Ansible-based deployments**."},
            {"name": "Web Development", "proficiency": "95%", "description": "Develops responsive applications with **React**, **Flask**, **Django**, **FastAPI**, and **Streamlit**."},
            {"name": "DBMS & SQL", "proficiency": "80%", "description": "Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB)."},
            {"name": "Cloud Engineering", "proficiency": "85%", "description": "Optimizes **AWS**, **GCP**, and **Azure** for cost and performance."},
            {"name": "IoT", "proficiency": "80%", "description": "Builds real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**."},
            {"name": "System Administration", "proficiency": "85%", "description": "Automates **Linux** system management with **Ansible**, **bash**, and **systemd**."}
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
                    "Automated **Linux system administration** using **Ansible** and **bash**, optimizing server performance. "
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
                    "Designed **automated data pipelines** for **sensor data** processing. "
                    "Created **bash scripts** for system monitoring and resource management."
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
                "category": "SysAdmin-GPT: AI-Powered Linux System Management",
                "description": (
                    "A **BERT-based classification model** fine-tuned on **Red Hat Enterprise Linux (RHEL)** documentation to classify queries into categories (e.g., Security, Networking, Storage). "
                    "Integrated with a **chatbot interface** for automated IT support, achieving high accuracy in query resolution. "
                    "Deployable as a **Linux system management tool** for real-time assistance."
                ),
                "link": "[SysAdmin-GPT](https://siddharamayya.in/projects)"
            },
            {
                "category": "Workout & Fitness Tracker ML Model",
                "description": (
                    "A **machine learning model** analyzing **10,000+ workout records** to predict workout efficiency based on health metrics. "
                    "Deployed with a **Streamlit UI** for interactive, real-time fitness insights, helping users optimize routines."
                ),
                "link": "[Workout Tracker](https://siddharamayya.in/projects)"
            },
            {
                "category": "Crop Recommendation System",
                "description": (
                    "A **Random Forest Classifier** recommending crops based on soil characteristics and environmental factors. "
                    "Features a **Streamlit UI** for farmers to input data, promoting **sustainable farming** and **yield optimization**."
                ),
                "link": "[Crop Recommendation](https://siddharamayya.in/projects)"
            },
            {
                "category": "Blood Donation Prediction",
                "description": (
                    "A **Random Forest Classifier** predicting blood donation likelihood using **Taiwan blood transfusion data**. "
                    "Deployed with a **Streamlit app** for user-friendly interaction, analyzing features like recency and frequency."
                ),
                "link": "[Blood Donation Prediction]https://siddharamayya.in/projects)"
            },
            {
                "category": "Credit Default Prediction",
                "description": (
                    "A **machine learning model** predicting credit card defaults using simulated financial data. "
                    "Handles **imbalanced datasets** with techniques like SMOTE, achieving robust **creditworthiness assessments**."
                ),
                "link": "[Credit Default Prediction](https://siddharamayya.in/projects)"
            },
            {
                "category": "Heart Disease Prediction",
                "description": (
                    "A **web-based application** predicting **10-year heart disease risk** using a **machine learning model**. "
                    "Built with **Flask**, **Bootstrap**, and **scikit-learn**, featuring a responsive UI with gradient aesthetics."
                ),
                "link": "[Heart Disease Prediction](https://siddharamayya.in/projects)"
            },
            {
                "category": "CodeSage: AI-Powered Documentation Suite",
                "description": (
                    "An **LLM-powered assistant** using **RAG** to answer questions from custom **PDFs/docs** and generate context-aware code in **Python**, **JavaScript**, and more. "
                    "Ideal for developers needing **documentation-driven development**."
                ),
                "link": "[CodeSage](https://siddharamayya.in/projects)"
            },
            {
                "category": "Containerized AI-Powered Automation Lab",
                "description": (
                    "A **containerized platform** integrating **CodeServer**, **GitLab**, **AWX**, and **Linux hosts** with an **AI assistant** for **Ansible playbook** generation. "
                    "Features **LDAP/AD authentication**, **SSH key automation**, and **Docker/OpenShift** deployment on **GCP**."
                ),
                "link": "[Automation Lab](https://siddharamayya.in/projects)"
            },
            {
                "category": "NMON Analyser for Linux",
                "description": (
                    "A **Python tool** automating **NMON file** collection and analysis from remote **Linux servers** via **SSH**. "
                    "Generates **performance reports** in **.docx** and sends **email notifications** with insights and plots."
                ),
                "link": "[NMON Analyser](https://siddharamayya.in/projects)"
            },
            {
                "category": "Ansible Automation Platform API",
                "description": (
                    "A **Python script** for **Red Hat Ansible Automation Platform (AAP)** to launch **job templates** and extract **notifications**. "
                    "Supports **CI/CD pipeline** integration for **DevOps** automation."
                ),
                "link": "[Ansible AAP API](https://siddharamayya.in/projects)"
            },
            {
                "category": "VMware Automation for vSphere",
                "description": (
                    "A **Python script** using **pyVmomi** to fetch **VM details** (CPU, memory, OS) from **vCenter servers** and create **VMs** with predefined specs. "
                    "Simplifies **VM management** for **DevOps** teams."
                ),
                "link": "[VMware Automation](https://siddharamayya.in/projects)"
            },
            {
                "category": "Ansible Role: AWS VM Creation",
                "description": (
                    "An **Ansible script** creating **VMs** in **AWS EC2** and hosting jobs on **AWX**. "
                    "Automates **cloud infrastructure** provisioning for scalable deployments."
                ),
                "link": "[AWS VM Creation](https://siddharamayya.in/projects)"
            },
            {
                "category": "Ansible Role: Systemd Service",
                "description": (
                    "An **Ansible role** to create and manage **systemd services** on **Linux** systems. "
                    "Ensures reliable service deployment and configuration."
                ),
                "link": "[Systemd Service Role](https://siddharamayya.in/projects)"
            },
            {
                "category": "Ansible Role: Swap Management",
                "description": (
                    "An **Ansible role** to manage **swap space** on **Linux** machines, automating creation and extension of swap files."
                ),
                "link": "[Swap Management Role](https://siddharamayya.in/projects)"
            },
            {
                "category": "Ansible Role: Kernel Parameter Management",
                "description": (
                    "An **Ansible role** to configure **kernel parameters** in **sysctl** and **GRUB** for **Linux** systems, optimizing performance."
                ),
                "link": "[Kernel Parameter Role](https://siddharamayya.in/projects)"
            },
            {
                "category": "System Resource Usage Monitoring",
                "description": (
                    "A **bash script** monitoring **CPU**, **memory**, and **swap usage** on **Linux** systems. "
                    "Outputs data in **CSV** for analysis, featuring user-friendly table formatting."
                ),
                "link": "[Resource Monitoring](https://siddharamayya.in/projects)"
            },
            {
                "category": "Remote Swap Management",
                "description": (
                    "A **bash script** for managing **swap space** on remote **Linux servers** via **SSH**. "
                    "Automates swap file creation and reports usage in **CSV**."
                ),
                "link": "[Remote Swap Management](https://siddharamayya.in/projects)"
            },
            {
                "category": "Systemd Service Deployment",
                "description": (
                    "A **bash script** deploying **systemd services** across remote **Linux servers**, configuring **SELinux** and enabling services."
                ),
                "link": "[Systemd Deployment](https://siddharamayya.in/projects)"
            },
            {
                "category": "User Disabling Script",
                "description": (
                    "A **bash script** disabling user accounts on remote **Linux servers**, backing up system files, and sending **HTML/CSV reports** via **email**."
                ),
                "link": "[User Disabling](https://siddharamayya.in/projects)"
            },
            {
                "category": "Check Last Patch",
                "description": (
                    "A **bash script** collecting **system update** information (uptime, Red Hat version, kernel) from remote **Linux servers** via **SSH**, saving to **log files**."
                ),
                "link": "[Check Last Patch](https://siddharamayya.in/projects)"
            },
            {
                "category": "NFS/CIFS Share Monitoring",
                "description": (
                    "A **bash script** checking **NFS/CIFS share** mount status on **Linux servers** via **SSH**, verifying access and generating **CSV/HTML reports**."
                ),
                "link": "[Share Monitoring](https://siddharamayya.in/projects)"
            },
            {
                "category": "CIFS fstab Updater",
                "description": (
                    "A **bash script** updating **CIFS entries** in **/etc/fstab** on remote **Linux servers**, replacing usernames with **UIDs/GIDs** for proper mounts."
                ),
                "link": "[CIFS fstab Updater](https://siddharamayya.in/projects)"
            },
            {
                "category": "Django Tutor Application",
                "description": (
                    "A **Django-based web application** for expert talks, guiding beginners to create and host apps on **AWS EC2**. "
                    "Features user-friendly tutorials and deployment scripts."
                ),
                "link": "[Django Tutor](https://siddharamayya.in/projects)"
            },
            {
                "category": "FastAPI Microservice Application",
                "description": (
                    "A **FastAPI** and **MySQL** application deployed as a **Docker microservice** with a **React frontend**. "
                    "Demonstrates scalable **web development** and **containerization**."
                ),
                "link": "[FastAPI Microservice](https://siddharamayya.in/projects)"
            },
            {
                "category": "LinkedIn Activity Scraper",
                "description": (
                    "A **Python-based tool** using **Selenium** and **BeautifulSoup** to extract **LinkedIn** activity data (posts, likes, shares, comments). "
                    "Features **dynamic content handling** and **ethical scraping**."
                ),
                "link": "[LinkedIn Comment Poster](https://siddharamayya.in/projects)"
            },
            {
                "category": "Recruitment Automation",
                "description": (
                    "An **AI-driven solution** extracting job data from career pages and generating personalized **cold emails** using **NLP** and **web scraping**, reducing effort by **80%**."
                ),
                "link": "[Recruitment Solution](https://siddharamayya.in/projects)"
            },
            {
                "category": "Stock Price Prediction",
                "description": (
                    "An **AI-powered finance agent** using **LLMs** to predict stock prices, integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy."
                ),
                "link": "[Stock Price Prediction](https://siddharamayya.in/projects)"
            },
            {
                "category": "Credit Risk Analysis",
                "description": (
                    "A **machine learning model** using **CatBoostClassifier** to predict credit card defaults, achieving **90% accuracy**."
                ),
                "link": "[Credit Risk Analysis](https://siddharamayya.in/projects)"
            },
            {
                "category": "Medical Assistant AI",
                "description": (
                    "A **web application** using **Google Generative AI** for **medical image analysis** with a **chat interface** for diagnostic insights."
                ),
                "link": "[Medical Assistant AI](https://siddharamayya.in/projects)"
            },
            {
                "category": "SQL Query Generator",
                "description": (
                    "A **Jupyter Notebook-based tool** integrating **Google Gemini LLM** with **MySQL** using **LangChain** for natural language **SQL query** generation."
                ),
                "link": "[Simple SQL with Gemini](https://siddharamayya.in/projects)"
            },
            {
                "category": "Student Study Assistant",
                "description": (
                    "A **chatbot** using **LangChain**, **Gemini LLM**, and **PDF processing** to answer student queries from uploaded documents."
                ),
                "link": "[Student Study Assistant](https://siddharamayya.in/projects)"
            },
            {
                "category": "Smart School Bus Tracking",
                "description": (
                    "An **IoT-based system** using **RFID**, **GPS**, and **Raspberry Pi** for real-time student location monitoring."
                ),
                "link": "[Smart Bus Tracking](https://siddharamayya.in/projects)"
            },
            {
                "category": "Car Security System",
                "description": (
                    "A **real-time AI-based system** using **IoT**, **facial recognition**, and **RFID** for vehicle security."
                ),
                "link": "[Car Security System](https://siddharamayya.in/projects)"
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
            "IoT & Real-Time Systems",
            "System Automation & Infrastructure Management"
        ],
        "tech_stack": [
            "Python", "C++", "Ruby", "JavaScript", "Golang", "Bash",
            "TensorFlow", "PyTorch", "LangChain", "Hugging Face", "PySpark", "Scrapy",
            "Flask", "Django", "FastAPI", "React", "Streamlit", "Bootstrap",
            "Docker", "Kubernetes", "Ansible", "Jenkins", "GitHub Actions", "MLflow", "Airflow",
            "AWS (SageMaker, EMR, Lambda, EC2)", "GCP", "Azure", "OpenShift",
            "MySQL", "PostgreSQL", "MongoDB", "ChromaDB", "FAISS",
            "Raspberry Pi", "ESP32", "RFID", "GPS", "MQTT",
            "pyVmomi", "pyVim", "Selenium", "BeautifulSoup"
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
                "**Siddharamayya Mathapati** has over **4 years** of professional experience as a **Senior Software Engineer** and **AI/ML Engineer**, delivering innovative solutions in **AI**, **MLOps**, **DevOps**, **IoT**, and **system automation**.\n\n"
                "- **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025):\n"
                "  - Led **AI/ML** projects, fine-tuning **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing computational overhead by **30%**.\n"
                "  - Designed **RAG applications** using **FAISS** and **ChromaDB**, improving response accuracy by **25%**.\n"
                "  - Automated **MLOps pipelines** with **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**.\n"
                "  - Developed **real-time inference pipelines** with **TensorRT** and **ONNX** for low-latency applications.\n"
                "  - Optimized legacy ML workloads with **Spark** and **Kubernetes**, enhancing scalability by **40%**.\n"
                "  - Automated **Linux system administration** using **Ansible** and **bash**, optimizing server performance.\n"
                "  - Mentored junior engineers and led research internships.\n"
                "- **Project Intern** at **X-Cencia Technology Solution Limited India** (February 2020 - April 2021):\n"
                "  - Built **computer vision models** with **TensorFlow** for real-time applications.\n"
                "  - Developed **IoT solutions**, including a **smart school bus tracking system** with **RFID** and **GPS**, and a **car security system** with **facial recognition**.\n"
                "  - Tuned **Word2Vec** models for **recommendation systems**.\n"
                "  - Created **bash scripts** for system monitoring and resource management.\n\n"
                "His extensive experience underscores his ability to deliver **scalable**, **innovative** solutions across diverse domains."
            )
        },
        {
            "content": "When will Siddharamayya be available to join?",
            "response": (
                "**Siddharamayya Mathapati** is **immediately available** to join new opportunities starting **May 2025**, following his tenure at **Capgemini Technology Service Limited India**.\n\n"
                "- **Availability Details**:\n"
                "  - Ready for roles in **AI/ML**, **MLOps**, **DevOps**, **web development**, **IoT**, or **system automation**.\n"
                "  - Brings **4+ years** of experience, including **LLM fine-tuning**, **RAG applications**, **cloud-native deployments**, **Ansible automation**, and **bash scripting**.\n"
                "  - Proven expertise at **Capgemini**, leading **AI-driven projects**, automating **MLOps pipelines**, and optimizing **Linux infrastructure**.\n"
                "- **Ideal Roles**:\n"
                "  - **AI/ML Engineer**: Developing **generative AI** or **predictive models**.\n"
                "  - **MLOps Engineer**: Automating **CI/CD pipelines** for ML deployments.\n"
                "  - **DevOps Engineer**: Managing **Kubernetes**, **AWS**, or **Ansible**-based infrastructure.\n"
                "  - **IoT Developer**: Building real-time systems with **Raspberry Pi** and **RFID**.\n"
                "  - **System Automation Engineer**: Scripting **bash** or **Ansible** solutions for **Linux** management.\n\n"
                "Contact him via **[Email](mailto:msidrm455@gmail.com)** or **Phone** (+91 97406 71620) to discuss opportunities."
            )
        },
        {
            "content": "What is Siddharamayya great at technically?",
            "response": (
                "**Siddharamayya Mathapati** is a versatile technologist with exceptional expertise in **AI/ML**, **DevOps**, **web development**, **cloud engineering**, **IoT**, and **system automation**, enabling him to build end-to-end solutions.\n\n"
                "- **Python** (90%):\n"
                "  - Writes clean, modular code for **AI/ML**, **web development**, **automation**, and **system scripting**.\n"
                "  - Leverages **TensorFlow**, **PyTorch**, **LangChain**, **FastAPI**, **Scrapy**, and **pyVmomi**.\n"
                "  - Example:\n"
                "    ```python\n"
                "    from langchain import LLMChain\n"
                "    from langchain.prompts import PromptTemplate\n"
                "    def create_rag_chain(model, vector_db):\n"
                "        prompt = PromptTemplate(input_variables=['query', 'context'])\n"
                "        return LLMChain(llm=model, prompt=prompt, retriever=vector_db)\n"
                "    ```\n"
                "- **Deep Learning** & **Machine Learning/AI** (85%):\n"
                "  - Designs **neural networks** and **LLMs** for **NLP**, **computer vision**, and **predictive modeling**.\n"
                "  - Fine-tunes **LLMs** with **QLoRA** and **RLHF**; builds models like **Random Forest** for agriculture and finance.\n"
                "- **Web Development** (95%):\n"
                "  - Builds responsive applications with **React**, **Flask**, **Django**, **FastAPI**, and **Streamlit**.\n"
                "  - Example:\n"
                "    ```python\n"
                "    from fastapi import FastAPI\n"
                "    app = FastAPI()\n"
                "    @app.get('/')\n"
                "    async def root():\n"
                "        return {'message': 'Hello, World!'}\n"
                "    ```\n"
                "- **DevOps**:\n"
                "  - Automates **CI/CD pipelines** with **Docker**, **Kubernetes**, **Ansible**, **Jenkins**, and **GitHub Actions**.\n"
                "  - Manages **AWS**, **GCP**, **Azure**, and **OpenShift** infrastructure.\n"
                "- **DBMS & SQL** (80%):\n"
                "  - Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB).\n"
                "- **IoT** (80%):\n"
                "  - Develops real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**.\n"
                "- **System Administration** (85%):\n"
                "  - Automates **Linux** management with **Ansible**, **bash**, and **systemd**, optimizing server performance.\n\n"
                "His **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning** keep him at the forefront of technology."
            )
        },
        {
            "content": "How good is Siddharamayya in AI/ML?",
            "response": (
                "**Siddharamayya Mathapati** is highly proficient in **AI/ML** (85%), with extensive experience in **large language models (LLMs)**, **deep learning**, **generative AI**, **predictive modeling**, and **MLOps**, making him a standout in the field.\n\n"
                "- **LLM Expertise**:\n"
                "  - Fine-tuned **LLMs** using **QLoRA** and **LoRA** at **Capgemini**, optimizing efficiency for **finance**, **healthcare**, and **cybersecurity**.\n"
                "  - Applied **RLHF** to align models, improving contextual accuracy by **20%**.\n"
                "  - Example: Developed **CodeSage** using **RAG** for documentation-driven code generation.\n"
                "- **Deep Learning**:\n"
                "  - Designed **neural networks** for **NLP**, **computer vision**, and **predictive modeling** using **TensorFlow** and **PyTorch**.\n"
                "  - Built **SysAdmin-GPT** with **BERT** for **RHEL** query classification.\n"
                "- **Predictive Modeling**:\n"
                "  - Developed models like **Random Forest** for **Crop Recommendation** and **Blood Donation Prediction**, achieving high accuracy.\n"
                "  - Handled **imbalanced datasets** in **Credit Default Prediction** with SMOTE.\n"
                "- **RAG Applications**:\n"
                "  - Deployed **Retrieval-Augmented Generation** systems with **FAISS** and **ChromaDB**, enhancing response accuracy by **25%**.\n"
                "  - Example: **Student Study Assistant** chatbot.\n"
                "- **MLOps**:\n"
                "  - Automated **model training**, **deployment**, and **monitoring** with **Kubernetes**, **Docker**, and **AWS SageMaker**.\n"
                "  - Built **retraining pipelines** with **drift detection**, reducing errors by **30%**.\n"
                "- **Certifications**:\n"
                "  - **LLM Engineering** and **Deep Learning Masterclass** (Udemy) validate his expertise.\n\n"
                "His portfolio, including **Heart Disease Prediction** and **Workout & Fitness Tracker**, demonstrates his ability to apply **AI/ML** to diverse domains."
            )
        },
        {
            "content": "What are Siddharamayya's projects?",
            "response": (
                "**Siddharamayya Mathapati** has an extensive portfolio showcasing expertise in **AI/ML**, **web development**, **DevOps**, **IoT**, **Ansible automation**, **bash scripting**, and **system administration**. Below is a comprehensive overview:\n\n"
                "- **SysAdmin-GPT: AI-Powered Linux System Management** ([SysAdmin-GPT: AI-Powered Linux System Managemen](https://siddharamayya.in/projects)):\n"
                "  - A **BERT-based classification model** fine-tuned on **Red Hat Enterprise Linux (RHEL)** documentation to classify queries (e.g., Security, Networking, Storage).\n"
                "  - Integrated with a **chatbot interface** for automated **IT support**, enhancing query resolution efficiency.\n"
                "- **Workout & Fitness Tracker ML Model** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **machine learning model** analyzing **10,000+ workout records** to predict workout efficiency.\n"
                "  - Deployed with a **Streamlit UI** for real-time fitness insights.\n"
                "- **Crop Recommendation System** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Random Forest Classifier** recommending crops based on soil and environmental data.\n"
                "  - Features a **Streamlit UI** for farmers, promoting **sustainable farming**.\n"
                "- **Blood Donation Prediction** ([Blood Donation Prediction](https://siddharamayya.in/projects)):\n"
                "  - A **Random Forest Classifier** predicting blood donation likelihood using **Taiwan transfusion data**.\n"
                "  - Deployed with a **Streamlit app** for user interaction.\n"
                "- **Credit Default Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **machine learning model** predicting credit card defaults, handling **imbalanced datasets** with SMOTE.\n"
                "  - Supports **financial risk management**.\n"
                "- **Heart Disease Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **web-based app** predicting **10-year heart disease risk** using **scikit-learn**.\n"
                "  - Built with **Flask** and **Bootstrap** for a responsive UI.\n"
                "- **CodeSage: AI-Powered Documentation Suite** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **LLM-powered assistant** using **RAG** to generate context-aware code from **PDFs/docs** in **Python**, **JavaScript**, etc.\n"
                "  - Ideal for **documentation-driven development**.\n"
                "- **Containerized AI-Powered Automation Lab** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Docker/OpenShift** platform with **CodeServer**, **GitLab**, **AWX**, and an **AI assistant** for **Ansible playbook** generation.\n"
                "  - Features **LDAP/AD authentication** and **SSH key automation**.\n"
                "- **NMON Analyser for Linux** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Python tool** automating **NMON file** analysis from remote **Linux servers** via **SSH**, generating **.docx reports** and **email notifications**.\n"
                "- **Ansible Automation Platform API** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Python script** for **Red Hat Ansible Automation Platform** to launch **job templates** and extract **notifications** for **CI/CD** integration.\n"
                "- **VMware Automation for vSphere** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Python script** using **pyVmomi** to fetch **VM details** and create **VMs** on **vCenter servers**.\n"
                "- **Ansible Role: AWS VM Creation** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **Ansible script** creating **AWS EC2 VMs** and hosting jobs on **AWX**.\n"
                "- **Ansible Role: Systemd Service** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **Ansible role** managing **systemd services** on **Linux** systems.\n"
                "- **Ansible Role: Swap Management** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **Ansible role** automating **swap space** creation and extension.\n"
                "- **Ansible Role: Kernel Parameter Management** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **Ansible role** configuring **kernel parameters** in **sysctl** and **GRUB**.\n"
                "- **System Resource Usage Monitoring** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** monitoring **CPU**, **memory**, and **swap usage**, outputting to **CSV**.\n"
                "- **Remote Swap Management** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** managing **swap space** on remote **Linux servers** via **SSH**, reporting in **CSV**.\n"
                "- **Systemd Service Deployment** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** deploying **systemd services** with **SELinux** configuration.\n"
                "- **User Disabling Script** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** disabling user accounts, backing up files, and sending **HTML/CSV reports** via **email**.\n"
                "- **Check Last Patch** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** collecting **system update** info (uptime, kernel) from remote **Linux servers**.\n"
                "- **NFS/CIFS Share Monitoring** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **bash script** checking **NFS/CIFS share** mount status, generating **CSV/HTML reports**.\n"
                "- **CIFS fstab Updater** ([GitHub](https://siddharamayya.in/projectsr)):\n"
                "  - A **bash script** updating **CIFS entries** in **/etc/fstab** with **UIDs/GIDs**.\n"
                "- **Django Tutor Application** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Django app** guiding beginners to create and host apps on **AWS EC2**.\n"
                "- **FastAPI Microservice Application** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **FastAPI** and **MySQL** app with a **React frontend**, deployed as a **Docker microservice**.\n"
                "- **LinkedIn Activity Scraper** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Python tool** using **Selenium** and **BeautifulSoup** for **LinkedIn** data extraction.\n"
                "- **Recruitment Automation** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **AI-driven tool** generating **cold emails** using **NLP** and **web scraping**, reducing effort by **80%**.\n"
                "- **Stock Price Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **LLM-based finance agent** integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy.\n"
                "- **Credit Risk Analysis** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **CatBoostClassifier** model predicting credit defaults with **90% accuracy**.\n"
                "- **Medical Assistant AI** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **web app** using **Google Generative AI** for **medical image analysis**.\n"
                "- **SQL Query Generator** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Jupyter tool** using **Google Gemini LLM** for **SQL query** generation.\n"
                "- **Student Study Assistant** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **chatbot** using **LangChain** and **Gemini LLM** for student queries.\n"
                "- **Smart School Bus Tracking** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **IoT system** using **RFID**, **GPS**, and **Raspberry Pi** for student tracking.\n"
                "- **Car Security System** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **AI-based IoT system** with **facial recognition** and **RFID** for vehicle security.\n\n"
                "Explore more on his **[GitHub](https://github.com/mtptisid)** and **[Portfolio](https://mtptisid.github.io)**."
            )
        },
        {
            "content": "What is Siddharamayya's GitHub page?",
            "response": (
                "**Siddharamayya Mathapati**’s **GitHub** profile showcases his expertise in **AI/ML**, **web development**, **DevOps**, **IoT**, **Ansible automation**, and **bash scripting**.\n\n"
                "- **[GitHub](https://github.com/mtptisid)**:\n"
                "  - Features projects like **SysAdmin-GPT**, **CodeSage**, **Containerized Automation Lab**, and **NMON Analyser**.\n"
                "  - Demonstrates proficiency in **Python**, **LLMs**, **Ansible**, **bash**, and **Docker**.\n"
                "  - Includes **open-source contributions** and **DevOps** workflows with **Kubernetes** and **GitHub Actions**.\n\n"
                "Visit his **[Portfolio](https://mtptisid.github.io)** for a curated overview."
            )
        },
        {
            "content": "What is Siddharamayya's portfolio page?",
            "response": (
                "**Siddharamayya Mathapati**’s **portfolio** is a professional showcase of his work in **AI/ML**, **web development**, **DevOps**, **IoT**, and **system automation**.\n\n"
                "- **[Portfolio](https://mtptisid.github.io)**:\n"
                "  - Highlights projects like **Crop Recommendation System**, **Heart Disease Prediction**, **Containerized Automation Lab**, and **Ansible Role: AWS VM Creation**.\n"
                "  - Showcases skills in **Python**, **React**, **Kubernetes**, **Ansible**, and **AWS**.\n"
                "  - Provides insights into his **Udemy certifications** and **4+ years** of experience.\n"
                "- **Complementary Links**:\n"
                "  - **[GitHub](https://github.com/mtptisid)**: Detailed project repositories.\n"
                "  - **[LinkedIn](https://linkedin.com/in/siddharamayya)**: Professional achievements and endorsements.\n\n"
                "The portfolio is an excellent resource for exploring his **innovative solutions**."
            )
        },
        {
            "content": "What are Siddharamayya's key skills?",
            "response": (
                "**Siddharamayya Mathapati** possesses a robust skill set across **AI/ML**, **DevOps**, **web development**, **cloud engineering**, **IoT**, and **system administration**.\n\n"
                "- **Python** (90%):\n"
                "  - Expert in **AI/ML**, **web development**, **automation**, and **system scripting** with **TensorFlow**, **PyTorch**, **LangChain**, **FastAPI**, and **pyVmomi**.\n"
                "- **Deep Learning** & **Machine Learning/AI** (85%):\n"
                "  - Designs **neural networks** and **LLMs** for **NLP**, **computer vision**, and **predictive modeling**.\n"
                "- **Web Development** (95%):\n"
                "  - Builds responsive apps with **React**, **Flask**, **Django**, **FastAPI**, and **Streamlit**.\n"
                "- **DevOps**:\n"
                "  - Automates **CI/CD pipelines** with **Docker**, **Kubernetes**, **Ansible**, and **GitHub Actions**.\n"
                "- **Cloud Engineering** (85%):\n"
                "  - Optimizes **AWS** (SageMaker, EC2), **GCP**, **Azure**, and **OpenShift**.\n"
                "- **DBMS & SQL** (80%):\n"
                "  - Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases**.\n"
                "- **IoT** (80%):\n"
                "  - Develops systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**.\n"
                "- **System Administration** (85%):\n"
                "  - Automates **Linux** management with **Ansible**, **bash**, and **systemd**.\n\n"
                "His skills are validated by **Udemy certifications** and applied in projects like **SysAdmin-GPT** and **Containerized Automation Lab**."
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
                "His education, combined with **Udemy certifications**, supports his technical expertise."
            )
        },
        {
            "content": "What is Siddharamayya’s current role at Capgemini?",
            "response": (
                "**Siddharamayya Mathapati** serves as a **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025), leading **AI/ML**, **MLOps**, and **system automation** initiatives.\n\n"
                "- **Responsibilities**:\n"
                "  - Fine-tunes **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing memory usage by **30%**.\n"
                "  - Designs **RAG applications** with **FAISS** and **ChromaDB**, improving accuracy by **25%**.\n"
                "  - Automates **MLOps pipelines** using **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**.\n"
                "  - Develops **real-time inference pipelines** with **TensorRT** and **ONNX**.\n"
                "  - Migrates legacy ML workloads to **Spark** and **Kubernetes**, enhancing scalability by **40%**.\n"
                "  - Automates **Linux infrastructure** with **Ansible** and **bash**, optimizing performance.\n"
                "  - Mentors junior engineers and supervises research internships.\n\n"
                "His role showcases his ability to deliver **high-impact**, **scalable** solutions."
            )
        },
        {
            "content": "What languages does Siddharamayya speak?",
            "response": (
                "**Siddharamayya Mathapati** is fluent in **seven languages**, enhancing his ability to collaborate across diverse teams.\n\n"
                "- **Kannada**: Native; fluent in speaking, reading, and writing.\n"
                "- **English**: Professional fluency; excels in technical writing and global collaboration.\n"
                "- **Hindi**: Fluent; skilled in verbal and written communication.\n"
                "- **Telugu**: Conversational fluency; comfortable in professional discussions.\n"
                "- **Tamil**: Conversational fluency; adept in technical contexts.\n"
                "- **Marathi**: Intermediate fluency; effective in professional settings.\n"
                "- **Malayalam**: Basic proficiency; can engage in simple conversations.\n\n"
                "His linguistic versatility supports his work in **finance**, **healthcare**, and **IT**."
            )
        },
        {
            "content": "How can I contact Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** is accessible for professional inquiries and collaborations.\n\n"
                "- **[Email](mailto:msidrm455@gmail.com)**: Preferred for formal communication or job opportunities.\n"
                "- **Phone**: +91 97406 71620: Available for calls or messages to discuss **AI/ML** or **DevOps** projects.\n"
                "- **Address**: #372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India: For official correspondence.\n"
                "- **[LinkedIn](https://linkedin.com/in/siddharamayya)**: Ideal for networking and professional updates.\n\n"
                "He is responsive and open to discussing **innovative** opportunities."
            )
        },
        {
            "content": "What are Siddharamayya’s interests in AI?",
            "response": (
                "**Siddharamayya Mathapati** is passionate about **AI** and its transformative potential.\n\n"
                "- **Generative AI & LLM Development**: Builds **LLMs** for **text generation** and **image analysis** (e.g., **CodeSage**).\n"
                "- **AI-Powered Financial Modeling**: Develops **predictive models** for stock analysis (e.g., **Stock Price Prediction**).\n"
                "- **NLP & Chatbots**: Creates **AI agents** for natural language tasks (e.g., **SysAdmin-GPT**).\n"
                "- **Cloud-Native MLOps**: Integrates **AI** with **cloud platforms** for scalable deployments.\n"
                "- **IoT & Real-Time AI**: Combines **AI** with **IoT** for systems like **Smart School Bus Tracking**.\n"
                "- **AI-Driven System Automation**: Automates **IT infrastructure** with **AI** (e.g., **Containerized Automation Lab**).\n\n"
                "His interests align with his **4+ years** of experience and **Udemy certifications**."
            )
        },
        {
            "content": "What generative AI projects has Siddharamayya worked on?",
            "response": (
                "**Siddharamayya Mathapati** has developed several **generative AI** projects leveraging **LLMs** and **deep learning**.\n\n"
                "- **CodeSage: AI-Powered Documentation Suite** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **LLM-powered assistant** using **RAG** to generate **Python**, **JavaScript**, and other code from **PDFs/docs**.\n"
                "  - Enhances **documentation-driven development** for developers.\n"
                "- **Medical Assistant AI** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **web app** using **Google Generative AI** for **medical image analysis** and diagnostic insights via a **chat interface**.\n"
                "- **Stock Price Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **AI-powered finance agent** using **LLMs** for stock price predictions, improving accuracy by **25%**.\n"
                "- **Student Study Assistant** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **chatbot** with **LangChain** and **Gemini LLM** generating responses from **PDF documents**.\n"
                "- **SQL Query Generator** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **Jupyter tool** using **Google Gemini LLM** for natural language **SQL query** generation.\n"
                "- **Containerized AI-Powered Automation Lab** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **AI assistant** generating **Ansible playbooks** in a **Docker/OpenShift** environment.\n\n"
                "These projects highlight his expertise in **generative AI**, supported by his **LLM Engineering** certification."
            )
        },
        {
            "content": "What DevOps tools does Siddharamayya use?",
            "response": (
                "**Siddharamayya Mathapati** is proficient in **DevOps tools** for automating **CI/CD pipelines**, **cloud infrastructure**, and **system deployments**.\n\n"
                "- **Docker**: Creates **containerized applications** for portability (e.g., **FastAPI Microservice**).\n"
                "- **Kubernetes**: Orchestrates **containerized workloads** for scalability (e.g., **MLOps pipelines** at **Capgemini**).\n"
                "- **Ansible**: Automates **configuration management** and **server provisioning** (e.g., **AWS VM Creation**).\n"
                "- **Jenkins**: Builds **CI/CD pipelines** for automated testing and deployment.\n"
                "- **GitHub Actions**: Automates **workflows** for code deployment (e.g., **Stock Price Prediction**).\n"
                "- **AWS SageMaker**: Deploys **ML models** in the cloud.\n"
                "- **MLflow**: Tracks **ML experiments** and manages model lifecycles.\n"
                "- **Airflow**: Schedules **data pipelines** for **ML workflows**.\n"
                "- **OpenShift**: Manages **containerized platforms** (e.g., **Automation Lab**).\n\n"
                "His **MLOps Bootcamp** certification enhances his **DevOps** expertise."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with LLMs?",
            "response": (
                "**Siddharamayya Mathapati** has extensive experience with **large language models (LLMs)**, focusing on **fine-tuning**, **deployment**, and **application development**.\n\n"
                "- **Fine-Tuning**:\n"
                "  - Fine-tuned **LLMs** using **QLoRA** and **LoRA** at **Capgemini**, reducing memory usage by **30%** for **finance** and **healthcare**.\n"
                "  - Applied **RLHF** to align models, improving accuracy by **20%**.\n"
                "- **RAG Applications**:\n"
                "  - Designed **Retrieval-Augmented Generation** systems with **FAISS** and **ChromaDB**, enhancing accuracy by **25%**.\n"
                "  - Example: **CodeSage** for code generation from **PDFs**.\n"
                "- **Project Highlights**:\n"
                "  - **SysAdmin-GPT** ([GitHub](https://siddharamayya.in/projects)): Fine-tuned **BERT** for **RHEL** query classification.\n"
                "  - **CodeSage** ([GitHub](https://siddharamayya.in/projects)): Uses **RAG** for **code generation**.\n"
                "  - **Stock Price Prediction** ([GitHub]https://siddharamayya.in/projects)): **LLM-based** market analysis.\n"
                "  - **SQL Query Generator** ([GitHub](https://siddharamayya.in/projects)): **Gemini LLM** for **SQL queries**.\n"
                "  - **Containerized Automation Lab** ([GitHub](https://siddharamayya.in/projects)): **LLM** for **Ansible playbooks**.\n"
                "- **Tools & Frameworks**:\n"
                "  - Proficient in **LangChain**, **Hugging Face**, **PyTorch**, and **AWS SageMaker**.\n"
                "- **Certifications**:\n"
                "  - **LLM Engineering** (Udemy) validates his expertise.\n\n"
                "His work demonstrates deep **LLM** optimization and application expertise."
            )
        },
        {
            "content": "What AI-powered finance projects has Siddharamayya built?",
            "response": (
                "**Siddharamayya Mathapati** has developed impactful **AI-powered finance projects** using **LLMs** and **machine learning**.\n\n"
                "- **Stock Price Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - An **AI-powered finance agent** using **LLMs** to predict stock prices, integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy.\n"
                "- **Credit Default Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **machine learning model** handling **imbalanced datasets** to predict credit card defaults, supporting **risk management**.\n"
                "- **Credit Risk Analysis** ([GitHub](https://siddharamayya.in/projects)):\n"
                "  - A **CatBoostClassifier** model predicting credit defaults with **90% accuracy**.\n"
                "- **Capgemini Finance Projects**:\n"
                "  - Fine-tuned **LLMs** for **financial analytics**, optimizing **contextual accuracy** in budgeting and forecasting.\n"
                "  - Deployed **RAG applications** for financial document retrieval, enhancing efficiency by **30%**.\n\n"
                "These projects highlight his ability to solve complex financial challenges with **AI/ML**."
            )
        },
        {
            "content": "What is Siddharamayya’s tech stack?",
            "response": (
                "**Siddharamayya Mathapati**’s **tech stack** covers **AI/ML**, **web development**, **DevOps**, **cloud engineering**, **IoT**, and **system administration**.\n\n"
                "- **Programming Languages**: **Python**, **C++**, **Ruby**, **JavaScript**, **Golang**, **Bash**.\n"
                "- **AI/ML Frameworks**: **TensorFlow**, **PyTorch**, **LangChain**, **Hugging Face**, **PySpark**, **Scrapy**.\n"
                "- **Web Development**: **Flask**, **Django**, **FastAPI**, **React**, **Streamlit**, **Bootstrap**.\n"
                "- **DevOps Tools**: **Docker**, **Kubernetes**, **Ansible**, **Jenkins**, **GitHub Actions**, **MLflow**, **Airflow**, **OpenShift**.\n"
                "- **Cloud Platforms**: **AWS** (SageMaker, EMR, Lambda, EC2), **GCP**, **Azure**.\n"
                "- **Databases**: **MySQL**, **PostgreSQL**, **MongoDB**, **ChromaDB**, **FAISS**.\n"
                "- **IoT Technologies**: **Raspberry Pi**, **ESP32**, **RFID**, **GPS**, **MQTT**.\n"
                "- **System Administration**: **pyVmomi**, **pyVim**, **Selenium**, **BeautifulSoup**, **systemd**.\n"
                "- **Other Tools**: **Git**, **VS Code**, **RabbitMQ**, **Postman**, **LangGraph**, **Firebase**.\n\n"
                "His stack supports projects like **SysAdmin-GPT**, **FastAPI Microservice**, and **NFS/CIFS Share Monitoring**."
            )
        },
        {
            "content": "How does Siddharamayya use Docker and Kubernetes?",
            "response": (
                "**Siddharamayya Mathapati** leverages **Docker** and **Kubernetes** for **CI/CD pipelines**, **ML model** deployment, and **cloud-native** applications.\n\n"
                "- **Docker**:\n"
                "  - Creates **containerized environments** for **AI/ML** and **web apps** (e.g., **FastAPI Microservice**).\n"
                "  - Packages **ML models** for consistent deployments.\n"
                "- **Kubernetes**:\n"
                "  - Orchestrates **containerized workloads** for scalability (e.g., **MLOps pipelines** at **Capgemini**).\n"
                "  - Deployed **RAG applications** on **Kubernetes** clusters.\n"
                "- **Integration**:\n"
                "  - Uses **Docker** and **Kubernetes** in **AWS**, **GCP**, and **OpenShift** (e.g., **Containerized Automation Lab**).\n"
                "  - Integrates with **Jenkins** and **GitHub Actions** for **CI/CD**.\n"
                "- **Monitoring**:\n"
                "  - Implements **Prometheus** and **Grafana** for **Kubernetes** monitoring.\n\n"
                "His **MLOps** certification enhances his expertise."
            )
        },
        {
            "content": "What are Siddharamayya’s contributions to web development?",
            "response": (
                "**Siddharamayya Mathapati** has significant contributions to **web development** (95% proficiency), building responsive applications.\n\n"
                "- **Projects**:\n"
                "  - **FastAPI Microservice Application** ([GitHub](https://siddharamayya.in/projects)):\n"
                "    - A **FastAPI** app with a **React frontend**, deployed as a **Docker microservice**.\n"
                "  - **Django Tutor Application** ([GitHub](https://siddharamayya.in/projects)):\n"
                "    - A **Django app** guiding beginners to host apps on **AWS EC2**.\n"
                "  - **Heart Disease Prediction** ([GitHub](https://siddharamayya.in/projects)):\n"
                "    - A **Flask** app with **Bootstrap** for **heart disease risk** prediction.\n"
                "- **Technologies**:\n"
                "  - **Frontend**: **React**, **Bootstrap**, **HTML**, **CSS**.\n"
                "  - **Backend**: **Flask**, **Django**, **FastAPI**.\n"
                "  - **Deployment**: **Docker**, **Kubernetes**, **AWS EC2**.\n"
                "- **Capgemini Contributions**:\n"
                "  - Developed **web dashboards** for **ML model** monitoring using **Streamlit**.\n"
                "  - Automated **web scraping** with **Selenium** (e.g., **LinkedIn Activity Scraper**).\n\n"
                "His work combines **UX design**, **backend development**, and **DevOps**."
            )
        },
        {
            "content": "Visit Sid's portfolio",
            "response": (
                "To explore **Siddharamayya Mathapati**’s work, visit his **portfolio**:\n\n"
                "- **[Portfolio](https://mtptisid.github.io)**:\n"
                "  - Showcases projects like **SysAdmin-GPT**, **CodeSage**, **FastAPI Microservice**, and **Ansible Automation Lab**.\n"
                "  - Highlights **AI/ML**, **web development**, **DevOps**, and **IoT** expertise.\n"
                "- **Additional Links**:\n"
                "  - **[GitHub](https://github.com/mtptisid)**: Project repositories.\n"
                "  - **[LinkedIn](https://linkedin.com/in/siddharamayya)**: Professional updates.\n\n"
                "His **portfolio** is a comprehensive resource for his **innovative** contributions."
            )
        },
        {
            "content": "Download Resume",
            "response": (
                "**Siddharamayya Mathapati**’s resume details his **4+ years** of experience in **AI/ML**, **MLOps**, **DevOps**, and **web development**. For a downloadable version, contact him:\n\n"
                "- **[Email](mailto:msidrm455@gmail.com)**: Request a PDF copy tailored to **AI/ML** or **DevOps** roles.\n"
                "- **Phone**: +91 97406 71620: Discuss resume details.\n"
                "- **[LinkedIn](https://linkedin.com/in/siddharamayya)**: View professional summary.\n\n"
                "Explore his **[Portfolio](https://mtptisid.github.io)** or **[GitHub](https://github.com/mtptisid)** for project details."
            )
        },
        {
            "content": "What certifications does Siddharamayya have?",
            "response": (
                "**Siddharamayya Mathapati** has earned **Udemy certifications** validating his expertise in **AI/ML**, **MLOps**, and **data science**.\n\n"
                "- **LLM Engineering: Master AI, Large Language Models & Agents**: Covers **LLM fine-tuning** and **AI agents** (e.g., **CodeSage**).\n"
                "- **MLOps Bootcamp: Mastering AI Operations for Success**: Focuses on **CI/CD pipelines** and **cloud deployments** (e.g., **Containerized Automation Lab**).\n"
                "- **Python for Data Analysis & Visualization**: Covers **Pandas**, **NumPy**, and **Matplotlib** (e.g., **Credit Risk Analysis**).\n"
                "- **Deep Learning Masterclass with TensorFlow 2**: Explores **neural networks** (e.g., **SysAdmin-GPT**).\n\n"
                "These certifications complement his **4+ years** of experience."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with IoT?",
            "response": (
                "**Siddharamayya Mathapati** has significant **IoT** experience (80% proficiency), developing real-time systems.\n\n"
                "- **X-Cencia Internship** (February 2020 - April 2021):\n"
                "  - Developed **Smart School Bus Tracking System** using **RFID**, **GPS**, and **Raspberry Pi**.\n"
                "  - Built **Car Security System** with **IoT**, **facial recognition**, and **RFID**.\n"
                "  - Designed **data pipelines** for **sensor data** from **ESP32**.\n"
                "- **Projects**:\n"
                "  - **Smart School Bus Tracking** ([GitHub](https://siddharamayya.in/projects)): Real-time student tracking with a **web interface**.\n"
                "  - **Car Security System** ([GitHub](https://siddharamayya.in/projects)): **AI-based** security with **IoT**.\n"
                "- **Technologies**:\n"
                "  - **Raspberry Pi**, **ESP32**, **RFID**, **GPS**, **MQTT**.\n"
                "  - **Python** and **Flask** for **IoT** interfaces.\n"
                "- **Capgemini Contributions**:\n"
                "  - Integrated **IoT data** with **ML models** for real-time analytics.\n\n"
                "His **IoT** expertise enhances his **AI/ML** capabilities."
            )
        },
        {
            "content": "What is Siddharamayya’s experience with MLOps?",
            "response": (
                "**Siddharamayya Mathapati** has extensive **MLOps** experience, automating **machine learning** workflows.\n\n"
                "- **Capgemini** (May 2021 - April 2025):\n"
                "  - Automated **MLOps pipelines** with **Kubernetes**, **Docker**, and **AWS SageMaker**, reducing deployment time by **50%**.\n"
                "  - Built **retraining pipelines** with **drift detection**, cutting errors by **30%**.\n"
                "  - Developed **monitoring** systems with **Prometheus** and **Grafana**.\n"
                "- **Tools**:\n"
                "  - **MLflow**: Tracks **ML experiments**.\n"
                "  - **Airflow**: Schedules **data pipelines**.\n"
                "  - **Jenkins** & **GitHub Actions**: Automates **CI/CD**.\n"
                "- **Projects**:\n"
                "  - **Workout & Fitness Tracker** ([GitHub](https://siddharamayya.in/projects)): Deployed with **MLOps** automation.\n"
                "  - **Crop Recommendation System** ([GitHub](https://siddharamayya.in/projects)): Managed with **MLflow**.\n"
                "- **Certification**:\n"
                "  - **MLOps Bootcamp** (Udemy) validates his expertise.\n\n"
                "His **MLOps** skills ensure robust **ML** deployments."
            )
        },
        {
        "content": "What are Siddharamayya’s Ansible-based projects?",
        "response": (
            "**Siddharamayya Mathapati** has developed several **Ansible-based projects** for automating **infrastructure** and **system administration** tasks.\n\n"
            "- **Containerized AI-Powered Automation Lab** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - A **Docker-based lab** integrating **AWX**, **CodeServer**, and **GitLab** with an **AI assistant** generating **Ansible** playbooks.\n"
            "- **Ansible Automation Platform API** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - A **Python script** launching **Ansible Automation Platform** jobs and extracting notifications for **CI/CD** integration.\n"
            "- **Ansible Role: AWS VM Creation** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - An **Ansible script** creating **VMs** in **AWS** and hosting jobs on **AWX**.\n"
            "- **Ansible Role: Systemd Service** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - An **Ansible role** automating **systemd service** creation on **Linux**.\n"
            "- **Ansible Role: Swap Management** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - An **Ansible role** managing **swap space** for performance optimization.\n"
            "- **Ansible Role: Kernel Parameter Management** ([GitHub](https://siddharamayya.in/projects)):\n"
            "  - An **Ansible role** modifying **sysctl** and **GRUB** parameters.\n\n"
            "These projects highlight his **Ansible** expertise for **cloud** and **Linux** automation."
        )
    }
]
    if message.model == "groq":
        system_prompt = (
        "You are an AI assistant with detailed knowledge of Siddharamayya Mathapati, a Senior Software Engineer at Capgemini (2021-2025) with over 4 years of experience in AI/ML, MLOps, DevOps, web development, IoT, and system automation. "
        "Key details:\n"
        "- **Skills**: Expert in Python, TensorFlow, PyTorch, LangChain, FastAPI, Docker, Kubernetes, Ansible, AWS, MySQL, and vector databases (FAISS, ChromaDB). Proficient in LLMs, RAG applications, and bash scripting.\n"
        "- **Experience**: Led AI/ML projects at Capgemini, fine-tuning LLMs with QLoRA, building RAG systems, and automating MLOps pipelines, reducing deployment time by 50%. Developed IoT solutions like Smart School Bus Tracking at X-Cencia (2020-2021).\n"
        "- **Projects**: SysAdmin-GPT (BERT-based Linux support), CodeSage (LLM-powered code generator), Crop Recommendation System (Random Forest), and FastAPI Microservice (Dockerized web app).\n"
        "- **Education**: MCA (Acharya Institute, 2018-2020), BCA (B K College, 2015-2018).\n"
        "- **Contact**: Email (msidrm455@gmail.com), GitHub ([mtptisid](https://github.com/mtptisid)), LinkedIn ([siddharamayya](https://linkedin.com/in/siddharamayya)).\n"
        "Respond in Markdown format using **bold**, `code blocks`, - bulleted lists, and [name](link) for URLs. Use chat history and web search results (if provided) for accurate, context-aware answers."
        )
    else:
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
            "\n".join([f"**Q**: {ex['content']}\n  {ex['response']}" for ex in FEW_SHOT_EXAMPLES]) + "\n\n"
            "**Instructions**:\n"
            "- Always respond in **Markdown format** with proper spacing and structure.\n"
            "- Use **bold** for key terms (e.g., names, roles, technologies).\n"
            "- Wrap code snippets in triple backticks (e.g., ```python ... ```).\n"
            "- Use `-` for bulleted lists, with one item per line and blank lines before/after.\n"
            "- Format URLs as **[name](link)** (e.g., **[project name](https://siddharamayya.in/projects)**).\n"
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

@router.get("/")
def homestart():
    return Response(status_code=204)
