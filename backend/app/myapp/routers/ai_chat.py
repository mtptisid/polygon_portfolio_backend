from fastapi import APIRouter, HTTPException, Request, Response
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime
from uuid import uuid4
from ..services.ai import ai_manager
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
    session_id: Optional[str] = None  # Changed to str for consistency
    tool: Optional[str] = None
class MessageResponse(BaseModel):
    content: str
    is_bot: bool
    session_id: str  # Changed to str
    timestamp: datetime
    tool_used: Optional[str] = None
class SessionMessage(BaseModel):
    content: str
    is_bot: bool
    timestamp: datetime
    tool_used: Optional[str] = None
class Session(BaseModel):
    session_id: str  # Changed to str
    messages: List[SessionMessage]
    created_at: datetime
from langchain_community.tools import DuckDuckGoSearchRun
# Initialize DuckDuckGo Search Run tool
search = DuckDuckGoSearchRun()
# In-memory session store (replace with database for production)
SESSIONS = {}
async def search_web(query: str, model: str = "groq") -> str:
    """Perform a web search using DuckDuckGo via LangChain without site restrictions for broader results."""
    full_query = query
    logger.info(f"Performing search with query: {full_query}")
   
    try:
        search_results = await asyncio.to_thread(search.run, full_query)
        if not search_results or "No good" in search_results:
            return "Web Search Results: Limited or no specific results found. Rely on general knowledge for explanation."
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
        return f"Web Search Results: Search failed: {str(e)}. Proceed with general knowledge."
def clean_text(text: str) -> str:
    """Clean text by removing excessive newlines and unwanted characters."""
    if not text:
        return "No response generated. Please try rephrasing your query."
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
                "title": "Senior Artificial Intelligence Engineer",
                "company": "Maveric Systems Limited",
                "location": "Bangalore, India",
                "duration": "June 2025 - Present",
                "description": (
                    "Architecting advanced **AI-driven systems** and **LLM-integrated applications** for enterprise-grade automation and insights. "
                    "Built and deployed a scalable **MCP (Model Context Protocol) server** with modular tool/resource orchestration and secure multi-client support. "
                    "Developed **MCP clients** capable of selectively accessing tools and resources, enabling role-based permissions for agents. "
                    "Engineered **Agentic AI frameworks** using **AutoGen**, **CrewAI**, and **Agno**, facilitating intelligent multi-agent collaboration across domains. "
                    "Designed reusable **tool wrappers**, **resource templates**, and **prompt adapters** to abstract system functionality for agents. "
                    "Performed rigorous **prompt evaluation** using automated benchmarks and manual feedback loops to optimize agent reliability. "
                    "Led efforts on **static code analysis** and **security hardening** for LLM-based applications, ensuring compliance with **OWASP** and **LLM-specific threat models**. "
                    "Implemented **role prompting**, **context chaining**, and **memory injection** strategies for high-fidelity LLM interaction. "
                    "Integrated **LangChain**, **Ollama**, and **FAISS/ChromaDB** to enable scalable RAG pipelines with offline compatibility. "
                    "Worked cross-functionally to deliver **AI-first automation**, improving decision velocity and reducing operational costs by **40%**."
                )
            },
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
                "link": "https://siddharamayya.in/projects"
            },
            {
               "category": "YOLOv5 Object Detection Practice",
               "description": (
                   "A practical repository demonstrating the use of **YOLOv5** for real-time object detection tasks. "
                   "Includes custom dataset preparation, annotation using Roboflow, model training, evaluation, and inference. "
                   "Covers the complete object detection workflow from data pipeline to deployment-ready predictions. "
                   "Ideal for learners and developers working with computer vision and real-time image recognition systems."
                ),
               "link": "https://github.com/mtptisid/YOLOV5_Practice"
            },
            {
               "category": "Custom CNN Model with PyTorch",
               "description": (
                    "An educational deep learning project building a **Convolutional Neural Network (CNN)** from scratch using **PyTorch**. "
                    "Demonstrates architecture design, training loops, and evaluation on image datasets. "
                    "Ideal for those learning about CNNs, PyTorch fundamentals, and custom model creation for image classification tasks."
              ),
              "link": "https://github.com/mtptisid/PyTorch-CNN-model"
            },
            {
               "category": "ML Hands-On: Practical Machine Learning Exploration",
               "description": (
                    "A comprehensive hands-on repository that demonstrates key machine learning concepts using real-world datasets. "
                    "Includes end-to-end projects covering data cleaning, visualization, feature engineering, model training, and evaluation. "
                    "Designed as a learning resource for beginners and intermediates to grasp ML fundamentals in practice, using popular Python libraries such as Scikit-learn and Pandas. "
                    "Projects are built using Jupyter notebooks for interactivity and clarity."
              ),
              "link": "https://github.com/mtptisid/ml-hands-on"
            },
            {
              "category": "RAG Hands-On: Building Retrieval-Augmented Generation Pipelines",
              "description": (
                "An applied project demonstrating how to build Retrieval-Augmented Generation (RAG) systems using large language models and vector databases. "
                "The repository integrates LangChain and FAISS to enable document-aware question answering using custom embeddings. "
                "This project serves as a practical introduction to RAG for developers and researchers interested in enhancing LLM capabilities with external knowledge sources."
              ),
              "link": "https://github.com/mtptisid/RAG-hands-on"
            },
            {
              "category": "IPL Score Prediction: Cricket Analytics with ML",
              "description": (
                "A machine learning project designed to predict the final score of an IPL (Indian Premier League) innings based on real match data. "
                "Utilizes regression algorithms and feature engineering techniques to forecast outcomes using current match states like overs, runs, and wickets. "
                "Includes detailed exploratory data analysis (EDA), visualization, model training, and evaluation. "
                "Ideal for sports analytics enthusiasts and aspiring data scientists."
              ),
              "link": "https://github.com/mtptisid/IPL_score_prediction"
            },
            {
                "category": "Workout & Fitness Tracker ML Model",
                "description": (
                    "A **machine learning model** analyzing **10,000+ workout records** to predict workout efficiency based on health metrics. "
                    "Deployed with a **Streamlit UI** for interactive, real-time fitness insights, helping users optimize routines."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Crop Recommendation System",
                "description": (
                    "A **Random Forest Classifier** recommending crops based on soil characteristics and environmental factors. "
                    "Features a **Streamlit UI** for farmers to input data, promoting **sustainable farming** and **yield optimization**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Blood Donation Prediction",
                "description": (
                    "A **Random Forest Classifier** predicting blood donation likelihood using **Taiwan blood transfusion data**. "
                    "Deployed with a **Streamlit app** for user-friendly interaction, analyzing features like recency and frequency."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
               "category": "FLAN-T5-Small Fine-Tuning with LoRA",
               "description": (
               "A practical guide and implementation for fine-tuning the **FLAN-T5-Small** language model using **Low-Rank Adaptation (LoRA)**. "
               "This repository walks through setting up parameter-efficient fine-tuning (PEFT) for downstream NLP tasks with Hugging Face Transformers. "
               "LoRA significantly reduces the computational requirements and memory footprint, making fine-tuning accessible even on modest hardware setups. "
               "Ideal for NLP researchers, developers, and enthusiasts working with resource-constrained environments."
             ),
              "link": "https://github.com/mtptisid/FLAN-T5-Small_finetuning_LoRA"
            },
            {
                "category": "Credit Default Prediction",
                "description": (
                    "A **machine learning model** predicting credit card defaults using simulated financial data. "
                    "Handles **imbalanced datasets** with techniques like SMOTE, achieving robust **creditworthiness assessments**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Heart Disease Prediction",
                "description": (
                    "A **web-based application** predicting **10-year heart disease risk** using a **machine learning model**. "
                    "Built with **Flask**, **Bootstrap**, and **scikit-learn**, featuring a responsive UI with gradient aesthetics."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "CodeSage: AI-Powered Documentation Suite",
                "description": (
                    "An **LLM-powered assistant** using **RAG** to answer questions from custom **PDFs/docs** and generate context-aware code in **Python**, **JavaScript**, and more. "
                    "Ideal for developers needing **documentation-driven development**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Containerized AI-Powered Automation Lab",
                "description": (
                    "A **containerized platform** integrating **CodeServer**, **GitLab**, **AWX**, and **Linux hosts** with an **AI assistant** for **Ansible playbook** generation. "
                    "Features **LDAP/AD authentication**, **SSH key automation**, and **Docker/OpenShift** deployment on **GCP**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "NMON Analyser for Linux",
                "description": (
                    "A **Python tool** automating **NMON file** collection and analysis from remote **Linux servers** via **SSH**. "
                    "Generates **performance reports** in **.docx** and sends **email notifications** with insights and plots."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Ansible Automation Platform API",
                "description": (
                    "A **Python script** for **Red Hat Ansible Automation Platform (AAP)** to launch **job templates** and extract **notifications**. "
                    "Supports **CI/CD pipeline** integration for **DevOps** automation."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "VMware Automation for vSphere",
                "description": (
                    "A **Python script** using **pyVmomi** to fetch **VM details** (CPU, memory, OS) from **vCenter servers** and create **VMs** with predefined specs. "
                    "Simplifies **VM management** for **DevOps** teams."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Ansible Role: AWS VM Creation",
                "description": (
                    "An **Ansible script** creating **VMs** in **AWS EC2** and hosting jobs on **AWX**. "
                    "Automates **cloud infrastructure** provisioning for scalable deployments."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Ansible Role: Systemd Service",
                "description": (
                    "An **Ansible role** to create and manage **systemd services** on **Linux** systems. "
                    "Ensures reliable service deployment and configuration."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Ansible Role: Swap Management",
                "description": (
                    "An **Ansible role** to manage **swap space** on **Linux** machines, automating creation and extension of swap files."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Ansible Role: Kernel Parameter Management",
                "description": (
                    "An **Ansible role** to configure **kernel parameters** in **sysctl** and **GRUB** for **Linux** systems, optimizing performance."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "System Resource Usage Monitoring",
                "description": (
                    "A **bash script** monitoring **CPU**, **memory**, and **swap usage** on **Linux** systems. "
                    "Outputs data in **CSV** for analysis, featuring user-friendly table formatting."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Remote Swap Management",
                "description": (
                    "A **bash script** for managing **swap space** on remote **Linux servers** via **SSH**. "
                    "Automates swap file creation and reports usage in **CSV**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Systemd Service Deployment",
                "description": (
                    "A **bash script** deploying **systemd services** across remote **Linux servers**, configuring **SELinux** and enabling services."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "User Disabling Script",
                "description": (
                    "A **bash script** disabling user accounts on remote **Linux servers**, backing up system files, and sending **HTML/CSV reports** via **email**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Check Last Patch",
                "description": (
                    "A **bash script** collecting **system update** information (uptime, Red Hat version, kernel) from remote **Linux servers** via **SSH**, saving to **log files**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "NFS/CIFS Share Monitoring",
                "description": (
                    "A **bash script** checking **NFS/CIFS share** mount status on **Linux servers** via **SSH**, verifying access and generating **CSV/HTML reports**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "CIFS fstab Updater",
                "description": (
                    "A **bash script** updating **CIFS entries** in **/etc/fstab** on remote **Linux servers**, replacing usernames with **UIDs/GIDs** for proper mounts."
                ),
                "link": "https://siddharamayya.in/projects"  # Fixed typo from /projectsr
            },
            {
                "category": "Django Tutor Application",
                "description": (
                    "A **Django-based web application** for expert talks, guiding beginners to create and host apps on **AWS EC2**. "
                    "Features user-friendly tutorials and deployment scripts."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "FastAPI Microservice Application",
                "description": (
                    "A **FastAPI** and **MySQL** application deployed as a **Docker microservice** with a **React frontend**. "
                    "Demonstrates scalable **web development** and **containerization**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "LinkedIn Activity Scraper",
                "description": (
                    "A **Python-based tool** using **Selenium** and **BeautifulSoup** to extract **LinkedIn** activity data (posts, likes, shares, comments). "
                    "Features **dynamic content handling** and **ethical scraping**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Recruitment Automation",
                "description": (
                    "An **AI-driven solution** extracting job data from career pages and generating personalized **cold emails** using **NLP** and **web scraping**, reducing effort by **80%**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Stock Price Prediction",
                "description": (
                    "An **AI-powered finance agent** using **LLMs** to predict stock prices, integrating **Yahoo Finance** and **NewsAPI** for **25%** improved accuracy."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Credit Risk Analysis",
                "description": (
                    "A **machine learning model** using **CatBoostClassifier** to predict credit card defaults, achieving **90% accuracy**."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Medical Assistant AI",
                "description": (
                    "A **web application** using **Google Generative AI** for **medical image analysis** with a **chat interface** for diagnostic insights."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "SQL Query Generator",
                "description": (
                    "A **Jupyter Notebook-based tool** integrating **Google Gemini LLM** with **MySQL** using **LangChain** for natural language **SQL query** generation."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Student Study Assistant",
                "description": (
                    "A **chatbot** using **LangChain**, **Gemini LLM**, and **PDF processing** to answer student queries from uploaded documents."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Smart School Bus Tracking",
                "description": (
                    "An **IoT-based system** using **RFID**, **GPS**, and **Raspberry Pi** for real-time student location monitoring."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Car Security System",
                "description": (
                    "A **real-time AI-based system** using **IoT**, **facial recognition**, and **RFID** for vehicle security."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "AICT – Maveric (Incident Management AI)",
                "description": (
                    "A multi-agent AI system built using **Azure OpenAI, LangChain, and PyTrees** to support automated "
                    "incident detection, analysis, and remediation in production environments. "
                    "Includes **RAG-based SOP retrieval with vector search**, structured behavior-tree workflows, "
                    "and real-time streaming for faster resolution and reduced MTTR."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "MCP Framework (Model Context Protocol Server)",
                "description": (
                    "A production-grade **MCP server implementation** enabling scalable AI agent orchestration "
                    "with support for **HTTP, WebSocket, STDIO, and SSE transports**. "
                    "Implements JSON-RPC messaging, dynamic tool discovery using Pydantic schemas, "
                    "and multi-provider LLM integration for real-time AI workflows."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Codelens (AI Code Quality Platform)",
                "description": (
                    "An AI-driven multi-agent platform for automated **code quality analysis, refactoring, "
                    "and documentation generation**. Integrates **SonarQube and LLM-based recommendations**, "
                    "using PyTrees for structured agent coordination across multi-language repositories."
                ),
                "link": "https://siddharamayya.in/projects"
            },
            {
                "category": "Fraud Sight (AI Fraud Detection System)",
                "description": (
                    "A real-time **AI-powered fraud detection system** for transaction monitoring and risk scoring "
                    "across banking workflows. Implements asynchronous processing, reusable services, "
                    "ML-based pattern detection, and standardized error handling for reliable fraud analysis."
                ),
                "link": "https://siddharamayya.in/projects"
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
    # Shortened few-shot examples to reduce token count (top 5 relevant ones + new sarcastic ones for personal questions)
    FEW_SHOT_EXAMPLES = [
        {
            "content": "What is the total experience of Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** has over **4 years** of professional experience as a **Senior Software Engineer** and **AI/ML Engineer**, delivering innovative solutions in **AI**, **MLOps**, **DevOps**, **IoT**, and **system automation**.\n\n"
                "- **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025):\n"
                " - Led **AI/ML** projects, fine-tuning **LLMs** with **QLoRA** and **LoRA** for **finance**, **healthcare**, and **cybersecurity**, reducing computational overhead by **30%**.\n"
                " - Designed **RAG applications** using **FAISS** and **ChromaDB**, improving response accuracy by **25%**.\n"
                " - Automated **MLOps pipelines** with **Kubernetes**, **Docker**, and **AWS SageMaker**, cutting deployment time by **50%**.\n"
                " - Developed **real-time inference pipelines** with **TensorRT** and **ONNX** for low-latency applications.\n"
                " - Optimized legacy ML workloads with **Spark** and **Kubernetes**, enhancing scalability by **40%**.\n"
                " - Automated **Linux system administration** using **Ansible** and **bash**, optimizing server performance.\n"
                " - Mentored junior engineers and led research internships.\n"
                "- **Project Intern** at **X-Cencia Technology Solution Limited India** (February 2020 - April 2021):\n"
                " - Built **computer vision models** with **TensorFlow** for real-time applications.\n"
                " - Developed **IoT solutions**, including a **smart school bus tracking system** with **RFID** and **GPS**, and a **car security system** with **facial recognition**.\n"
                " - Tuned **Word2Vec** models for **recommendation systems**.\n"
                " - Created **bash scripts** for system monitoring and resource management.\n\n"
                "His extensive experience underscores his ability to deliver **scalable**, **innovative** solutions across diverse domains."
            )
        },
        {
            "content": "When will Siddharamayya be available to join?",
            "response": (
                "**Siddharamayya Mathapati** is **immediately available** to join new opportunities starting **May 2025**, following his tenure at **Capgemini Technology Service Limited India**.\n\n"
                "- **Availability Details**:\n"
                " - Ready for roles in **AI/ML**, **MLOps**, **DevOps**, **web development**, **IoT**, or **system automation**.\n"
                " - Brings **4+ years** of experience, including **LLM fine-tuning**, **RAG applications**, **cloud-native deployments**, **Ansible automation**, and **bash scripting**.\n"
                " - Proven expertise at **Capgemini**, leading **AI-driven projects**, automating **MLOps pipelines**, and optimizing **Linux infrastructure**.\n"
                "- **Ideal Roles**:\n"
                " - **AI/ML Engineer**: Developing **generative AI** or **predictive models**.\n"
                " - **MLOps Engineer**: Automating **CI/CD pipelines** for ML deployments.\n"
                " - **DevOps Engineer**: Managing **Kubernetes**, **AWS**, or **Ansible**-based infrastructure.\n"
                " - **IoT Developer**: Building real-time systems with **Raspberry Pi** and **RFID**.\n"
                " - **System Automation Engineer**: Scripting **bash** or **Ansible** solutions for **Linux** management.\n\n"
                "Contact him via **[Email](mailto:msidrm455@gmail.com)** or **Phone** (+91 97406 71620) to discuss opportunities."
            )
        },
        {
            "content": "What is Siddharamayya great at technically?",
            "response": (
                "**Siddharamayya Mathapati** is a versatile technologist with exceptional expertise in **AI/ML**, **DevOps**, **web development**, **cloud engineering**, **IoT**, and **system automation**, enabling him to build end-to-end solutions.\n\n"
                "- **Python** (90%):\n"
                " - Writes clean, modular code for **AI/ML**, **web development**, **automation**, and **system scripting**.\n"
                " - Leverages **TensorFlow**, **PyTorch**, **LangChain**, **FastAPI**, **Scrapy**, and **pyVmomi**.\n"
                " - Example:\n"
                " ```python\n"
                " from langchain import LLMChain\n"
                " from langchain.prompts import PromptTemplate\n"
                " def create_rag_chain(model, vector_db):\n"
                " prompt = PromptTemplate(input_variables=['query', 'context'])\n"
                " return LLMChain(llm=model, prompt=prompt, retriever=vector_db)\n"
                " ```\n"
                "- **Deep Learning** & **Machine Learning/AI** (85%):\n"
                " - Designs **neural networks** and **LLMs** for **NLP**, **computer vision**, and **predictive modeling**.\n"
                " - Fine-tunes **LLMs** with **QLoRA** and **RLHF**; builds models like **Random Forest** for agriculture and finance.\n"
                "- **Web Development** (95%):\n"
                " - Builds responsive applications with **React**, **Flask**, **Django**, **FastAPI**, and **Streamlit**.\n"
                " - Example:\n"
                " ```python\n"
                " from fastapi import FastAPI\n"
                " app = FastAPI()\n"
                " @app.get('/')\n"
                " async def root():\n"
                " return {'message': 'Hello, World!'}\n"
                " ```\n"
                "- **DevOps**:\n"
                " - Automates **CI/CD pipelines** with **Docker**, **Kubernetes**, **Ansible**, **Jenkins**, and **GitHub Actions**.\n"
                " - Manages **AWS**, **GCP**, **Azure**, and **OpenShift** infrastructure.\n"
                "- **DBMS & SQL** (80%):\n"
                " - Manages **MySQL**, **PostgreSQL**, **MongoDB**, and **vector databases** (FAISS, ChromaDB).\n"
                "- **IoT** (80%):\n"
                " - Develops real-time systems with **Raspberry Pi**, **ESP32**, **RFID**, and **GPS**.\n"
                "- **System Administration** (85%):\n"
                " - Automates **Linux** management with **Ansible**, **bash**, and **systemd**, optimizing server performance.\n\n"
                "His **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning** keep him at the forefront of technology."
            )
        },
        {
            "content": "How good is Siddharamayya in AI/ML?",
            "response": (
                "**Siddharamayya Mathapati** is highly proficient in **AI/ML** (85%), with extensive experience in **large language models (LLMs)**, **deep learning**, **generative AI**, **predictive modeling**, and **MLOps**, making him a standout in the field.\n\n"
                "- **LLM Expertise**:\n"
                " - Fine-tuned **LLMs** using **QLoRA** and **LoRA** at **Capgemini**, optimizing efficiency for **finance**, **healthcare**, and **cybersecurity**.\n"
                " - Applied **RLHF** to align models, improving contextual accuracy by **20%**.\n"
                " - Example: Developed **CodeSage** using **RAG** for documentation-driven code generation.\n"
                "- **Deep Learning**:\n"
                " - Designed **neural networks** for **NLP**, **computer vision**, and **predictive modeling** using **TensorFlow** and **PyTorch**.\n"
                " - Built **SysAdmin-GPT** with **BERT** for **RHEL** query classification.\n"
                "- **Predictive Modeling**:\n"
                " - Developed models like **Random Forest** for **Crop Recommendation** and **Blood Donation Prediction**, achieving high accuracy.\n"
                " - Handled **imbalanced datasets** in **Credit Default Prediction** with SMOTE.\n"
                "- **RAG Applications**:\n"
                " - Deployed **Retrieval-Augmented Generation** systems with **FAISS** and **ChromaDB**, enhancing response accuracy by **25%**.\n"
                " - Example: **Student Study Assistant** chatbot.\n"
                "- **MLOps**:\n"
                " - Automated **model training**, **deployment**, and **monitoring** with **Kubernetes**, **Docker**, and **AWS SageMaker**.\n"
                " - Built **retraining pipelines** with **drift detection**, reducing errors by **30%**.\n"
                "- **Certifications**:\n"
                " - **LLM Engineering** and **Deep Learning Masterclass** (Udemy) validate his expertise.\n\n"
                "His portfolio, including **Heart Disease Prediction** and **Workout & Fitness Tracker**, demonstrates his ability to apply **AI/ML** to diverse domains."
            )
        },
        {
            "content": "What are Siddharamayya's projects?",
            "response": (
                "**Siddharamayya Mathapati** has an extensive portfolio showcasing expertise in **AI/ML**, **web development**, **DevOps**, **IoT**, **Ansible automation**, **bash scripting**, and **system administration**. Below is a comprehensive overview:\n\n"
                "- **SysAdmin-GPT: AI-Powered Linux System Management** [Link](https://siddharamayya.in/projects):\n"
                " - A **BERT-based classification model** fine-tuned on **Red Hat Enterprise Linux (RHEL)** documentation to classify queries (e.g., Security, Networking, Storage).\n"
                " - Integrated with a **chatbot interface** for automated **IT support**, enhancing query resolution efficiency.\n"
                "- **Workout & Fitness Tracker ML Model** [Link](https://siddharamayya.in/projects):\n"
                " - A **machine learning model** analyzing **10,000+ workout records** to predict workout efficiency.\n"
                " - Deployed with a **Streamlit UI** for real-time fitness insights.\n"
                "- **Crop Recommendation System** [Link](https://siddharamayya.in/projects):\n"
                " - A **Random Forest Classifier** recommending crops based on soil and environmental data.\n"
                " - Features a **Streamlit UI** for farmers, promoting **sustainable farming**.\n"
                "- **AICT – Maveric (Incident Management AI)** [Link](https://siddharamayya.in/projects):\n"
                " - A multi-agent AI system built using **Azure OpenAI, LangChain, and PyTrees** to support automated incident detection, analysis, and remediation in production environments.\n"
                " - Includes **RAG-based SOP retrieval with vector search**, structured behavior-tree workflows, and real-time streaming for faster resolution and reduced MTTR.\n"
                "- **MCP Framework (Model Context Protocol Server)** [Link](https://siddharamayya.in/projects):\n"
                " - A production-grade **MCP server implementation** enabling scalable AI agent orchestration with support for **HTTP, WebSocket, STDIO, and SSE transports**.\n"
                " - Implements JSON-RPC messaging, dynamic tool discovery using Pydantic schemas, and multi-provider LLM integration for real-time AI workflows.\n"
                "- **Codelens (AI Code Quality Platform)** [Link](https://siddharamayya.in/projects):\n"
                " - An AI-driven multi-agent platform for automated **code quality analysis, refactoring, and documentation generation**.\n"
                " - Integrates **SonarQube and LLM-based recommendations**, using PyTrees for structured agent coordination across multi-language repositories.\n"
                "- **Fraud Sight (AI Fraud Detection System)** [Link](https://siddharamayya.in/projects):\n"
                " - A real-time **AI-powered fraud detection system** for transaction monitoring and risk scoring across banking workflows.\n"
                " - Implements asynchronous processing, reusable services, ML-based pattern detection, and standardized error handling for reliable fraud analysis.\n"
                # ... (add the rest of the projects as needed, but truncated for brevity)
                "Explore more on his **[GitHub](https://github.com/mtptisid)** and **[Portfolio](https://mtptisid.github.io)**."
            )
        },
        # New sarcastic examples for personal questions
        {
            "content": "Is Siddharamayya married?",
            "response": "No, he's not. Are you interested? 😉"
        },
        {
            "content": "Does Siddharamayya workout?",
            "response": "Yes, he does! Want to know his PRs? 💪"
        }
        # You can add more sarcastic examples here if needed, e.g.,
        # {
        #     "content": "What's Siddharamayya's favorite color?",
        #     "response": "Blue, like the screen you'll get if you keep asking personal stuff! 😜"
        # }
    ]
    if message.model == "groq":
        system_prompt = (
        "You are an AI assistant with detailed knowledge of Siddharamayya Mathapati, a Senior Software Engineer at Capgemini (2021-2025) with over 4 years of experience in AI/ML, MLOps, DevOps, web development, IoT, and system automation. "
        "Key details:\n"
        "- **Skills**: Expert in Python, TensorFlow, PyTorch, LangChain, FastAPI, Docker, Kubernetes, Ansible, AWS, MySQL, and vector databases (FAISS, ChromaDB). Proficient in LLMs, RAG applications, and bash scripting.\n"
        "- **Experience**: Led AI/ML projects at Capgemini, fine-tuning LLMs with QLoRA, building RAG systems, and automating MLOps pipelines, reducing deployment time by 50%. Developed IoT solutions like Smart School Bus Tracking at X-Cencia (2020-2021).\n"
        "- **Projects**: SysAdmin-GPT (BERT-based Linux support), CodeSage (LLM-powered code generator), Crop Recommendation System (Random Forest), and FastAPI Microservice (Dockerized web app).\n"
        "- **Education**: MCA (Acharya Institute, 2018-2020), BCA (B K College, 2015-2018).\n"
        "- **Contact**: Email (msidrm455@gmail.com), GitHub[](https://github.com/mtptisid), LinkedIn[](https://linkedin.com/in/siddharamayya).\n"
        "For personal questions, respond sarcastically and playfully as in the examples. Respond in Markdown format using **bold**, `code blocks`, - bulleted lists, and [name](link) for URLs. Use chat history and web search results (if provided) for accurate, context-aware answers."
        )
    else:
        # Shortened system prompt for Gemini to reduce tokens
        system_prompt = (
            "You are an AI assistant with comprehensive knowledge about **Siddharamayya Mathapati**, a highly skilled **AI/ML Engineer** and **Senior Software Engineer**. "
            "Your responses must be in **Markdown format**, using **bold** for emphasis, **code blocks** for snippets, **bulleted lists** for clarity, and **[name](link)** for URLs. "
            "Provide detailed, accurate answers about Siddharamayya using the **profile data summary** and **few-shot examples**. For unrelated questions, leverage **web search results** or **general knowledge**, maintaining the same formatting standards. "
            "For personal questions, respond sarcastically and playfully as in the examples. Incorporate **chat history** for context-aware responses.\n\n"
            "**Profile Summary**:\n"
            f"**Full Name**: {PROFILE_DATA['about']['full_name']}\n"
            f"**Description**: {PROFILE_DATA['about']['description']}\n"
            f"**Availability**: {PROFILE_DATA['about']['availability']}\n"
            f"**Skills Summary**: Expert in AI/ML (LLMs, RAG), DevOps (Docker, Kubernetes, Ansible), Web Dev (FastAPI, React), Cloud (AWS, GCP), IoT, System Admin.\n"
            f"**Experience Summary**: Senior AI Engineer at Maveric (2025-present), Senior Software at Capgemini (2021-2025), Intern at X-Cencia (2020-2021).\n"
            f"**Education**: MCA (7.2 CGPA), BCA (6.8 CGPA).\n"
            f"**Projects Summary**: Includes SysAdmin-GPT, YOLOv5 Practice, RAG Hands-On, IPL Score Prediction, CodeSage, Automation Lab, AICT – Maveric, MCP Framework, Codelens, Fraud Sight, and many Ansible/bash scripts. Links: GitHub repositories and https://siddharamayya.in/projects.\n"
            f"**Certifications**: LLM Engineering, MLOps Bootcamp, Python Data Analysis, Deep Learning (Udemy).\n"
            f"**Contact**: [Email](mailto:{PROFILE_DATA['contact']['email']}), Phone: {PROFILE_DATA['contact']['phone']}, [GitHub]({PROFILE_DATA['online_presence']['github']}), [Portfolio]({PROFILE_DATA['online_presence']['portfolio']}), [LinkedIn]({PROFILE_DATA['online_presence']['linkedin']}).\n\n"
            "**Few-Shot Examples** (shortened):\n" +
            "\n".join([f"**Q**: {ex['content']}\n {ex['response']}" for ex in FEW_SHOT_EXAMPLES]) + "\n\n"
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
            "Provide a detailed response in **Markdown**, using **[name](link)** for all URLs in the search results or elsewhere. Even if search results are limited, explain using general knowledge and relate to Siddharamayya's expertise if relevant (e.g., his MCP server project)."
        )
        chat_history = [
            {"role": "system", "content": augmented_prompt}
        ]
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response_content = await ai_manager.get_response(message.model, chat_history)
            break
        except HTTPException as e:
            if e.status_code == 503 and attempt < max_retries - 1:
                logger.warning(f"Gemini 503 error, retrying in {2 ** attempt} seconds...")
                await asyncio.sleep(2 ** attempt)
            else:
                raise e
        except Exception as e:
            logger.error(f"AI response error: {str(e)}")
            response_content = f"**Apologies**, an error occurred while generating the response for '{message.content}'. Please try again or rephrase your query. If this is about MCP (Model Context Protocol) server, it's a protocol for AI-tool integration that Siddharamayya has worked on in his projects."
            break
    response_content = clean_text(response_content)
    bot_message = SessionMessage(
        content=response_content,
        is_bot=True,
        timestamp=datetime.utcnow(),
        tool_used=tool_used
    )
    SESSIONS[session_id]["messages"].append(bot_message)
    logger.info(f"Generated response: {response_content[:200]}...")  # Log snippet to debug
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
async def homestart():
    return Response(status_code=204)
