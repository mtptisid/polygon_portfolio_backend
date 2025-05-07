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
    sites = ["github.com", "linkedin.com"]  # Fixed sites
    
    site_queries = [f"site:{site}" for site in sites]
    full_query = f"{query} {' OR '.join(site_queries)}"
    logger.info(f"Performing search with query: {full_query}")
    
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
        logger.info(f"Formatted search results: {formatted_results}")
        return formatted_results
    except Exception as e:
        logger.error(f"Search failed: {str(e)}")
        return f"Web Search Results: Search failed: {str(e)}"

def clean_text(text: str) -> str:
    """Clean text by removing excessive newlines and unwanted characters."""
    # Replace 3+ newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Remove unwanted control characters
    text = re.sub(r'[\r\t]', '', text)
    # Trim leading/trailing whitespace
    text = text.strip()
    return text

@router.post("/request", response_model=MessageResponse)
async def send_message(request: Request, message: MessageCreate):
    """Send a message to the selected AI model, optionally using tools."""
    # Log raw request payload for debugging
    raw_body = await request.body()
    logger.info(f"Raw request payload: {raw_body.decode('utf-8')}")
    
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
            "full_name": "Siddharamayya Mathapati",
            "dob": "April 14, 1997",
            "languages": ["English", "Kannada", "Hindi", "Telugu", "Tamil", "Marathi", "Malayalam"],
            "email": "msidrm455@gmail.com",
            "phone": "+91 97406 71620",
            "location": "Yadur, Chikodi, Belagavi, Karnataka, India",
            "description": (
                "Siddharamayya Mathapati is a seasoned **AI/ML Engineer** and **Senior Software Engineer** with a passion for leveraging cutting-edge technologies to solve complex problems. "
                "With over **4 years** of experience, he specializes in **large language models (LLMs)**, **generative AI**, **MLOps**, and **DevOps**, delivering scalable and innovative solutions. "
                "His expertise spans **cloud platforms** (AWS, GCP, Azure), **containerization** (Docker, Kubernetes), and **automation**, ensuring high-performance deployments. "
                "Siddharamayya is committed to continuous learning, as evidenced by his **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning**. "
                "He thrives in collaborative environments, mentoring junior engineers and driving process optimization."
            ),
            "availability": "Immediately available for new opportunities as of May 2025."
        },
        "skills": [
            {"name": "Python", "proficiency": "90%", "description": "Expert in writing efficient, modular code for **AI/ML**, **web development**, and **automation**, using libraries like **TensorFlow**, **PyTorch**, **LangChain**, and **FastAPI**."},
            {"name": "Deep Learning", "proficiency": "85%", "description": "Skilled in designing and training **neural networks** for tasks like **image analysis**, **NLP**, and **predictive modeling**, with tools like **TensorFlow** and **PyTorch**."},
            {"name": "Machine Learning/AI", "proficiency": "85%", "description": "Proficient in building **predictive models**, **LLMs**, and **RAG applications**, with expertise in **fine-tuning** and **RLHF**."},
            {"name": "Data Science", "proficiency": "85%", "description": "Experienced in **data preprocessing**, **visualization**, and **statistical analysis**, using tools like **Pandas**, **NumPy**, and **Matplotlib**."},
            {"name": "DevOps", "tools": ["Docker", "Kubernetes", "Ansible", "Jenkins", "GitHub Actions"], "description": "Adept at automating **CI/CD pipelines**, managing **cloud infrastructure**, and optimizing deployments."},
            {"name": "Web Development", "proficiency": "95%", "description": "Builds responsive, user-friendly applications using **React**, **Flask**, **Django**, and **Streamlit**, with a focus on scalability and UX."},
            {"name": "DBMS & SQL", "proficiency": "80%", "description": "Proficient in managing **relational** (MySQL, PostgreSQL) and **NoSQL** (MongoDB, ChromaDB) databases, including **vector databases** for **RAG**."},
            {"name": "Cloud Engineering", "proficiency": "85%", "description": "Experienced with **AWS** (SageMaker, EMR, Lambda), **GCP**, and **Azure**, optimizing costs and ensuring high availability."}
        ],
        "experience": [
            {
                "title": "Senior Software Engineer",
                "company": "Capgemini Technology Service Limited India",
                "location": "Navi Mumbai, India",
                "duration": "May 2021 - April 2025",
                "description": (
                    "Led **AI/ML** initiatives, focusing on **LLM fine-tuning** using **QLoRA** and **LoRA**, optimizing model efficiency and reducing computational overhead. "
                    "Designed and deployed **RAG applications** with **vector databases** (FAISS, ChromaDB) for domain-specific use cases in **finance**, **healthcare**, and **cybersecurity**. "
                    "Spearheaded **MLOps** pipelines, automating model training, deployment, and monitoring using **Kubernetes**, **Docker**, and **AWS SageMaker**. "
                    "Developed **real-time inference pipelines** with **TensorRT** and **ONNX**, achieving low-latency deployments. "
                    "Mentored junior engineers, supervised research internships, and optimized legacy ML workloads by migrating to **Spark** and **Kubernetes**, improving scalability by **40%**. "
                    "Built **model retraining pipelines** with **drift detection**, reducing prediction errors by **30%**."
                )
            },
            {
                "title": "Project Intern",
                "company": "X-Cencia Technology Solution Limited India",
                "location": "Bengaluru, India",
                "duration": "February 2020 - April 2021",
                "description": (
                    "Developed **computer vision models** using **TensorFlow** for real-time applications. "
                    "Supported **NLP research** by tuning **Word2Vec** models for recommendation systems. "
                    "Designed **IoT-based solutions**, including a **smart school bus tracking system** using **RFID**, **GPS**, and **Raspberry Pi**, and a **real-time car security system** with **facial recognition**. "
                    "Built **automated data pipelines** for processing sensor data, enhancing system efficiency."
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
                    "Developed a **Python-based automation tool** using **Selenium** and **BeautifulSoup** to extract user activity data (posts, likes, shares, comments) from **LinkedIn**. "
                    "Implemented **dynamic content handling** and **error management** for robust scraping, ensuring ethical practices. "
                    "The tool supports **data analysis** for social media insights."
                ),
                "link": "[LinkedIn Comment Poster](https://github.com/mtptisid/linkedin-comment-poster)"
            },
            {
                "category": "Recruitment Automation",
                "description": (
                    "Created an **AI-driven solution** to automate job data extraction from career pages and generate personalized **cold emails** for outreach. "
                    "Leveraged **NLP** and **web scraping** to enhance efficiency, reducing manual effort by **80%**."
                ),
                "link": "[Recruitment Solution](https://github.com/mtptisid/recruitment-solution)"
            },
            {
                "category": "Stock Price Prediction",
                "description": (
                    "Built an **AI-powered finance agent** using **LLMs** to predict stock prices and analyze market trends. "
                    "Integrated **Yahoo Finance** for stock data and **NewsAPI** for sentiment analysis, improving prediction accuracy by **25%**."
                ),
                "link": "[Stock Price Prediction](https://github.com/mtptisid/stock-price-prediction)"
            },
            {
                "category": "Credit Risk Analysis",
                "description": (
                    "Developed a **machine learning model** using **CatBoostClassifier** to predict credit card defaults. "
                    "Achieved **90% accuracy** in assessing creditworthiness, supporting financial institutions in risk management."
                ),
                "link": "[Credit Risk Analysis](https://github.com/mtptisid/credit-risk-analysis)"
            },
            {
                "category": "Medical Assistant AI",
                "description": (
                    "Created a **web application** using **Google Generative AI** to analyze medical images and provide diagnostic insights. "
                    "Features a **chat interface** for user interaction, enhancing support for medical professionals."
                ),
                "link": "[Medical Assistant AI](https://github.com/mtptisid/medical-assistant-ai)"
            },
            {
                "category": "SQL Query Generator",
                "description": (
                    "Built a **Jupyter Notebook-based tool** integrating **Google Gemini LLM** with **MySQL** using **LangChain**. "
                    "Generates accurate **SQL queries** from natural language inputs, leveraging **few-shot learning** for precision."
                ),
                "link": "[Simple SQL with Gemini](https://github.com/mtptisid/simple-sql-gemini)"
            },
            {
                "category": "Student Study Assistant",
                "description": (
                    "Developed a **chatbot** using **LangChain**, **Gemini LLM**, and **PDF processing** to answer student queries based on uploaded documents. "
                    "Improves study efficiency by providing context-aware responses."
                ),
                "link": "[Student Study Assistant](https://github.com/mtptisid/student-study-assistant)"
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
            "Natural Language Processing & Chatbots",
            "Cloud-Native MLOps & DevOps",
            "IoT and Real-Time Systems"
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

    # Enhanced few-shot examples with detailed, Markdown-formatted answers
    FEW_SHOT_EXAMPLES = [
        {
            "content": "What is the total experience of Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** has accumulated over **4 years** of professional experience as a **Senior Software Engineer** and **AI/ML Engineer**, delivering innovative solutions in **AI**, **MLOps**, and **DevOps**.\n\n"
                "- **Senior Software Engineer** at **Capgemini Technology Service Limited India** (May 2021 - April 2025):\n"
                "  - Led **AI/ML** projects, including **LLM fine-tuning** with **QLoRA** and **LoRA**, optimizing model performance for **finance**, **healthcare**, and **cybersecurity** applications.\n"
                "  - Designed **RAG applications** with **vector databases** (FAISS, ChromaDB), enhancing response accuracy for domain-specific tasks.\n"
                "  - Automated **MLOps pipelines** using **Kubernetes**, **Docker**, and **AWS SageMaker**, reducing deployment time by **50%**.\n"
                "  - Developed **real-time inference pipelines** with **TensorRT** and **ONNX**, achieving **low-latency** model deployments.\n"
                "  - Mentored junior engineers and optimized legacy ML workloads, improving scalability by **40%**.\n"
                "- **Project Intern** at **X-Cencia Technology Solution Limited India** (February 2020 - April 2021):\n"
                "  - Built **computer vision models** using **TensorFlow** for real-time applications.\n"
                "  - Developed **IoT-based solutions**, including a **smart school bus tracking system** and a **car security system** with **facial recognition**.\n"
                "  - Supported **NLP research** by tuning **Word2Vec** models for recommendation systems.\n\n"
                "This extensive experience highlights his ability to bridge **development** and **operations**, delivering high-impact solutions."
            )
        },
        {
            "content": "What is Siddharamayya great at technically?",
            "response": (
                "**Siddharamayya Mathapati** is a versatile technologist with exceptional skills across **AI/ML**, **DevOps**, **web development**, and **cloud engineering**, enabling him to build end-to-end solutions.\n\n"
                "- **Python** (90%):\n"
                "  - Writes clean, modular code for **AI/ML**, **web development**, and **automation**.\n"
                "  - Leverages libraries like **TensorFlow**, **PyTorch**, **LangChain**, and **FastAPI** for scalable applications.\n"
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
                "  - Fine-tunes **LLMs** with **QLoRA** and **RLHF**, optimizing for domain-specific tasks.\n"
                "- **Web Development** (95%):\n"
                "  - Builds responsive applications with **React**, **Flask**, **Django**, and **Streamlit**, ensuring seamless UX.\n"
                "  - Example:\n"
                "    ```python\n"
                "    from flask import Flask, render_template\n"
                "    app = Flask(__name__)\n"
                "    @app.route('/')\n"
                "    def home():\n"
                "        return render_template('index.html')\n"
                "    ```\n"
                "- **DevOps**:\n"
                "  - Automates **CI/CD pipelines** using **Docker**, **Kubernetes**, **Ansible**, and **GitHub Actions**.\n"
                "  - Optimizes **cloud infrastructure** on **AWS**, **GCP**, and **Azure** for cost and performance.\n"
                "- **DBMS & SQL** (80%):\n"
                "  - Manages **relational** (MySQL, PostgreSQL) and **NoSQL** (MongoDB, ChromaDB) databases.\n"
                "  - Implements **vector databases** for **RAG** applications.\n\n"
                "His technical prowess is complemented by **Udemy certifications** in **LLM Engineering**, **MLOps**, and **Deep Learning**, ensuring he stays at the forefront of technology."
            )
        },
        {
            "content": "What are Siddharamayya's projects?",
            "response": (
                "**Siddharamayya Mathapati** has an impressive portfolio of projects showcasing his expertise in **AI/ML**, **web development**, **NLP**, and **IoT**. Below is a detailed overview:\n\n"
                "- **LinkedIn Activity Scraper** ([GitHub](https://github.com/mtptisid/linkedin-comment-poster)):\n"
                "  - A **Python-based tool** using **Selenium** and **BeautifulSoup** to extract **LinkedIn** user activity (posts, likes, shares, comments).\n"
                "  - Features **dynamic content handling** and **error management**, ensuring ethical scraping for **social media analytics**.\n"
                "- **Recruitment Automation** ([GitHub](https://github.com/mtptisid/recruitment-solution)):\n"
                "  - An **AI-driven solution** for extracting job data from career pages and generating personalized **cold emails**.\n"
                "  - Uses **NLP** and **web scraping**, reducing manual effort by **80%**.\n"
                "- **Stock Price Prediction** ([GitHub](https://github.com/mtptisid/stock-price-prediction)):\n"
                "  - An **AI-powered finance agent** using **LLMs** to predict stock prices.\n"
                "  - Integrates **Yahoo Finance** and **NewsAPI**, improving accuracy by **25%** through sentiment analysis.\n"
                "- **Credit Risk Analysis** ([GitHub](https://github.com/mtptisid/credit-risk-analysis)):\n"
                "  - A **machine learning model** using **CatBoostClassifier** to predict credit card defaults.\n"
                "  - Achieves **90% accuracy**, supporting **financial risk management**.\n"
                "- **Medical Assistant AI** ([GitHub](https://github.com/mtptisid/medical-assistant-ai)):\n"
                "  - A **web application** using **Google Generative AI** for **medical image analysis** and diagnostic insights.\n"
                "  - Features a **chat interface** for seamless user interaction.\n"
                "- **SQL Query Generator** ([GitHub](https://github.com/mtptisid/simple-sql-gemini)):\n"
                "  - A **Jupyter Notebook-based tool** integrating **Google Gemini LLM** with **MySQL**.\n"
                "  - Generates **SQL queries** from natural language using **LangChain** and **few-shot learning**.\n"
                "- **Student Study Assistant** ([GitHub](https://github.com/mtptisid/student-study-assistant)):\n"
                "  - A **chatbot** built with **LangChain**, **Gemini LLM**, and **PDF processing**.\n"
                "  - Answers student queries based on uploaded documents, enhancing study efficiency.\n\n"
                "Explore more on his **[GitHub](https://github.com/mtptisid)** and **[Portfolio](https://mtptisid.github.io)**."
            )
        },
        {
            "content": "What is Siddharamayya's GitHub page?",
            "response": (
                "**Siddharamayya Mathapati**’s **GitHub** profile is a rich repository of his technical contributions, showcasing his expertise in **AI/ML**, **web development**, and **DevOps**.\n\n"
                "- **[GitHub](https://github.com/mtptisid)**:\n"
                "  - Features projects like **LinkedIn Activity Scraper**, **Stock Price Prediction**, and **Medical Assistant AI**, demonstrating proficiency in **Python**, **LLMs**, and **web scraping**.\n"
                "  - Includes **DevOps** workflows with **Docker**, **Kubernetes**, and **GitHub Actions** for automated deployments.\n"
                "  - Showcases **open-source contributions**, reflecting his commitment to collaborative development.\n\n"
                "Visit his **[Portfolio](https://mtptisid.github.io)** for a curated overview of his work."
            )
        },
        {
            "content": "How can I contact Siddharamayya?",
            "response": (
                "**Siddharamayya Mathapati** is readily accessible for professional inquiries, collaborations, or opportunities.\n\n"
                "- **[Email](mailto:msidrm455@gmail.com)**:\n"
                "  - Preferred for formal communication, project proposals, or job opportunities.\n"
                "  - Example: Reach out for **AI/ML** consulting or **DevOps** pipeline optimization.\n"
                "- **Phone**: +91 97406 71620\n"
                "  - Available for direct calls or messages to discuss technical projects or collaborations.\n"
                "  - Best for urgent or real-time discussions.\n"
                "- **Address**: #372, Ward No. 3, Yadur, Chikodi, Belagavi, Karnataka, India\n"
                "  - Suitable for official correspondence or in-person meetings (by appointment).\n"
                "- **[LinkedIn](https://linkedin.com/in/siddharamayya)**:\n"
                "  - Ideal for professional networking and viewing his latest updates or endorsements.\n\n"
                "Siddharamayya is responsive and open to discussing **AI**, **MLOps**, or **web development** opportunities."
            )
        },
        {
            "content": "What languages does Siddharamayya speak?",
            "response": (
                "**Siddharamayya Mathapati** is a multilingual professional, fluent in **seven languages**, which enhances his ability to collaborate across diverse teams and regions.\n\n"
                "- **Kannada**:\n"
                "  - Native language; fluent in speaking, reading, and writing, used in daily communication and professional settings in **Karnataka**.\n"
                "- **English**:\n"
                "  - Professional fluency; proficient in technical writing, presentations, and global collaboration, essential for **AI/ML** and **DevOps** projects.\n"
                "- **Hindi**:\n"
                "  - Fluent; skilled in verbal and written communication, widely used in **India** for cross-regional interactions.\n"
                "- **Telugu**:\n"
                "  - Conversational fluency; comfortable in professional and casual discussions, particularly in **South India**.\n"
                "- **Tamil**:\n"
                "  - Conversational fluency; adept at understanding and communicating in everyday and technical contexts.\n"
                "- **Marathi**:\n"
                "  - Intermediate fluency; capable of effective communication in professional and social settings, especially in **Maharashtra**.\n"
                "- **Malayalam**:\n"
                "  - Basic proficiency; can comprehend and engage in simple conversations, useful for interactions in **Kerala**.\n\n"
                "This linguistic versatility supports his work in diverse industries like **finance**, **healthcare**, and **education**."
            )
        },
        {
            "content": "When will Siddharamayya be available to join?",
            "response": (
                "**Siddharamayya Mathapati** is **immediately available** to join new opportunities starting **May 2025**, following the completion of his role at **Capgemini Technology Service Limited India**.\n\n"
                "- **Availability Details**:\n"
                "  - Ready to take on roles in **AI/ML**, **MLOps**, **web development**, or **DevOps**.\n"
                "  - Brings over **4 years** of experience, including **LLM fine-tuning**, **RAG applications**, and **cloud-native deployments**.\n"
                "  - Proven track record at **Capgemini** (May 2021 - April 2025), where he led **AI-driven projects** and mentored teams.\n"
                "- **Ideal Roles**:\n"
                "  - **AI/ML Engineer**: Developing **generative AI** or **predictive models**.\n"
                "  - **MLOps Engineer**: Automating **CI/CD pipelines** for ML deployments.\n"
                "  - **DevOps Engineer**: Managing **Kubernetes** and **AWS**-based infrastructure.\n\n"
                "Contact him at **[Email](mailto:msidrm455@gmail.com)** or **Phone** (+91 97406 71620) to discuss opportunities."
            )
        },
        {
            "content": "What certifications does Siddharamayya have?",
            "response": (
                "**Siddharamayya Mathapati** has earned several **Udemy certifications**, reflecting his commitment to continuous learning and staying updated with industry trends in **AI/ML** and **MLOps**.\n\n"
                "- **LLM Engineering: Master AI, Large Language Models & Agents**:\n"
                "  - Covers **LLM fine-tuning**, **prompt engineering**, and **agent-based AI systems**.\n"
                "  - Applied in projects like **Stock Price Prediction** and **Medical Assistant AI**.\n"
                "- **MLOps Bootcamp: Mastering AI Operations for Success**:\n"
                "  - Focuses on **CI/CD pipelines**, **model monitoring**, and **cloud deployments**.\n"
                "  - Used at **Capgemini** to automate **MLOps workflows** with **Kubernetes** and **AWS SageMaker**.\n"
                "- **Python for Data Analysis & Visualization**:\n"
                "  - Covers **Pandas**, **NumPy**, and **Matplotlib** for data processing and visualization.\n"
                "  - Applied in **Credit Risk Analysis** for data preprocessing and model evaluation.\n"
                "- **Deep Learning Masterclass with TensorFlow 2**:\n"
                "  - Explores **neural network design** and **computer vision** applications.\n"
                "  - Used in **Medical Assistant AI** for **image analysis**.\n\n"
                "These certifications enhance his ability to deliver cutting-edge solutions in **AI**, **data science**, and **DevOps**."
            )
        },
        {
            "content": "What is Siddharamayya's educational background?",
            "response": (
                "**Siddharamayya Mathapati** has a strong academic foundation in **computer applications**, equipping him with the skills to excel in **AI/ML**, **software development**, and **DevOps**.\n\n"
                "- **MCA - Master of Computer Applications**:\n"
                "  - **Institution**: Acharya Institute of Technology, Bangalore\n"
                "  - **Duration**: July 2018 - September 2020\n"
                "  - **Score**: CGPA 7.2/10\n"
                "  - Focused on **advanced programming**, **database systems**, and **software engineering**, laying the groundwork for his **AI/ML** expertise.\n"
                "- **BCA - Bachelor of Computer Applications**:\n"
                "  - **Institution**: B K College, Chikodi\n"
                "  - **Duration**: July 2015 - June 2018\n"
                "  - **Score**: CGPA 6.8/10\n"
                "  - Covered **programming fundamentals**, **web development**, and **database management**, fostering his passion for technology.\n\n"
                "His education, combined with **Udemy certifications**, supports his ability to tackle complex technical challenges."
            )
        }
    ]

    # Prepare system prompt with detailed Markdown formatting
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
            "Provide a detailed response in **Markdown**, using **[name](link)** for all URLs in the search results or elsewhere."
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
