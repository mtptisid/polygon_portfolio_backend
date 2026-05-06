from fastapi import HTTPException
import openai
import aiohttp
from groq import Groq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from typing import List, Dict
import os
from dotenv import load_dotenv

# Load .env file for local development
load_dotenv()

# --------------------------
# OpenAI GPT Service
# --------------------------
class OpenAIService:
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key or self.api_key == "your_openai_api_key_here":
            raise ValueError("OPENAI_API_KEY not set in environment variables")
        openai.api_key = self.api_key
        self.model = "gpt-3.5-turbo"

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model=self.model,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message['content']
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"OpenAI error: {str(e)}"
            )

# --------------------------
# Gemini Service
# --------------------------
_ROLE_MAP = {
    "system": SystemMessage,
    "user": HumanMessage,
    "assistant": AIMessage,
}

class GeminiService:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError("GEMINI_API_KEY not set in environment variables")
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        lc_messages = [
            _ROLE_MAP.get(msg["role"], HumanMessage)(content=msg["content"])
            for msg in messages
        ]
        try:
            response = await self.llm.ainvoke(lc_messages)
            return response.content
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Gemini error: {str(e)}"
            )

# --------------------------
# Groq Service
# --------------------------
class GroqService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY")
        if not self.api_key or self.api_key == "your_groq_api_key_here":
            raise ValueError("GROQ_API_KEY not set in environment variables")
        self.model = "llama-3.3-70b-versatile" #llama3-70b-8192 and llama3-8b-8192 will be deprecated from GroqCloud™.
        self.client = Groq(api_key=self.api_key)

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model
            )
            return chat_completion.choices[0].message.content
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Groq API error: {str(e)}"
            )

# --------------------------
# Deepseek Service
# --------------------------
class DeepseekService:
    def __init__(self):
        self.api_key = os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key or self.api_key == "your_deepseek_api_key_here":
            raise ValueError("DEEPSEEK_API_KEY not set in environment variables")
        self.model = "deepseek-chat"
        self.base_url = "https://api.deepseek.com/v1/chat/completions"

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7
        }
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(self.base_url, headers=headers, json=data) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return result['choices'][0]['message']['content']
        except aiohttp.ClientResponseError as e:
            raise HTTPException(
                status_code=e.status,
                detail=f"Deepseek API error: {e.message}"
            )
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Deepseek connection error: {str(e)}"
            )

# --------------------------
# Unified AI Router
# --------------------------
class AIManager:
    def __init__(self):
        """Initialize AI services with error handling"""
        self.services = {}
        
        # Try to initialize each service, skip if API key is missing
        try:
            self.services["openai"] = OpenAIService()
        except Exception as e:
            logger.warning(f"OpenAI service not available: {e}")
        
        try:
            self.services["gemini"] = GeminiService()
        except Exception as e:
            logger.warning(f"Gemini service not available: {e}")
        
        try:
            self.services["groq"] = GroqService()
        except Exception as e:
            logger.warning(f"Groq service not available: {e}")
        
        try:
            self.services["deepseek"] = DeepseekService()
        except Exception as e:
            logger.warning(f"DeepSeek service not available: {e}")
        
        if not self.services:
            logger.error("No AI services available! Please set API keys.")

    async def get_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        if model not in self.services:
            # Try to find any available service
            if self.services:
                available = list(self.services.keys())[0]
                logger.warning(f"Model {model} not available, using {available}")
                model = available
            else:
                raise HTTPException(status_code=503, detail="No AI services available")
        
        return await self.services[model].get_response(messages)

# Export singleton
ai_manager = AIManager()
