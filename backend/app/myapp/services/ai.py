from fastapi import HTTPException
import openai
import aiohttp
import anyio
from groq import Groq
import google.generativeai as genai
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
        if not self.api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY not set in environment variables")
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
class GeminiService:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not set in environment variables")
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    async def get_response(self, messages: List[Dict[str, str]]) -> str:
        try:
            parts = [msg["content"] for msg in messages]
            response = await anyio.to_thread.run_sync(
                lambda: self.model.generate_content(parts)
            )
            return response.text
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
        if not self.api_key:
            raise HTTPException(status_code=500, detail="GROQ_API_KEY not set in environment variables")
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
        if not self.api_key:
            raise HTTPException(status_code=500, detail="DEEPSEEK_API_KEY not set in environment variables")
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
        self.services = {
            "openai": OpenAIService(),
            "gemini": GeminiService(),
            "groq": GroqService(),
            "deepseek": DeepseekService()
        }

    async def get_response(self, model: str, messages: List[Dict[str, str]]) -> str:
        if model not in self.services:
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
        return await self.services[model].get_response(messages)

# Export singleton
ai_manager = AIManager()
