"""
HR Session Analysis Generation.

Generates AI-powered analysis of recruiter sessions including:
- Conversation summary
- Key topics discussed
- Role fit analysis
- Interest level inference
- Recommended next steps
"""

import json
import logging
from typing import List, Optional
import asyncio

from myapp.models.hr_models import (
    ChatMessage,
    RecruiterInfo,
    SessionAnalysis
)

logger = logging.getLogger(__name__)


class AnalysisGenerator:
    """
    Generates structured analysis of recruiter sessions using LLM.
    """
    
    def __init__(self, ai_manager=None):
        """
        Initialize the analysis generator.
        
        Args:
            ai_manager: AI manager instance for LLM calls (optional, will import if None)
        """
        if ai_manager is None:
            from myapp.services.ai import ai_manager as default_manager
            self.ai_manager = default_manager
        else:
            self.ai_manager = ai_manager
        
        logger.info("AnalysisGenerator initialized")
    
    def _build_analysis_prompt(
        self,
        chat_history: List[ChatMessage],
        recruiter_info: RecruiterInfo
    ) -> str:
        """
        Build the prompt for analysis generation.
        
        Args:
            chat_history: List of chat messages
            recruiter_info: Recruiter information
            
        Returns:
            Formatted prompt string
        """
        # Format chat history
        formatted_history = []
        for msg in chat_history:
            role = "Recruiter" if msg.role == "user" else "Assistant"
            formatted_history.append(f"{role}: {msg.content}")
        
        history_text = "\n".join(formatted_history)
        
        # Build prompt
        prompt = f"""Analyze this recruiter conversation and provide a structured analysis.

**Recruiter Information:**
- Name: {recruiter_info.name}
- Company: {recruiter_info.company}
- Role: {recruiter_info.role or "Not specified"}
- Email: {recruiter_info.email}

**Conversation Transcript:**
{history_text}

**Instructions:**
Provide a JSON analysis with the following structure:
{{
  "conversation_summary": "3-5 sentence summary of the conversation, including recruiter's company, role interest, and main discussion points",
  "key_topics_discussed": ["topic1", "topic2", "topic3", ...],
  "role_fit_analysis": "If a role was discussed, analyze how well the candidate fits. If no role mentioned, set to null",
  "interest_level": "low|medium|high - Infer based on: number of questions, depth of questions, follow-up questions, time spent",
  "recommended_next_steps": ["step1", "step2", "step3", ...]
}}

**Interest Level Guidelines:**
- **high**: 8+ messages, detailed technical questions, asked about multiple areas, requested contact info
- **medium**: 4-7 messages, some technical questions, showed interest but limited depth
- **low**: <4 messages, generic questions, brief conversation

**Recommended Next Steps Guidelines:**
- If high interest: Suggest technical interview, portfolio review, specific project discussions
- If medium interest: Suggest follow-up call, share additional materials
- If low interest: Suggest staying in touch, sharing updates

Respond ONLY with valid JSON, no additional text."""

        return prompt
    
    async def generate_analysis(
        self,
        chat_history: List[ChatMessage],
        recruiter_info: RecruiterInfo,
        model: str = "groq"  # Use Groq for cost efficiency
    ) -> SessionAnalysis:
        """
        Generate analysis for a recruiter session.
        
        Args:
            chat_history: List of chat messages
            recruiter_info: Recruiter information
            model: LLM model to use (default: groq for cost efficiency)
            
        Returns:
            SessionAnalysis object
        """
        try:
            # Build prompt
            prompt = self._build_analysis_prompt(chat_history, recruiter_info)
            
            # Prepare messages for LLM
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert HR analyst. Analyze recruiter conversations and provide structured insights in JSON format."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # Call LLM
            logger.info(f"Generating analysis using {model}")
            response = await self.ai_manager.get_response(model, messages)
            
            # Parse JSON response
            analysis_dict = self._parse_json_response(response)
            
            # Create SessionAnalysis object
            analysis = SessionAnalysis(**analysis_dict)
            
            logger.info(
                f"Analysis generated successfully "
                f"(interest: {analysis.interest_level}, "
                f"topics: {len(analysis.key_topics_discussed)})"
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to generate analysis: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(chat_history, recruiter_info)
    
    def _parse_json_response(self, response: str) -> dict:
        """
        Parse JSON from LLM response.
        
        Args:
            response: LLM response string
            
        Returns:
            Parsed JSON dict
        """
        try:
            # Try to find JSON in the response
            # Sometimes LLMs add extra text before/after JSON
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response[start_idx:end_idx]
                return json.loads(json_str)
            else:
                # Try parsing the whole response
                return json.loads(response)
                
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.debug(f"Response was: {response[:500]}")
            raise
    
    def _create_fallback_analysis(
        self,
        chat_history: List[ChatMessage],
        recruiter_info: RecruiterInfo
    ) -> SessionAnalysis:
        """
        Create a basic fallback analysis when LLM fails.
        
        Args:
            chat_history: List of chat messages
            recruiter_info: Recruiter information
            
        Returns:
            Basic SessionAnalysis object
        """
        logger.warning("Creating fallback analysis")
        
        # Count messages
        message_count = len(chat_history)
        
        # Infer interest level based on message count
        if message_count >= 8:
            interest_level = "high"
        elif message_count >= 4:
            interest_level = "medium"
        else:
            interest_level = "low"
        
        # Extract topics from messages (simple keyword extraction)
        topics = self._extract_topics_simple(chat_history)
        
        # Create summary
        summary = (
            f"Recruiter from {recruiter_info.company} "
            f"({recruiter_info.name}) had a conversation about the candidate's profile. "
        )
        
        if recruiter_info.role:
            summary += f"Discussed fit for {recruiter_info.role} role. "
        
        summary += f"Exchanged {message_count} messages covering {len(topics)} topics."
        
        # Role fit analysis
        role_fit = None
        if recruiter_info.role:
            role_fit = (
                f"Based on the conversation, the candidate appears to have relevant "
                f"experience for the {recruiter_info.role} position. "
                f"Further technical evaluation recommended."
            )
        
        # Recommended next steps based on interest level
        if interest_level == "high":
            next_steps = [
                "Schedule technical interview to discuss specific projects",
                "Review candidate's GitHub portfolio",
                "Discuss compensation and availability",
                "Arrange team introduction call"
            ]
        elif interest_level == "medium":
            next_steps = [
                "Follow up with more specific role requirements",
                "Share additional candidate materials",
                "Schedule brief introductory call"
            ]
        else:
            next_steps = [
                "Keep candidate profile for future opportunities",
                "Share role updates if requirements change"
            ]
        
        return SessionAnalysis(
            conversation_summary=summary,
            key_topics_discussed=topics[:10],  # Limit to 10
            role_fit_analysis=role_fit,
            interest_level=interest_level,
            recommended_next_steps=next_steps
        )
    
    def _extract_topics_simple(self, chat_history: List[ChatMessage]) -> List[str]:
        """
        Simple topic extraction from chat history.
        
        Args:
            chat_history: List of chat messages
            
        Returns:
            List of extracted topics
        """
        # Common technical keywords to look for
        keywords = [
            "langchain", "langgraph", "rag", "llm", "ai", "ml", "machine learning",
            "docker", "kubernetes", "devops", "mlops", "python", "fastapi",
            "react", "typescript", "aws", "gcp", "azure", "cloud",
            "experience", "projects", "skills", "education", "certifications"
        ]
        
        topics = set()
        
        # Combine all messages
        all_text = " ".join([msg.content.lower() for msg in chat_history])
        
        # Find keywords
        for keyword in keywords:
            if keyword in all_text:
                topics.add(keyword.title())
        
        # If no topics found, add generic ones
        if not topics:
            topics = {"General Discussion", "Profile Overview"}
        
        return list(topics)


# Global analysis generator instance
_analysis_generator: Optional[AnalysisGenerator] = None


def get_analysis_generator() -> AnalysisGenerator:
    """
    Get the global analysis generator instance.
    
    Returns:
        AnalysisGenerator instance
    """
    global _analysis_generator
    
    if _analysis_generator is None:
        _analysis_generator = AnalysisGenerator()
    
    return _analysis_generator
