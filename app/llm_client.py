"""
SHL Assessment Recommender - LLM Client
========================================
Abstraction layer for LLM API calls.
Supports Gemini (primary) and Groq (fallback).

Design choices:
- Abstraction layer allows swapping LLM providers without changing business logic
- JSON parsing with fallback handles inconsistent LLM outputs
- Timeout handling prevents slow responses from blocking the API
- Temperature=0.1 for deterministic, factual responses (not creative writing)
"""

import json
import re
import os
from typing import Dict, Any, Optional

from app.config import get_settings
from app.logger import get_logger

logger = get_logger("llm")


class LLMClient:
    """
    Unified LLM client supporting multiple providers.
    
    Why this abstraction?
    - Swap between Gemini/Groq without changing calling code
    - Centralized error handling and retry logic
    - JSON output parsing with fallback strategies
    """

    def __init__(self):
        self.settings = get_settings()
        self._client = None
        self._provider = self.settings.llm_provider.lower()

    def _init_gemini(self):
        """Initialize Google Gemini client."""
        from google import genai
        self._client = genai.Client(api_key=self.settings.gemini_api_key)
        logger.info(f"Gemini client initialized with model: {self.settings.gemini_model}")

    def _init_groq(self):
        """Initialize Groq client via OpenAI-compatible API."""
        from openai import OpenAI
        self._client = OpenAI(
            api_key=self.settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1"
        )
        logger.info(f"Groq client initialized with model: {self.settings.groq_model}")

    def _ensure_client(self):
        """Lazy initialization of LLM client."""
        if self._client is None:
            if self._provider == "gemini":
                self._init_gemini()
            elif self._provider == "groq":
                self._init_groq()
            else:
                raise ValueError(f"Unknown LLM provider: {self._provider}")

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_tokens: int = 2048,
    ) -> str:
        """
        Generate text from the LLM.
        
        Args:
            system_prompt: System-level instructions
            user_prompt: User-level prompt with context
            temperature: Sampling temperature (low = deterministic)
            max_tokens: Maximum response length
        
        Returns:
            Raw text response from the LLM
        """
        self._ensure_client()

        try:
            if self._provider == "gemini":
                return self._generate_gemini(system_prompt, user_prompt, temperature, max_tokens)
            elif self._provider == "groq":
                return self._generate_groq(system_prompt, user_prompt, temperature, max_tokens)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

    def _generate_gemini(
        self, system_prompt: str, user_prompt: str, 
        temperature: float, max_tokens: int
    ) -> str:
        """Generate using Google Gemini API."""
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.settings.gemini_model,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                response_mime_type="application/json",
            ),
        )
        return response.text

    def _generate_groq(
        self, system_prompt: str, user_prompt: str,
        temperature: float, max_tokens: int
    ) -> str:
        """Generate using Groq API (OpenAI-compatible)."""
        response = self._client.chat.completions.create(
            model=self.settings.groq_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def generate_json(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Generate and parse JSON from LLM.
        
        Includes fallback parsing strategies because LLMs sometimes:
        - Wrap JSON in markdown code blocks
        - Add trailing text after JSON
        - Return slightly malformed JSON
        
        Returns parsed dict or fallback error dict.
        """
        raw = self.generate(system_prompt, user_prompt, temperature)
        return self._parse_json_response(raw)

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """
        Parse JSON from LLM response with multiple fallback strategies.
        
        Strategy order:
        1. Direct JSON parse
        2. Extract from markdown code blocks
        3. Find JSON object with regex
        4. Return error fallback
        """
        # Strategy 1: Direct parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Strategy 2: Extract from markdown code blocks
        code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', raw, re.DOTALL)
        if code_block_match:
            try:
                return json.loads(code_block_match.group(1).strip())
            except json.JSONDecodeError:
                pass

        # Strategy 3: Find JSON object with regex
        json_match = re.search(r'\{.*\}', raw, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass

        # Strategy 4: Fallback
        logger.warning(f"Failed to parse JSON from LLM response: {raw[:200]}")
        return {
            "reply": "I apologize, but I encountered an issue processing your request. Could you please rephrase your question about SHL assessments?",
            "recommendations": [],
            "end_of_conversation": False,
        }


# Singleton
_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    """Get or create the singleton LLM client."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
