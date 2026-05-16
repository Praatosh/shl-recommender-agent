"""
SHL Assessment Recommender - Configuration Management
=====================================================
Centralizes all configuration using pydantic-settings for validation.
Uses environment variables with .env file fallback.

Design choice: Single source of truth for all config values.
This prevents scattered magic numbers and makes deployment straightforward.
"""

import os
from functools import lru_cache
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # --- LLM Configuration ---
    llm_provider: str = Field(default="gemini", description="LLM provider: gemini or groq")
    gemini_api_key: str = Field(default="", description="Google Gemini API key")
    gemini_model: str = Field(default="gemini-2.0-flash", description="Gemini model name")
    groq_api_key: str = Field(default="", description="Groq API key")
    groq_model: str = Field(default="llama-3.1-70b-versatile", description="Groq model name")

    # --- Embedding Configuration ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Sentence-transformers model for embeddings"
    )

    # --- Data Paths ---
    faiss_index_path: str = Field(default="data/faiss_index", description="FAISS index directory")
    catalog_path: str = Field(default="data/catalog.json", description="Catalog JSON path")

    # --- Server Configuration ---
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")

    # --- Retrieval Configuration ---
    top_k: int = Field(default=15, description="Number of candidates to retrieve from FAISS")
    rerank_top_k: int = Field(default=10, description="Max recommendations after reranking")

    # --- Conversation Limits ---
    max_conversation_turns: int = Field(default=8, description="Max conversation turns")
    max_recommendations: int = Field(default=10, description="Max recommendations per response")

    # --- Logging ---
    log_level: str = Field(default="INFO", description="Logging level")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
    }


@lru_cache()
def get_settings() -> Settings:
    """
    Cached settings loader.
    Uses lru_cache so settings are loaded once and reused.
    This is important because loading .env and validating on every request wastes time.
    """
    return Settings()
