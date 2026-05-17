"""
SHL Assessment Recommender - Pydantic Schemas
==============================================
Strict schema definitions for API request/response.
Every field is typed and validated. This is the contract between client and server.

Design choice: Using Pydantic v2 with strict validation ensures
the LLM output is always schema-compliant before it reaches the client.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator

from app.utils import derive_test_type_code


class ChatMessage(BaseModel):
    """A single message in the conversation history."""
    role: str = Field(..., description="Message role: 'user' or 'assistant'")
    content: str = Field(..., description="Message content text")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{v}'")
        return v


class ChatRequest(BaseModel):
    """
    POST /chat request body.
    Contains full conversation history (stateless API design).
    """
    messages: List[ChatMessage] = Field(
        ...,
        min_length=1,
        description="Full conversation history, oldest first"
    )


class AssessmentRecommendation(BaseModel):
    """A single assessment recommendation with catalog-grounded data."""
    name: str = Field(..., description="Assessment name from SHL catalog")
    url: str = Field(..., description="Catalog URL for the assessment")
    test_type: str = Field(..., description="Test type code (e.g., K, P, A, S, B, C)")


class ChatResponse(BaseModel):
    """
    POST /chat response body.
    
    Rules:
    - recommendations is [] when clarifying or refusing
    - recommendations has 1-10 items when recommending
    - end_of_conversation is True only after final shortlist confirmation
    """
    reply: str = Field(..., description="Agent's response text")
    recommendations: List[AssessmentRecommendation] = Field(
        default_factory=list,
        description="Assessment recommendations (empty if clarifying)"
    )
    end_of_conversation: bool = Field(
        default=False,
        description="True only after final shortlist is confirmed"
    )

    @field_validator("recommendations")
    @classmethod
    def validate_recommendations_length(cls, v: list) -> list:
        if len(v) > 10:
            return v[:10]  # Defensive: truncate to max 10
        return v


class HealthResponse(BaseModel):
    """GET /health response."""
    status: str = "healthy"
    version: str = "1.0.0"


# --- Internal Schemas (not exposed via API) ---

class CatalogItem(BaseModel):
    """Schema for a single SHL catalog assessment."""
    entity_id: str
    name: str
    link: str
    description: str = ""
    job_levels: List[str] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)
    duration: str = ""
    remote: str = "yes"
    adaptive: str = "no"
    keys: List[str] = Field(default_factory=list)

    def get_test_type_code(self) -> str:
        """
        Derive test type code from assessment keys.
        Delegates to the shared utility function for consistency.
        
        Mapping:
        - Knowledge & Skills -> K
        - Personality & Behavior -> P
        - Ability & Aptitude -> A
        - Simulations -> S
        - Biodata & Situational Judgment -> B
        - Competencies -> C
        - Assessment Exercises -> E
        - Development & 360 -> D
        """
        return derive_test_type_code(self.keys)

    def to_search_text(self) -> str:
        """
        Create a rich text representation for embedding.
        Combines name, description, keys, job levels, and duration
        into a single searchable string.
        
        Why: Dense retrieval works better when we embed a rich
        text that captures multiple facets of the assessment.
        """
        parts = [
            f"Assessment: {self.name}",
            f"Description: {self.description}",
            f"Categories: {', '.join(self.keys)}",
            f"Job Levels: {', '.join(self.job_levels)}",
            f"Duration: {self.duration}",
            f"Remote: {self.remote}",
            f"Adaptive: {self.adaptive}",
            f"Languages: {', '.join(self.languages[:5])}",
        ]
        return " | ".join(parts)
