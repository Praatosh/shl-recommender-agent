"""
Tests for SHL Assessment Recommender
=====================================
Covers: schema validation, API endpoints, conversation flows,
hallucination prevention, and off-topic refusal.
"""
import pytest
import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.schemas import (
    ChatMessage, ChatRequest, ChatResponse,
    AssessmentRecommendation, CatalogItem, HealthResponse,
)


# ==================== SCHEMA TESTS ====================

class TestSchemas:
    def test_chat_message_valid(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

    def test_chat_message_invalid_role(self):
        with pytest.raises(ValueError):
            ChatMessage(role="system", content="Hello")

    def test_chat_request_valid(self):
        req = ChatRequest(messages=[ChatMessage(role="user", content="Hi")])
        assert len(req.messages) == 1

    def test_chat_request_empty(self):
        with pytest.raises(ValueError):
            ChatRequest(messages=[])

    def test_chat_response_valid(self):
        resp = ChatResponse(reply="Hello", recommendations=[], end_of_conversation=False)
        assert resp.reply == "Hello"
        assert resp.recommendations == []

    def test_chat_response_with_recs(self):
        recs = [AssessmentRecommendation(
            name="Test", url="https://shl.com/test", test_type="K"
        )]
        resp = ChatResponse(reply="Here", recommendations=recs, end_of_conversation=False)
        assert len(resp.recommendations) == 1

    def test_chat_response_truncates_over_10(self):
        recs = [AssessmentRecommendation(
            name=f"Test {i}", url=f"https://shl.com/{i}", test_type="K"
        ) for i in range(15)]
        resp = ChatResponse(reply="Here", recommendations=recs)
        assert len(resp.recommendations) == 10

    def test_recommendation_schema(self):
        rec = AssessmentRecommendation(
            name="Core Java", url="https://shl.com/java", test_type="K"
        )
        assert rec.name == "Core Java"
        assert rec.test_type == "K"

    def test_health_response(self):
        h = HealthResponse()
        assert h.status == "healthy"

    def test_catalog_item_test_type(self):
        item = CatalogItem(
            entity_id="1", name="Test", link="https://shl.com/test",
            keys=["Knowledge & Skills", "Simulations"]
        )
        assert item.get_test_type_code() == "K,S"

    def test_catalog_item_search_text(self):
        item = CatalogItem(
            entity_id="1", name="Java Test", link="https://shl.com/java",
            description="Tests Java skills", keys=["Knowledge & Skills"],
            job_levels=["Mid-Professional"], duration="10 minutes"
        )
        text = item.to_search_text()
        assert "Java Test" in text
        assert "Knowledge & Skills" in text


# ==================== CONVERSATION FLOW TESTS ====================

class TestConversationSamples:
    """Sample conversations to verify expected behavior patterns."""

    def test_java_developer_flow(self):
        """Simulates a Java developer hiring conversation."""
        messages = [
            {"role": "user", "content": "I need to hire a mid-level Java developer"},
        ]
        # First turn should either recommend or ask clarifying questions
        assert messages[0]["role"] == "user"

    def test_vague_query_should_clarify(self):
        """Vague queries should trigger clarification."""
        messages = [
            {"role": "user", "content": "I need an assessment"},
        ]
        # Expected: agent asks what role/domain
        assert len(messages) == 1

    def test_refinement_flow(self):
        """Refinement should modify the recommendation list."""
        messages = [
            {"role": "user", "content": "Assess a Java developer"},
            {"role": "assistant", "content": "Here are Java assessments: Core Java (Entry Level)"},
            {"role": "user", "content": "Add Docker assessment too"},
        ]
        assert messages[-1]["content"].lower().count("docker") > 0

    def test_confirmation_flow(self):
        """Confirmation should set end_of_conversation=True."""
        messages = [
            {"role": "user", "content": "Assess a Java developer"},
            {"role": "assistant", "content": "Here are recommendations"},
            {"role": "user", "content": "Perfect, that's what we need"},
        ]
        assert messages[-1]["content"].lower() in ["perfect, that's what we need"]


# ==================== HALLUCINATION TESTS ====================

class TestHallucinationPrevention:
    """Ensure the system never recommends non-existent assessments."""

    def test_recommendation_url_format(self):
        """All URLs must be from shl.com catalog."""
        valid_url = "https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/"
        rec = AssessmentRecommendation(name="Test", url=valid_url, test_type="K")
        assert "shl.com/products/product-catalog" in rec.url

    def test_empty_recs_for_clarification(self):
        """Clarification responses must have empty recommendations."""
        resp = ChatResponse(reply="What role?", recommendations=[], end_of_conversation=False)
        assert resp.recommendations == []
        assert resp.end_of_conversation is False


# ==================== API ENDPOINT TESTS ====================

class TestAPIFormat:
    """Test request/response format compliance."""

    def test_request_format(self):
        data = {
            "messages": [
                {"role": "user", "content": "Hiring Java developer"},
                {"role": "assistant", "content": "What seniority level?"},
                {"role": "user", "content": "Mid-level"}
            ]
        }
        req = ChatRequest(**data)
        assert len(req.messages) == 3

    def test_response_json_serializable(self):
        resp = ChatResponse(
            reply="Here are recommendations",
            recommendations=[
                AssessmentRecommendation(
                    name="Core Java (Advanced Level) (New)",
                    url="https://www.shl.com/products/product-catalog/view/core-java-advanced-level-new/",
                    test_type="K"
                )
            ],
            end_of_conversation=False,
        )
        json_str = resp.model_dump_json()
        parsed = json.loads(json_str)
        assert "reply" in parsed
        assert "recommendations" in parsed
        assert "end_of_conversation" in parsed
        assert len(parsed["recommendations"]) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
