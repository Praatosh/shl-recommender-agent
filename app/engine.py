"""
SHL Assessment Recommender - Recommendation Engine
====================================================
Core engine that orchestrates the full pipeline:
    Analyze → Retrieve → Generate → Validate

This is the main orchestrator that ties all components together.

Architecture:
    ConversationAnalyzer → Intent + Search Queries
                         ↓
    VectorStore.search() → Retrieved Catalog Items
                         ↓
    LLM.generate_json()  → Grounded Response
                         ↓
    Schema Validation    → ChatResponse

Design choices:
1. Single orchestrator class: Keeps the pipeline readable and debuggable
2. Intent-based routing: Different intents get different prompts
3. Grounded generation: LLM only sees retrieved catalog items
4. Post-generation validation: Ensures URLs exist in catalog
"""

import json
from typing import Dict, Any, List, Optional, Tuple

from app.analyzer import ConversationAnalyzer, get_analyzer
from app.embeddings import VectorStore, get_vector_store
from app.llm_client import LLMClient, get_llm_client
from app.schemas import ChatMessage, ChatResponse, AssessmentRecommendation
from app.prompts import (
    SYSTEM_PROMPT,
    CATALOG_CONTEXT_TEMPLATE,
    RECOMMENDATION_PROMPT,
    COMPARISON_PROMPT,
    REFUSAL_PROMPT,
    format_catalog_for_prompt,
    format_conversation,
)
from app.config import get_settings
from app.logger import get_logger

logger = get_logger("engine")


class RecommendationEngine:
    """
    Main orchestrator for the SHL Assessment Recommender.
    
    Pipeline:
    1. Analyze conversation → extract intent and requirements
    2. Route based on intent
    3. Retrieve relevant catalog items (if needed)
    4. Generate grounded LLM response
    5. Validate and enforce schema
    """

    def __init__(self):
        self.settings = get_settings()
        self.analyzer = get_analyzer()
        self.vector_store = get_vector_store()
        self.llm = get_llm_client()
        # Build a URL lookup for validation
        self._valid_urls: set = set()
        self._catalog_by_name: Dict[str, Dict] = {}
        self._init_catalog_lookup()

    def _init_catalog_lookup(self):
        """Build lookup tables for catalog validation."""
        for item in self.vector_store.catalog_items:
            url = item.get("link", "")
            name = item.get("name", "")
            if url:
                self._valid_urls.add(url)
            if name:
                self._catalog_by_name[name.lower()] = item

    def process_chat(self, messages: List[ChatMessage]) -> ChatResponse:
        """
        Main entry point: process a chat request and return a response.
        
        This is the full pipeline executed on every /chat request.
        """
        # Convert Pydantic models to dicts for internal processing
        msg_dicts = [{"role": m.role, "content": m.content} for m in messages]

        # Check conversation length limit
        user_turns = sum(1 for m in msg_dicts if m["role"] == "user")
        if user_turns > self.settings.max_conversation_turns:
            return ChatResponse(
                reply="We've reached the maximum conversation length. Here's a summary of our discussion. Feel free to start a new conversation if you need further help!",
                recommendations=[],
                end_of_conversation=True,
            )

        # Step 1: Analyze conversation
        logger.info(f"Analyzing conversation with {len(msg_dicts)} messages")
        analysis = self.analyzer.analyze(msg_dicts)
        intent = analysis.get("intent", "recommend")
        logger.info(f"Detected intent: {intent}")

        # Step 2: Route based on intent
        if intent == "off_topic":
            return self._handle_off_topic(msg_dicts)
        elif intent == "greeting":
            return self._handle_greeting(msg_dicts)
        elif intent == "confirm":
            return self._handle_confirmation(msg_dicts)
        elif intent == "compare":
            return self._handle_comparison(msg_dicts, analysis)
        elif intent == "refine":
            return self._handle_refinement(msg_dicts, analysis)
        else:
            # recommend or clarify - let the LLM decide based on context
            return self._handle_recommendation(msg_dicts, analysis)

    def _handle_off_topic(self, messages: list) -> ChatResponse:
        """Handle off-topic or unsafe requests."""
        latest = messages[-1]["content"] if messages else ""
        return ChatResponse(
            reply="I'm specialized in recommending SHL assessments for hiring and talent management. I'd be happy to help you find the right assessment — what role are you looking to fill?",
            recommendations=[],
            end_of_conversation=False,
        )

    def _handle_greeting(self, messages: list) -> ChatResponse:
        """Handle greeting messages."""
        return ChatResponse(
            reply="Hello! I'm the SHL Assessment Advisor. I can help you find the right SHL assessments for your hiring needs. What role or position are you looking to assess candidates for?",
            recommendations=[],
            end_of_conversation=False,
        )

    def _handle_confirmation(self, messages: list) -> ChatResponse:
        """
        Handle user confirming the recommendation list.
        Re-emit the last recommendations with end_of_conversation=True.
        """
        # Extract previous recommendations from assistant messages
        prev_recommendations = self._extract_previous_recommendations(messages)

        if prev_recommendations:
            return ChatResponse(
                reply="Great, your assessment shortlist is confirmed! You can proceed with setting up these assessments in your SHL platform. Feel free to start a new conversation if you need help with another role.",
                recommendations=prev_recommendations,
                end_of_conversation=True,
            )
        else:
            # No previous recommendations to confirm
            return ChatResponse(
                reply="I don't have a previous recommendation to confirm. What role would you like me to recommend assessments for?",
                recommendations=[],
                end_of_conversation=False,
            )

    def _handle_comparison(
        self, messages: list, analysis: Dict[str, Any]
    ) -> ChatResponse:
        """Handle assessment comparison requests."""
        # Retrieve items mentioned in the conversation
        search_queries = analysis.get("search_queries", [])
        if not search_queries:
            latest = messages[-1]["content"]
            search_queries = [latest]

        retrieved = self.vector_store.multi_query_search(search_queries, top_k=10)
        catalog_text = format_catalog_for_prompt(retrieved)
        conversation_text = format_conversation(messages)

        prompt = COMPARISON_PROMPT.format(
            conversation=conversation_text,
            catalog_items=catalog_text,
        )

        result = self.llm.generate_json(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
        )

        return self._build_response(result, retrieved)

    def _handle_refinement(
        self, messages: list, analysis: Dict[str, Any]
    ) -> ChatResponse:
        """
        Handle refinement requests (add/remove/swap assessments).
        Uses the full recommendation flow but with refinement context.
        """
        search_queries = analysis.get("search_queries", [])

        # Also search for specific assessments to add
        to_add = analysis.get("assessments_to_add", [])
        if to_add:
            search_queries.extend(to_add)

        if not search_queries:
            latest = messages[-1]["content"]
            search_queries = [latest]

        retrieved = self.vector_store.multi_query_search(search_queries, top_k=15)
        return self._generate_recommendation_response(messages, analysis, retrieved)

    def _handle_recommendation(
        self, messages: list, analysis: Dict[str, Any]
    ) -> ChatResponse:
        """
        Handle recommendation requests (or let LLM clarify if needed).
        
        Flow:
        1. Generate search queries from analysis
        2. Retrieve relevant catalog items
        3. Feed to LLM with conversation context
        4. Validate and return response
        """
        search_queries = analysis.get("search_queries", [])

        # Fallback: use latest user message as search query
        if not search_queries:
            for msg in reversed(messages):
                if msg["role"] == "user":
                    search_queries = [msg["content"]]
                    break

        # Retrieve from vector store
        if search_queries:
            retrieved = self.vector_store.multi_query_search(
                search_queries,
                top_k=self.settings.top_k
            )
        else:
            retrieved = []

        return self._generate_recommendation_response(messages, analysis, retrieved)

    def _generate_recommendation_response(
        self,
        messages: list,
        analysis: Dict[str, Any],
        retrieved: List[Tuple[Dict, float]],
    ) -> ChatResponse:
        """
        Generate a recommendation response using the LLM.
        Injects retrieved catalog items as grounding context.
        """
        conversation_text = format_conversation(messages)
        catalog_text = format_catalog_for_prompt(retrieved)
        latest_message = messages[-1]["content"] if messages else ""

        prompt = RECOMMENDATION_PROMPT.format(
            conversation=conversation_text,
            latest_message=latest_message,
            analysis=json.dumps(analysis, indent=2),
            catalog_items=catalog_text,
        )

        # Full system prompt with catalog context
        full_system = SYSTEM_PROMPT + "\n\n" + CATALOG_CONTEXT_TEMPLATE.format(
            catalog_items=catalog_text
        )

        result = self.llm.generate_json(
            system_prompt=full_system,
            user_prompt=prompt,
        )

        return self._build_response(result, retrieved)

    def _build_response(
        self,
        llm_result: Dict[str, Any],
        retrieved: List[Tuple[Dict, float]],
    ) -> ChatResponse:
        """
        Build a validated ChatResponse from LLM output.
        
        Critical validation:
        - Ensures all recommended URLs exist in the catalog
        - Truncates recommendations to max 10
        - Sets default values for missing fields
        """
        reply = llm_result.get("reply", "I can help you find SHL assessments. What role are you hiring for?")
        raw_recs = llm_result.get("recommendations", [])
        end = llm_result.get("end_of_conversation", False)

        # Validate and clean recommendations
        validated_recs = []
        for rec in raw_recs:
            if not isinstance(rec, dict):
                continue

            name = rec.get("name", "")
            url = rec.get("url", "")
            test_type = rec.get("test_type", "K")

            # Validate URL exists in catalog
            if url and url in self._valid_urls:
                validated_recs.append(
                    AssessmentRecommendation(
                        name=name,
                        url=url,
                        test_type=test_type,
                    )
                )
            elif name:
                # Try to find by name and fix the URL
                fixed = self._fix_recommendation(name, test_type)
                if fixed:
                    validated_recs.append(fixed)
                else:
                    logger.warning(f"Dropping recommendation with invalid URL: {name} -> {url}")

        # Truncate to max 10
        validated_recs = validated_recs[:self.settings.max_recommendations]

        return ChatResponse(
            reply=reply,
            recommendations=validated_recs,
            end_of_conversation=end,
        )

    def _fix_recommendation(
        self, name: str, test_type: str
    ) -> Optional[AssessmentRecommendation]:
        """
        Try to fix a recommendation by looking up the name in the catalog.
        This handles cases where the LLM gets the name right but URL wrong.
        """
        name_lower = name.lower().strip()

        # Exact match
        if name_lower in self._catalog_by_name:
            item = self._catalog_by_name[name_lower]
            # Derive test type from catalog if not provided
            type_map = {
                "Knowledge & Skills": "K",
                "Personality & Behavior": "P",
                "Ability & Aptitude": "A",
                "Simulations": "S",
                "Biodata & Situational Judgment": "B",
                "Competencies": "C",
                "Assessment Exercises": "E",
                "Development & 360": "D",
            }
            codes = [type_map[k] for k in item.get("keys", []) if k in type_map]
            catalog_type = ",".join(codes) if codes else test_type

            return AssessmentRecommendation(
                name=item["name"],
                url=item["link"],
                test_type=catalog_type,
            )

        # Fuzzy match: check if name is a substring
        for cat_name, item in self._catalog_by_name.items():
            if name_lower in cat_name or cat_name in name_lower:
                type_map = {
                    "Knowledge & Skills": "K",
                    "Personality & Behavior": "P",
                    "Ability & Aptitude": "A",
                    "Simulations": "S",
                    "Biodata & Situational Judgment": "B",
                    "Competencies": "C",
                    "Assessment Exercises": "E",
                    "Development & 360": "D",
                }
                codes = [type_map[k] for k in item.get("keys", []) if k in type_map]
                catalog_type = ",".join(codes) if codes else test_type

                return AssessmentRecommendation(
                    name=item["name"],
                    url=item["link"],
                    test_type=catalog_type,
                )

        return None

    def _extract_previous_recommendations(
        self, messages: list
    ) -> List[AssessmentRecommendation]:
        """
        Extract recommendations from previous assistant messages.
        Used when user confirms to re-emit the same list.
        
        Strategy: Search for SHL URLs in assistant messages
        and match them to catalog items.
        """
        import re
        recommendations = []
        seen_urls = set()

        for msg in reversed(messages):
            if msg["role"] == "assistant":
                content = msg["content"]
                # Find SHL catalog URLs
                urls = re.findall(
                    r'https://www\.shl\.com/products/product-catalog/view/[^\s\)\"\']+',
                    content
                )
                for url in urls:
                    url = url.rstrip("/")
                    if url not in seen_urls:
                        seen_urls.add(url)
                        # Find matching catalog item
                        for item in self.vector_store.catalog_items:
                            if item.get("link", "").rstrip("/") == url:
                                type_map = {
                                    "Knowledge & Skills": "K",
                                    "Personality & Behavior": "P",
                                    "Ability & Aptitude": "A",
                                    "Simulations": "S",
                                    "Biodata & Situational Judgment": "B",
                                    "Competencies": "C",
                                    "Assessment Exercises": "E",
                                    "Development & 360": "D",
                                }
                                codes = [type_map[k] for k in item.get("keys", []) if k in type_map]
                                test_type = ",".join(codes) if codes else "K"

                                recommendations.append(
                                    AssessmentRecommendation(
                                        name=item["name"],
                                        url=item["link"],
                                        test_type=test_type,
                                    )
                                )
                                break

                if recommendations:
                    break  # Use the most recent set

        return recommendations[:self.settings.max_recommendations]


# Singleton
_engine: Optional[RecommendationEngine] = None


def get_engine() -> RecommendationEngine:
    global _engine
    if _engine is None:
        _engine = RecommendationEngine()
    return _engine


def reset_engine():
    """Reset engine singleton (useful for testing)."""
    global _engine
    _engine = None
