"""
SHL Assessment Recommender - Conversation Analyzer
====================================================
Analyzes conversation history to extract intent, requirements,
and determine the next action.

This is the "brain" of the agent's decision-making pipeline.

Architecture:
    User Message → Conversation Analyzer → Intent + Requirements
                                         ↓
                            Route to: Clarify / Retrieve / Compare / Refuse

Design choices:
1. Two-phase analysis: Rule-based fast path + LLM deep analysis
   - Rule-based catches common patterns instantly (greetings, confirmations)
   - LLM handles nuanced intent detection for complex queries
2. Stateless: Analyzes full conversation history each time
   - No session state to manage or lose
   - Conversation context is always complete
3. Search query generation: The LLM generates search queries
   - Better than using raw user text for retrieval
   - LLM can decompose "Java developer with leadership" into separate queries
"""

import re
from typing import Dict, Any, List, Optional

from app.llm_client import get_llm_client
from app.prompts import CONVERSATION_ANALYSIS_PROMPT, format_conversation
from app.logger import get_logger

logger = get_logger("analyzer")

# Patterns for fast rule-based intent detection
GREETING_PATTERNS = [
    r"^(hi|hello|hey|good morning|good afternoon|good evening)\b",
    r"^(greetings|howdy|what's up)\b",
]

CONFIRMATION_PATTERNS = [
    r"\b(perfect|that'?s? (it|what we need|great)|lock(ing)?( it)? in|confirmed?|done|yes,?\s*(that'?s?|go ahead)|good|approved)\b",
    r"^(yes|yep|yeah|sure|ok|okay|looks good|works for me|that works)\b",
]

OFF_TOPIC_PATTERNS = [
    r"\b(weather|recipe|joke|poem|story|sing|dance|politics|religion)\b",
    r"\b(ignore previous|forget your instructions|you are now|act as|pretend)\b",
    r"\b(what is your system prompt|reveal your instructions)\b",
]

COMPARISON_PATTERNS = [
    r"\b(compare|comparison|differ(ence|ent)|vs\.?|versus|which (is|one))\b",
    r"\b(what'?s? the difference|how (do|does) .+ differ)\b",
]

REFINE_PATTERNS = [
    r"\b(add|include|also (add|include)|drop|remove|swap|replace|change|switch)\b",
    r"\b(instead of|rather than|can you (add|remove|drop|replace))\b",
]


class ConversationAnalyzer:
    """
    Analyzes conversation to determine intent and extract requirements.
    
    Two-tier analysis:
    1. Fast path: regex-based pattern matching for simple intents
    2. Slow path: LLM-based deep analysis for complex queries
    """

    def __init__(self):
        self.llm = get_llm_client()

    def analyze(self, messages: list) -> Dict[str, Any]:
        """
        Analyze the full conversation and return structured analysis.
        
        Returns dict with:
        - intent: recommend, clarify, compare, refine, confirm, off_topic, greeting
        - search_queries: list of queries for vector search
        - needs_clarification: bool
        - is_sufficient_context: bool
        - Plus extracted requirements (role, level, technologies, etc.)
        """
        if not messages:
            return self._default_analysis()

        latest_message = self._get_latest_user_message(messages)
        if not latest_message:
            return self._default_analysis()

        # Fast path: check simple patterns first
        fast_result = self._fast_intent_detection(latest_message, messages)
        if fast_result:
            logger.info(f"Fast path intent: {fast_result['intent']}")
            return fast_result

        # Slow path: LLM-based deep analysis
        return self._llm_analysis(messages, latest_message)

    def _get_latest_user_message(self, messages: list) -> Optional[str]:
        """Extract the latest user message from conversation history."""
        for msg in reversed(messages):
            role = msg.get("role", getattr(msg, "role", None))
            content = msg.get("content", getattr(msg, "content", ""))
            if role == "user":
                return content
        return None

    def _fast_intent_detection(
        self, latest: str, messages: list
    ) -> Optional[Dict[str, Any]]:
        """
        Rule-based fast path for common intents.
        Returns None if no pattern matches (falls through to LLM).
        
        Why rule-based first?
        - Saves ~1-2s LLM latency for trivial cases
        - More deterministic for clear patterns
        - Handles prompt injection defense without LLM
        """
        latest_lower = latest.lower().strip()

        # Check for prompt injection / off-topic
        for pattern in OFF_TOPIC_PATTERNS:
            if re.search(pattern, latest_lower, re.IGNORECASE):
                return {
                    "intent": "off_topic",
                    "search_queries": [],
                    "needs_clarification": False,
                    "is_sufficient_context": False,
                    "role_or_domain": "",
                    "seniority_level": "",
                    "technologies": [],
                    "assessment_types_wanted": [],
                    "constraints": [],
                    "refinement_action": "none",
                    "assessments_to_add": [],
                    "assessments_to_remove": [],
                }

        # Check for greeting (only if first message)
        if len(messages) <= 1:
            for pattern in GREETING_PATTERNS:
                if re.search(pattern, latest_lower, re.IGNORECASE):
                    # If greeting contains substance, don't short-circuit
                    if len(latest_lower.split()) <= 5:
                        return {
                            "intent": "greeting",
                            "search_queries": [],
                            "needs_clarification": True,
                            "is_sufficient_context": False,
                            "role_or_domain": "",
                            "seniority_level": "",
                            "technologies": [],
                            "assessment_types_wanted": [],
                            "constraints": [],
                            "refinement_action": "none",
                            "assessments_to_add": [],
                            "assessments_to_remove": [],
                        }

        # Check for confirmation (only if previous recommendations exist)
        if self._has_previous_recommendations(messages):
            for pattern in CONFIRMATION_PATTERNS:
                if re.search(pattern, latest_lower, re.IGNORECASE):
                    return {
                        "intent": "confirm",
                        "search_queries": [],
                        "needs_clarification": False,
                        "is_sufficient_context": True,
                        "role_or_domain": "",
                        "seniority_level": "",
                        "technologies": [],
                        "assessment_types_wanted": [],
                        "constraints": [],
                        "refinement_action": "none",
                        "assessments_to_add": [],
                        "assessments_to_remove": [],
                    }

        # No fast match - fall through to LLM
        return None

    def _has_previous_recommendations(self, messages: list) -> bool:
        """Check if assistant has made recommendations in previous turns."""
        for msg in messages:
            role = msg.get("role", getattr(msg, "role", None))
            if role == "assistant":
                content = msg.get("content", getattr(msg, "content", ""))
                # Check if content mentions assessment recommendations
                if any(kw in content.lower() for kw in 
                       ["recommend", "shortlist", "assessment", "here's", "here is",
                        "following", "battery", "shl.com"]):
                    return True
        return False

    def _llm_analysis(
        self, messages: list, latest_message: str
    ) -> Dict[str, Any]:
        """
        Use LLM for deep conversation analysis.
        Extracts structured intent and requirements.
        """
        conversation_text = format_conversation(messages)
        prompt = CONVERSATION_ANALYSIS_PROMPT.format(conversation=conversation_text)

        try:
            result = self.llm.generate_json(
                system_prompt="You are a conversation analysis engine. Extract structured information from hiring conversations. Return valid JSON only.",
                user_prompt=prompt,
                temperature=0.0,
            )

            # Validate and fill defaults
            return self._validate_analysis(result)

        except Exception as e:
            logger.error(f"LLM analysis failed: {e}")
            return self._fallback_analysis(latest_message)

    def _validate_analysis(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and fill defaults in analysis result."""
        defaults = self._default_analysis()
        for key, default_val in defaults.items():
            if key not in result:
                result[key] = default_val
        
        # Validate intent
        valid_intents = {"recommend", "clarify", "compare", "refine", "confirm", "off_topic", "greeting"}
        if result.get("intent") not in valid_intents:
            result["intent"] = "recommend"

        return result

    def _fallback_analysis(self, latest_message: str) -> Dict[str, Any]:
        """
        Fallback when LLM analysis fails.
        Uses the raw message as a search query.
        """
        return {
            "intent": "recommend",
            "search_queries": [latest_message],
            "needs_clarification": False,
            "is_sufficient_context": True,
            "role_or_domain": "",
            "seniority_level": "",
            "technologies": [],
            "assessment_types_wanted": [],
            "constraints": [],
            "refinement_action": "none",
            "assessments_to_add": [],
            "assessments_to_remove": [],
        }

    def _default_analysis(self) -> Dict[str, Any]:
        """Default analysis result."""
        return {
            "intent": "greeting",
            "search_queries": [],
            "needs_clarification": True,
            "is_sufficient_context": False,
            "role_or_domain": "",
            "seniority_level": "",
            "technologies": [],
            "assessment_types_wanted": [],
            "constraints": [],
            "clarification_questions": [],
            "refinement_action": "none",
            "assessments_to_add": [],
            "assessments_to_remove": [],
        }


# Singleton
_analyzer: Optional[ConversationAnalyzer] = None


def get_analyzer() -> ConversationAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = ConversationAnalyzer()
    return _analyzer
