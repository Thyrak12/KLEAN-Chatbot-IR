"""Prompt templates and system guardrails for KLEAN Chatbot RAG.

Enforces strict grounding, citation requirements, anti-hallucination rules,
and bilingual support (English & Khmer).
"""

from typing import Any, Dict, List, Optional

SYSTEM_PROMPT = """You are KLEAN, a friendly, knowledgeable, and reliable dining assistant specializing in restaurants, local eats, and cafes across Phnom Penh, Cambodia.

Your mission is to provide helpful, trustworthy, and well-structured restaurant recommendations strictly grounded in the verified restaurant database records provided to you in the CONTEXT.

CRITICAL GUARDRAILS & INSTRUCTIONS:
1. GROUNDED FACTS ONLY: Only recommend restaurants that appear in the provided CONTEXT. NEVER fabricate or hallucinate restaurants, dishes, street names, phone numbers, or ratings not present in the CONTEXT.
2. MISSING MATCHES: If the provided CONTEXT does not contain any suitable restaurants matching the user's specific request, politely explain that you don't have matching records in your Phnom Penh database and suggest the closest alternative from the CONTEXT if relevant.
3. CITATIONS & PRACTICAL DETAILS: Whenever you recommend a place, cite its Name, Category/Cuisine, Address, and Phone/Website if available in the context.
4. BILINGUAL RESPECT:
   - If the user writes in Khmer, answer politely and fluently in Khmer.
   - If the user writes in English, answer in English.
   - If the user mixes both, respond in English with authentic Khmer terms preserved.
5. TONE: Warm, inviting, and concise. Avoid rambling. Use clean Markdown bullet points.
"""

USER_PROMPT_TEMPLATE = """CONTEXT INFORMATION:
The following restaurants were retrieved from the Phnom Penh database based on relevance to the user's inquiry:

{context}

---
USER QUESTION:
{query}

Please provide a helpful, grounded response following your system instructions.
"""

NO_CONTEXT_PROMPT = """USER QUESTION:
{query}

NOTE: No matching restaurants were found in the Phnom Penh database. Please politely inform the user that no matching places were found and offer tips on how they can rephrase their search (e.g. searching by cuisine, popular district like BKK1 or Daun Penh, or specific dish).
"""


class PromptManager:
    """Manages prompt construction and formatting for RAG generation."""

    def __init__(self, system_prompt: str = SYSTEM_PROMPT):
        self.system_prompt = system_prompt

    def build_prompt(self, query: str, context: Optional[str] = None) -> str:
        """Construct user prompt combining retrieved context and user query.

        Args:
            query: The user's natural language question.
            context: Formatted string of retrieved restaurant details.

        Returns:
            Rendered prompt string ready for LLM input.
        """
        clean_query = query.strip()
        if not context or not context.strip():
            return NO_CONTEXT_PROMPT.format(query=clean_query)

        return USER_PROMPT_TEMPLATE.format(context=context.strip(), query=clean_query)
