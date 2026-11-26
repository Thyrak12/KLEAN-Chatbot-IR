"""Multi-provider LLM client for KLEAN Chatbot RAG.

Supports:
  - DeepSeek API (OpenAI-compatible)
  - OpenAI API (ChatGPT models)
  - Google Gemini API
  - Local Ollama (zero-cost offline models)
  - MockLLMClient (deterministic offline fallback for automated testing)
"""

from abc import ABC, abstractmethod
import json
import logging
import os
import time
from typing import Any, Dict, Iterator, List, Optional
import requests

from src.rag.prompt_templates import SYSTEM_PROMPT

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """Abstract base class for all LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        """Generate response text given a prompt and optional system prompt."""
        pass

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        """Stream response tokens or words given a prompt and optional system prompt."""
        text = self.generate(prompt, system_prompt, temperature, max_tokens)
        words = text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")


class DeepSeekLLMClient(BaseLLMClient):
    """Client for DeepSeek API (using OpenAI-compatible chat completions)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        api_base: str = "https://api.deepseek.com",
    ):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment or arguments.")
        self.model = model
        self.api_url = f"{api_base.rstrip('/')}/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        sys_p = system_prompt or SYSTEM_PROMPT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error("DeepSeek API call failed: %s", e)
            raise RuntimeError(f"DeepSeek generation error: {e}") from e

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        sys_p = system_prompt or SYSTEM_PROMPT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=30)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            logger.error("DeepSeek streaming failed: %s, falling back to full generate", e)
            text = self.generate(prompt, system_prompt, temperature, max_tokens)
            words = text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")


class OpenAILLMClient(BaseLLMClient):
    """Client for OpenAI API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
        api_base: str = "https://api.openai.com/v1",
    ):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY not found in environment or arguments.")
        self.model = model
        self.api_url = f"{api_base.rstrip('/')}/chat/completions"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        sys_p = system_prompt or SYSTEM_PROMPT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            raise RuntimeError(f"OpenAI generation error: {e}") from e

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        sys_p = system_prompt or SYSTEM_PROMPT
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, stream=True, timeout=30)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8") if isinstance(line, bytes) else line
                if line_str.startswith("data: "):
                    data_str = line_str[6:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk["choices"][0]["delta"].get("content", "")
                        if delta:
                            yield delta
                    except Exception:
                        continue
        except Exception as e:
            logger.error("OpenAI streaming failed: %s, falling back to full generate", e)
            text = self.generate(prompt, system_prompt, temperature, max_tokens)
            words = text.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")


class GeminiLLMClient(BaseLLMClient):
    """Client for Google Gemini REST API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-1.5-flash",
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment or arguments.")
        self.model = model
        self.api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent?key={self.api_key}"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        sys_p = system_prompt or SYSTEM_PROMPT
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [
                {"role": "user", "parts": [{"text": f"{sys_p}\n\n{prompt}"}]}
            ],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        try:
            resp = requests.post(self.api_url, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            return data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except Exception as e:
            logger.error("Gemini API call failed: %s", e)
            raise RuntimeError(f"Gemini generation error: {e}") from e


class OllamaLLMClient(BaseLLMClient):
    """Client for local Ollama instance."""

    def __init__(
        self,
        model: str = "llama3.2",
        api_base: str = "http://localhost:11434",
    ):
        self.model = model
        self.api_url = f"{api_base.rstrip('/')}/api/chat"

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        sys_p = system_prompt or SYSTEM_PROMPT
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": sys_p},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": temperature},
        }
        try:
            resp = requests.post(self.api_url, json=payload, timeout=60)
            resp.raise_for_status()
            data = resp.json()
            return data["message"]["content"].strip()
        except Exception as e:
            logger.error("Ollama API call failed: %s", e)
            raise RuntimeError(f"Ollama generation error: {e}") from e


class MockLLMClient(BaseLLMClient):
    """Deterministic offline mock LLM client for automated tests and CI environments."""

    def __init__(self, canned_response: Optional[str] = None):
        self.canned_response = canned_response

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> str:
        if self.canned_response:
            return self.canned_response

        # Parse context block from prompt
        if "No matching restaurants were found" in prompt:
            return (
                "I couldn't find any restaurants in Phnom Penh matching your request in our database. "
                "You might want to try searching with different keywords like cuisine type or neighborhood!"
            )

        # Extract lines that look like restaurant entries
        extracted_names = []
        for line in prompt.split("\n"):
            line_s = line.strip()
            if line_s.startswith("[") and "(ID:" in line_s:
                # e.g. [1] Trattoria Roma (ID: 10)
                parts = line_s.split("] ", 1)
                if len(parts) == 2:
                    name_part = parts[1].split(" (ID:")[0]
                    extracted_names.append(name_part)

        if not extracted_names:
            return "Here are the verified restaurant recommendations based on your query."

        items_str = "\n".join(f"- **{name}**" for name in extracted_names[:3])
        return (
            f"Based on our Phnom Penh restaurant database, here are the top recommendations for your request:\n\n"
            f"{items_str}\n\n"
            f"Enjoy your dining experience in Phnom Penh!"
        )

    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1000,
    ) -> Iterator[str]:
        full_text = self.generate(prompt, system_prompt, temperature, max_tokens)
        words = full_text.split(" ")
        for i, word in enumerate(words):
            yield word + (" " if i < len(words) - 1 else "")
            time.sleep(0.01)


def get_llm_client(
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    fallback_to_mock: bool = True,
) -> BaseLLMClient:
    """Factory function to instantiate the appropriate LLM client based on environment or parameters.

    Args:
        provider: 'deepseek', 'openai', 'gemini', 'ollama', or 'mock'. Defaults to env LLM_PROVIDER.
        model: Model name. Defaults to env LLM_MODEL.
        api_key: Optional API key override.
        fallback_to_mock: If True, falls back to MockLLMClient when credentials are missing.

    Returns:
        Configured BaseLLMClient instance.
    """
    prov = (provider or os.getenv("LLM_PROVIDER", "mock")).lower().strip()
    mdl = model or os.getenv("LLM_MODEL")

    logger.info("Initializing LLM client with provider='%s', model='%s'", prov, mdl)

    try:
        if prov == "deepseek":
            target_model = mdl or "deepseek-chat"
            return DeepSeekLLMClient(api_key=api_key, model=target_model)
        elif prov == "openai":
            target_model = mdl or "gpt-4o-mini"
            return OpenAILLMClient(api_key=api_key, model=target_model)
        elif prov in ("gemini", "google"):
            target_model = mdl or "gemini-1.5-flash"
            return GeminiLLMClient(api_key=api_key, model=target_model)
        elif prov == "ollama":
            target_model = mdl or "llama3.2"
            return OllamaLLMClient(model=target_model)
        elif prov == "mock":
            return MockLLMClient()
        else:
            logger.warning("Unknown LLM provider '%s'.", prov)
            if fallback_to_mock:
                return MockLLMClient()
            raise ValueError(f"Unsupported LLM provider: {prov}")
    except Exception as e:
        logger.warning("Failed to initialize '%s' client: %s.", prov, e)
        if fallback_to_mock:
            logger.info("Falling back to MockLLMClient.")
            return MockLLMClient()
        raise
