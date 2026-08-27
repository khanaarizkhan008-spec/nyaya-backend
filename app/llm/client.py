"""LLM client for the five-agent pipeline.

Talks to any OpenAI-compatible chat-completions endpoint:
  * Groq        (free cloud tier)  -> https://api.groq.com/openai/v1
  * HuggingFace (free cloud tier)  -> https://router.huggingface.co/v1
  * Ollama      (local)            -> http://localhost:11434/v1
  * demo mode   -> no network; callers fall back to deterministic agents

Every agent therefore has a deterministic fallback, so the product always
works end-to-end even without any key or connectivity.
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, TypeAdapter, ValidationError

from app.config import settings

logger = logging.getLogger("nyaya.llm")

T = TypeVar("T", bound=BaseModel)


class LLMUnavailable(Exception):
    """Raised when no live LLM provider is configured / reachable."""


@dataclass
class ProviderInfo:
    provider: str  # groq | huggingface | ollama | demo
    model: str
    base_url: str = ""
    api_key: str = ""
    live: bool = False


PROVIDER_DEFAULTS = {
    "groq": {
        "base_url": "https://api.groq.com/openai/v1",
        "model": "qwen/qwen3.6-27b",
    },
    "huggingface": {
        "base_url": "https://router.huggingface.co/v1",
        "model": "Qwen/Qwen2.5-7B-Instruct",
    },
    "ollama": {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen3:8b",
    },
}


def _strip_think_blocks(text: str) -> str:
    """Remove <think>...</think> reasoning blocks emitted by Qwen3-style models."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"^.*</think>", "", text, flags=re.DOTALL, count=1)
    return text.strip()


def extract_json_block(text: str) -> Any:
    """Extract the first balanced JSON object/array from an LLM response."""
    cleaned = _strip_think_blocks(text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip("`").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = cleaned.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = not in_str
                continue
            if in_str:
                continue
            if ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    candidate = cleaned[start : i + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    raise ValueError("No valid JSON found in LLM response")


class LLMClient:
    def __init__(self) -> None:
        self._provider: ProviderInfo | None = None
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------ resolution
    async def _ollama_alive(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                resp = await client.get(
                    settings.ollama_base_url.rstrip("/") + "/api/tags"
                )
                return resp.status_code == 200
        except Exception:
            return False

    async def resolve(self) -> ProviderInfo:
        """Resolve the active provider (cached)."""
        if self._provider is not None:
            return self._provider
        async with self._lock:
            if self._provider is not None:
                return self._provider
            choice = settings.llm_provider.lower()
            if choice == "auto":
                if settings.groq_api_key.strip():
                    choice = "groq"
                elif settings.hf_token.strip():
                    choice = "huggingface"
                elif await self._ollama_alive():
                    choice = "ollama"
                else:
                    choice = "demo"
            if choice == "demo":
                self._provider = ProviderInfo(provider="demo", model="deterministic")
                return self._provider

            defaults = PROVIDER_DEFAULTS.get(choice, {})
            base_url = settings.llm_base_url.strip() or defaults.get("base_url", "")
            model = settings.llm_model.strip() or defaults.get("model", "")
            api_key = ""
            if choice == "groq":
                api_key = settings.groq_api_key.strip()
            elif choice == "huggingface":
                api_key = settings.hf_token.strip()
            elif choice == "ollama":
                api_key = "ollama"

            if not api_key and choice != "ollama":
                logger.warning("LLM provider %s selected but no API key found", choice)
                self._provider = ProviderInfo(provider="demo", model="deterministic")
                return self._provider

            self._provider = ProviderInfo(
                provider=choice,
                model=model,
                base_url=base_url.rstrip("/"),
                api_key=api_key,
                live=True,
            )
            logger.info("LLM provider resolved: %s (%s)", choice, model)
            return self._provider

    async def provider_info(self) -> dict:
        info = await self.resolve()
        return {
            "provider": info.provider,
            "model": info.model,
            "mode": "live" if info.live else "demo",
        }

    # ------------------------------------------------------------ raw chat
    async def _post_chat(
        self,
        info: ProviderInfo,
        messages: list[dict],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        url = f"{info.base_url}/chat/completions"
        headers = {"Authorization": f"Bearer {info.api_key}"}
        payload: dict[str, Any] = {
            "model": info.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}

        async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
            try:
                resp = await client.post(url, headers=headers, json=payload)
            except httpx.HTTPError as exc:
                raise LLMUnavailable(f"LLM request failed: {exc}") from exc

        # Some providers reject response_format -> retry once without it
        if resp.status_code == 400 and json_mode:
            payload.pop("response_format", None)
            async with httpx.AsyncClient(timeout=settings.llm_timeout) as client:
                try:
                    resp = await client.post(url, headers=headers, json=payload)
                except httpx.HTTPError as exc:
                    raise LLMUnavailable(f"LLM request failed: {exc}") from exc

        if resp.status_code == 429:
            raise LLMUnavailable("LLM rate limited (429)")
        if resp.status_code >= 400:
            raise LLMUnavailable(
                f"LLM HTTP {resp.status_code}: {resp.text[:300]}"
            )
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMUnavailable(f"Unexpected LLM response shape: {str(data)[:200]}") from exc

    async def chat(
        self,
        system: str,
        user: str,
        *,
        temperature: float = 0.2,
        max_tokens: int = 3000,
        json_mode: bool = False,
    ) -> str:
        info = await self.resolve()
        if not info.live:
            raise LLMUnavailable("No live LLM provider configured (demo mode)")
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        return await self._post_chat(info, messages, temperature, max_tokens, json_mode)

    # ------------------------------------------------------------ JSON chat
    async def chat_json(
        self,
        system: str,
        user: str,
        schema: type[T],
        *,
        temperature: float = 0.2,
        max_tokens: int = 3500,
    ) -> T:
        """Ask the LLM for JSON matching a Pydantic schema; validate strictly."""
        info = await self.resolve()
        if not info.live:
            raise LLMUnavailable("No live LLM provider configured (demo mode)")

        json_schema = json.dumps(schema.model_json_schema(), indent=1)
        instruction = (
            f"{user}\n\n---\nOUTPUT REQUIREMENT: Respond with ONLY a valid JSON object "
            f"conforming exactly to this JSON schema (no markdown, no commentary, no "
            f"extra keys):\n{json_schema}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": instruction},
        ]

        last_error = "unknown"
        for attempt in range(2):
            raw = await self._post_chat(
                info, messages, temperature, max_tokens, json_mode=True
            )
            try:
                parsed = extract_json_block(raw)
                adapter = TypeAdapter(schema)
                return adapter.validate_python(parsed)
            except (ValueError, ValidationError) as exc:
                last_error = str(exc)[:500]
                logger.warning(
                    "LLM JSON validation failed (attempt %d): %s", attempt + 1, last_error
                )
                messages.append({"role": "assistant", "content": raw[:4000]})
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous output was invalid for this reason: "
                            f"{last_error}\nReturn ONLY the corrected JSON object "
                            "matching the schema. Do not include any other text."
                        ),
                    }
                )
        raise LLMUnavailable(f"LLM failed schema validation twice: {last_error}")


llm_client = LLMClient()
