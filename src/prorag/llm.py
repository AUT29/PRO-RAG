"""Provider-neutral OpenAI-compatible LLM client."""

from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

from openai import OpenAI

from .config import APISettings
from .usage import estimate_prompt_tokens, record_llm_call


class LLMClient:
    def __init__(
        self,
        settings: Optional[APISettings] = None,
        model: Optional[str] = None,
        temperature: float = 0.0,
    ):
        self.settings = settings or APISettings.from_env()
        self.model = model or self.settings.chat_model
        self.temperature = temperature
        self.client = OpenAI(
            api_key=self.settings.api_key,
            base_url=self.settings.base_url,
        )

    def complete(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 800,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
    ) -> str:
        started = time.perf_counter()
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=self.temperature if temperature is None else temperature,
                timeout=timeout,
            )
            content = (response.choices[0].message.content or "").strip()
            usage = getattr(response, "usage", None)
            prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
            if usage is None:
                prompt_tokens = estimate_prompt_tokens(messages)
                completion_tokens = max(1, len(content) // 4) if content else 0
            record_llm_call(
                prompt_tokens,
                completion_tokens,
                time.perf_counter() - started,
                estimated=usage is None,
            )
            return content
        except Exception:
            record_llm_call(
                estimate_prompt_tokens(messages),
                0,
                time.perf_counter() - started,
                estimated=True,
            )
            raise

    def complete_json(
        self,
        messages: List[Dict[str, str]],
        *,
        max_tokens: int = 1200,
    ) -> Dict:
        text = self.complete(messages, max_tokens=max_tokens)
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("The model response did not contain a JSON object")
        return json.loads(text[start : end + 1])

    def call_api(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 800,
        temperature: Optional[float] = None,
        timeout: float = 120.0,
        show_logs: bool = False,
        **_: object,
    ) -> str:
        """Compatibility wrapper for experiment runners."""
        return self.complete(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=timeout,
        )


LargeModelLLM = LLMClient
SmallModelLLM = LLMClient
