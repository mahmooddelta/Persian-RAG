"""
OpenAI-Compatible LLM Client implementing `LLMPort`.

Generates grounded Persian answers strictly derived from retrieved context chunks,
including explicit inline source citations (`[منبع: نام فایل، صفحه X]`) and
hallucination prevention guardrails.
"""

from __future__ import annotations

from typing import Sequence

import httpx

from src.domain.entities import RetrievalResult
from src.domain.exceptions import LLMGenerationError
from src.domain.interfaces import LLMPort

PERSIAN_RAG_SYSTEM_PROMPT = """شما یک دستیار هوشمند و دقیق پرسش‌وپاسخ مبتنی بر اسناد فارسی (سیستم RAG) هستید.
وظیفه شما پاسخ‌گویی علمی، دقیق، روان و کاملاً مستند به پرسش کاربر **صرفاً بر اساس قطعات متنی بازیابی‌شده (Context)** است.

قوانین الزامی:
۱. **پایبندی مطلق به متن (Strict Grounding):** فقط و فقط از اطلاعات موجود در «اسناد بازیابی‌شده» استفاده کنید. از دانش قبلی خود یا حدس و گمان استفاده نکنید.
۲. **ترکیب چندمنبعی (Multi-Source Synthesis):** اگر پرسش کاربر شامل چند بخش یا چند مفهوم است که در فایل‌های مختلف آمده‌اند (مثلاً تعریف یک مفهوم در یک سند و مفهوم دیگر در سند دیگر)، حتماً اطلاعات تمام اسناد مرتبط بازیابی‌شده را پوشش دهید و برای هر بخش منبع دقیق خودش را به صورت `[منبع: نام_فایل، صفحه X]` ذکر کنید.
۳. **عدم وجود اطلاعات کافی:** اگر پاسخ پرسش در قطعات بازیابی‌شده وجود ندارد، به صراحت اعلام کنید: «اطلاعات کافی برای پاسخ به این پرسش در اسناد موجود یافت نشد.» و از ساختن پاسخ خودداری کنید.
۴. **نگارش استاندارد فارسی:** پاسخ باید به زبان فارسی معیار، با رعایت نیم‌فاصله‌ها و ساختار منظم ارائه شود."""


class OpenAICompatibleLLMClient(LLMPort):
    """
    HTTP client for OpenAI-compatible `/v1/chat/completions` API.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        temperature: float = 0.1,
        max_tokens: int = 1024,
        timeout_seconds: float = 60.0,
        system_prompt: str = PERSIAN_RAG_SYSTEM_PROMPT,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._timeout_seconds = timeout_seconds
        self._system_prompt = system_prompt
        self._http_client = http_client

    @staticmethod
    def format_context_block(context_chunks: Sequence[RetrievalResult]) -> str:
        """
        Format retrieved chunks into a numbered, metadata-annotated Persian context prompt.
        """
        formatted_blocks: list[str] = []
        for idx, res in enumerate(context_chunks, start=1):
            chunk = res.chunk
            block = (
                f"--- قطعه شماره {idx} ---\n"
                f"نام فایل منبع: {chunk.source_filename}\n"
                f"شماره صفحه: {chunk.page_number}\n"
                f"امتیاز شباهت: {res.score:.4f}\n"
                f"متن مستند:\n{chunk.text.strip()}"
            )
            formatted_blocks.append(block)
        return "\n\n".join(formatted_blocks)

    def generate_grounded_answer(
        self,
        query: str,
        context_chunks: Sequence[RetrievalResult],
    ) -> str:
        """
        Send system instructions, formatted retrieved chunks, and user query to the
        OpenAI-compatible Chat Completions endpoint.
        """
        if not context_chunks:
            return "اطلاعات کافی برای پاسخ به این پرسش در اسناد بازیابی‌شده یافت نشد."

        context_str = self.format_context_block(context_chunks)
        user_prompt = (
            f"### اسناد بازیابی‌شده:\n{context_str}\n\n"
            f"### پرسش کاربر:\n{query}\n\n"
            "لطفاً بر اساس اسناد فوق، پاسخی دقیق و مستند همراه با ذکر نام فایل و شماره صفحه ارائه دهید:"
        )

        endpoint = f"{self._base_url}/chat/completions"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        payload = {
            "model": self._model,
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
            "messages": [
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }

        try:
            if self._http_client is not None:
                response = self._http_client.post(
                    endpoint,
                    json=payload,
                    headers=headers,
                    timeout=self._timeout_seconds,
                )
            else:
                with httpx.Client(timeout=self._timeout_seconds) as client:
                    response = client.post(
                        endpoint,
                        json=payload,
                        headers=headers,
                    )

            response.raise_for_status()
            data = response.json()
            choices = data.get("choices", [])
            if not choices:
                raise LLMGenerationError("LLM API returned no completion choices.")

            message_content = choices[0].get("message", {}).get("content", "")
            if not message_content or not message_content.strip():
                raise LLMGenerationError("LLM API returned an empty message content.")

            return message_content.strip()
        except Exception as exc:
            if isinstance(exc, LLMGenerationError):
                raise
            raise LLMGenerationError(
                f"Failed to generate answer from LLM endpoint '{endpoint}' with model '{self._model}'",
                details=str(exc),
            ) from exc

