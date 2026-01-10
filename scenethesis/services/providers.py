from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import google.genai as genai
from google.genai import types


@dataclass
class LLMConfig:
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    top_p: float = 0.95


class LLMProvider:
    def generate_scene_plan(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(self, api_key: Optional[str] = None, config: Optional[LLMConfig] = None) -> None:
        self._ensure_env_loaded()
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise EnvironmentError("未检测到 GEMINI_API_KEY，请在 .env 中配置 GEMINI_API_KEY。")
        self.config = config or LLMConfig()
        self.client = genai.Client(api_key=self.api_key)

    def generate_scene_plan(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        return self.generate_json(system_prompt, user_prompt)

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        prompt = "\n".join(
            [segment.strip() for segment in [system_prompt, user_prompt] if segment and segment.strip()]
        )
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                top_p=self.config.top_p,
                response_mime_type="application/json",
            ),
        )
        return self._extract_json(response)

    def chat(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        contents = [
            {
                "role": message.get("role", "user"),
                "parts": [message.get("content", "")],
            }
            for message in messages
        ]
        response = self.client.models.generate_content(
            model=self.config.model,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                top_p=self.config.top_p,
            ),
        )
        text = getattr(response, "text", "") or self._fallback_text(response)
        return {"text": text}

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        text = getattr(response, "text", "") or GeminiProvider._fallback_text(response)
        if not text:
            raise ValueError("Gemini 响应不包含可解析文本。")
        text = GeminiProvider._clean_json_text(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败: {exc.msg}. 原始响应: {text}") from exc

    @staticmethod
    def _clean_json_text(text: str) -> str:
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1]
            if "```" in cleaned:
                cleaned = cleaned.rsplit("```", 1)[0]
        return cleaned.strip()

    @staticmethod
    def _fallback_text(response: Any) -> str:
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        content = candidates[0].content
        if not content or not getattr(content, "parts", None):
            return ""
        parts = content.parts
        for part in parts:
            text = getattr(part, "text", None)
            if text:
                return text
        return ""

    @staticmethod
    def _ensure_env_loaded(env_path: str = ".env") -> None:
        if os.getenv("_SCENETHESIS_ENV_LOADED"):
            return
        env_file = Path(env_path)
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                if key and key not in os.environ:
                    os.environ[key] = value.strip()
        os.environ["_SCENETHESIS_ENV_LOADED"] = "1"
