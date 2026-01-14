from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import google.genai as genai
from google.genai import types


@dataclass
class LLMConfig:
    model: str = "gemini-2.0-flash-exp"
    temperature: float = 0.2
    max_output_tokens: int = 2048
    top_p: float = 0.95
    use_vertex_ai: bool = False
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None


@dataclass
class ImageProviderConfig:
    model: str = "gemini-3.0-pro-image"
    image_size: Tuple[int, int] = (640, 640)
    image_format: str = "png"
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    fallback_models: Tuple[str, ...] = ()
    api_version: str | None = None
    use_vertex_ai: bool = False
    vertex_project: Optional[str] = None
    vertex_location: Optional[str] = None


class LLMProvider:
    def generate_scene_plan(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def generate_json(self, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
        raise NotImplementedError

    def chat(self, messages: list[dict[str, str]]) -> Dict[str, Any]:
        raise NotImplementedError


class ImageGenerationProvider:
    def generate_image(self, prompt: str, size: Tuple[int, int] | None = None) -> bytes:
        raise NotImplementedError


class GeminiProvider(LLMProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[LLMConfig] = None,
    ) -> None:
        self._ensure_env_loaded()
        self.config = config or LLMConfig()
        self.use_vertex_ai = bool(self.config.use_vertex_ai)
        self.vertex_project = self.config.vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.vertex_location = self.config.vertex_location or os.getenv("GOOGLE_CLOUD_LOCATION")
        if self.use_vertex_ai:
            if not self.vertex_project or not self.vertex_location:
                raise EnvironmentError("Vertex AI 需要配置 project_id 与 location。")
            self.client = genai.Client(
                vertexai=True,
                project=self.vertex_project,
                location=self.vertex_location,
            )
            self.api_key = None
        else:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise EnvironmentError("未检测到 GEMINI_API_KEY，请在 .env 中配置 GEMINI_API_KEY。")
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


class GeminiImageProvider(ImageGenerationProvider):
    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[ImageProviderConfig] = None,
    ) -> None:
        GeminiProvider._ensure_env_loaded()
        self.config = config or ImageProviderConfig()
        self.use_vertex_ai = bool(self.config.use_vertex_ai)
        self.vertex_project = self.config.vertex_project or os.getenv("GOOGLE_CLOUD_PROJECT")
        self.vertex_location = self.config.vertex_location or os.getenv("GOOGLE_CLOUD_LOCATION")
        if self.use_vertex_ai:
            if not self.vertex_project or not self.vertex_location:
                raise EnvironmentError("Vertex AI 需要配置 project_id 与 location。")
            self.client = genai.Client(
                vertexai=True,
                project=self.vertex_project,
                location=self.vertex_location,
            )
            self.api_key = None
        else:
            self.api_key = api_key or os.getenv("GEMINI_API_KEY")
            if not self.api_key:
                raise EnvironmentError("未检测到 GEMINI_API_KEY，请在 .env 中配置 GEMINI_API_KEY。")
            self.client = genai.Client(api_key=self.api_key)
        fallback = tuple(config.fallback_models) if config and config.fallback_models else ()
        self.models_to_try = (self.config.model,) + fallback

    def generate_image(self, prompt: str, size: Tuple[int, int] | None = None) -> bytes:
        if not prompt.strip():
            raise ValueError("图像生成 prompt 不能为空。")

        target_size = size or self.config.image_size
        aspect_ratio = self._aspect_ratio_from_size(target_size)
        size_hint = f"[Resolution Request]: {target_size[0]}x{target_size[1]}"
        last_exc: Exception | None = None
        for model_name in self.models_to_try:
            resolved = self._resolve_model_name(model_name)
            for http_opt in self._http_options_candidates():
                try:
                    config_kwargs: Dict[str, Any] = {
                        "aspect_ratio": aspect_ratio,
                        "number_of_images": 1,
                        "output_mime_type": f"image/{self.config.image_format}",
                        "http_options": http_opt,
                    }
                    if self.config.guidance_scale is not None:
                        config_kwargs["guidance_scale"] = self.config.guidance_scale
                    if self.config.negative_prompt:
                        config_kwargs["negative_prompt"] = self.config.negative_prompt
                    response = self.client.models.generate_images(
                        model=resolved,
                        prompt=f"{prompt}\n{size_hint}",
                        config=types.GenerateImagesConfig(**config_kwargs),
                    )
                    return self._extract_image_bytes(response)
                except Exception as exc:
                    last_exc = exc
                    version_info = f"(api_version={http_opt.api_version})" if http_opt else ""
                    print(f"⚠️ [GeminiImageProvider] 模型 {resolved} {version_info} 生成失败：{exc}")
                    continue
        if last_exc:
            raise last_exc
        raise RuntimeError("图像生成失败，但未捕获异常。")

    def _http_options_candidates(self) -> list[types.HttpOptions | None]:
        if not self.config.api_version:
            return [None]
        return [types.HttpOptions(api_version=self.config.api_version), None]

    @staticmethod
    def _resolve_model_name(name: str) -> str:
        aliases = {
            "gemini-3-pro-image": "gemini-3.0-pro-image",
            "gemini-3-pro-image-001": "gemini-3.0-pro-image",
            "gemini-3.0-pro-image-001": "gemini-3.0-pro-image",
            "gemini-2.5-flash-image": "gemini-2.5-flash-image",
            "gemini-2-5-flash-image": "gemini-2.5-flash-image",
            "gemini-2.5-flash-image-nano-banana": "gemini-2.5-flash-image",
        }
        normalized = name.strip()
        return aliases.get(normalized, normalized)

    @staticmethod
    def _aspect_ratio_from_size(size: Tuple[int, int]) -> str:
        width, height = size
        if height == 0:
            return "1:1"
        ratio = width / height
        candidates = {
            "1:1": 1.0,
            "4:3": 4 / 3,
            "3:4": 3 / 4,
            "16:9": 16 / 9,
            "9:16": 9 / 16,
        }
        closest = min(candidates.items(), key=lambda kv: abs(kv[1] - ratio))[0]
        return closest

    @staticmethod
    def _extract_image_bytes(response: types.GenerateImagesResponse) -> bytes:
        images = getattr(response, "generated_images", None) or []
        if not images:
            raise ValueError("Gemini 图像响应不包含 generated_images。")
        image_obj = images[0].image if images[0] else None
        if not image_obj or not image_obj.image_bytes:
            raise ValueError("Gemini 图像响应缺少 image_bytes。")
        return image_obj.image_bytes


def create_image_provider(
    provider_name: str,
    config: Optional[ImageProviderConfig] = None,
    **kwargs: Any,
) -> ImageGenerationProvider:
    normalized = provider_name.lower()
    if normalized == "gemini":
        return GeminiImageProvider(config=config, **kwargs)
    raise ValueError(f"暂不支持的图像 Provider: {provider_name}")
