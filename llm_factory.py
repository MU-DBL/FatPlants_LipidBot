import json
import os
import re
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod
from data_service import LLMProvider
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import ollama
from ollama import AsyncClient
import requests
import json

class BaseLLM(ABC):
    """Abstract base class for LLM implementations"""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from prompt"""
        pass

    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the provider name"""
        pass

    def generate_stream(self, prompt: str):
        """Stream response tokens. Default: yield entire response at once."""
        yield self.generate(prompt)


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash", 
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 8192 
    ):
        """
        Initialize Gemini LLM
        """
        
        # API Key 
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Gemini API Key is missing! Please check config_sean.py")

        genai.configure(api_key=self.api_key)
        
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)
        
        self.generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_output_tokens=max_tokens,
        )

        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }
    
    def generate(self, prompt: str) -> str:
        """Generate response using Gemini"""
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini Generation Error: {e}")
            return ""

    def generate_stream(self, prompt: str):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=self.generation_config,
                safety_settings=self.safety_settings,
                stream=True,
            )
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            print(f"Gemini Streaming Error: {e}")
            yield self.generate(prompt)

    def get_provider_name(self) -> str:
        return "gemini"


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation"""
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        host: str = 'http://localhost:11434',
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        num_ctx: int = 4096,
        num_predict: int = 2048
    ):
        
        self.model_name = model_name
        self.host = host
        self.options = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "num_ctx": num_ctx,
            "num_predict": num_predict
        }
        
        if host:
            self.client = ollama.Client(host=host)
            self.async_client = AsyncClient(host=host)
        else:
            self.client = ollama.Client()
    
    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.options
        )
        return response['response'].strip()

    def generate_stream(self, prompt: str):
        for chunk in self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.options,
            stream=True,
        ):
            content = chunk.get('response') if isinstance(chunk, dict) else getattr(chunk, 'response', None)
            if content:
                yield content

    def get_provider_name(self) -> str:
        return "ollama"
    

class OpenRouterLLM(BaseLLM):
    def __init__(
        self,
        api_key: str,
        model: str = "openai/gpt-oss-20b",
        temperature: float = 0.5,
        enable_reasoning: bool = False,
        system_prompt: Optional[str] = None
    ):
        self.api_key = api_key
        self.model = model
        self.temperature = temperature
        self.enable_reasoning = enable_reasoning
        self.system_prompt = system_prompt
        self.base_url = "https://openrouter.ai/api/v1/chat/completions"

    def generate(self, prompt: str) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        messages = []

        if self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt
            })

        messages.append({
            "role": "user",
            "content": prompt
        })

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature
        }

        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}

        try:
            response = requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=60
            )

            response.raise_for_status()
            data = response.json()

            return data["choices"][0]["message"]["content"]

        except requests.RequestException as e:
            raise RuntimeError(f"OpenRouter API request failed: {e}")

        except KeyError:
            raise RuntimeError(f"Unexpected API response format: {data}")

    def generate_stream(self, prompt: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "stream": True,
        }
        if self.enable_reasoning:
            payload["reasoning"] = {"enabled": True}

        try:
            with requests.post(
                url=self.base_url,
                headers=headers,
                data=json.dumps(payload),
                stream=True,
                timeout=120,
            ) as response:
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line:
                        continue
                    decoded = line.decode("utf-8")
                    if not decoded.startswith("data: "):
                        continue
                    data_str = decoded[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                        content = chunk["choices"][0]["delta"].get("content", "")
                        if content:
                            yield content
                    except (json.JSONDecodeError, KeyError, IndexError):
                        pass
        except requests.RequestException as e:
            raise RuntimeError(f"OpenRouter streaming request failed: {e}")

    def get_provider_name(self) -> str:
        return "OpenRouter"

class LLMFactory:
    """Factory for creating LLM instances"""
    
    @staticmethod
    def create_llm(
        provider: Union[str, LLMProvider],
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        host: Optional[str] = None,
        temperature: float = 0.1,
        **kwargs
    ) -> BaseLLM:
        
        if isinstance(provider, str):
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                # LLMProvider
                pass
        
        if provider == LLMProvider.GEMINI:
            default_model = model_name or "gemini-1.5-flash"
            return GeminiLLM(
                api_key=api_key,
                model_name=default_model,
                temperature=temperature,
                **kwargs
            )
        
        elif provider == LLMProvider.OLLAMA:
            default_model = model_name or "llama3.1"
            return OllamaLLM(
                model_name=default_model,
                host=host,
                temperature=temperature,
                **kwargs
            )
        
        elif provider == LLMProvider.OPENROUTER:
            default_model = model_name
            return OpenRouterLLM(
                api_key=api_key,
                model = default_model,
                temperature=temperature,
                **kwargs
            )
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")
