import json
import os
import re
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

# config_sean에서 키 가져오기 시도 (없으면 None)
try:
    from config_sean import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = None

from data_service import LLMProvider

# Google Gemini imports
try:
    import google.generativeai as genai
    # [NEW] 안전 설정을 위한 타입 임포트
    from google.generativeai.types import HarmCategory, HarmBlockThreshold
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Ollama imports
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

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


class GeminiLLM(BaseLLM):
    """Google Gemini LLM implementation"""
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-1.5-flash", # [수정] 기본값 1.5로 변경 권장
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        max_tokens: int = 8192 # [수정] 넉넉하게 8192로 상향
    ):
        """
        Initialize Gemini LLM
        """
        if not GENAI_AVAILABLE:
            raise ImportError(
                "google-generativeai is not installed. "
                "Install it with: pip install google-generativeai"
            )
        
        # API Key 설정 (인자값 우선, 없으면 config_sean 값, 그것도 없으면 환경변수)
        self.api_key = api_key or GEMINI_API_KEY or os.getenv("GOOGLE_API_KEY")
        
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

        # [NEW] 안전 필터 해제 설정 (중요!)
        # 생물학 용어(kill, toxic 등)가 차단되지 않도록 모든 필터를 끕니다.
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
                safety_settings=self.safety_settings # [NEW] 안전 설정 적용
            )
            return response.text.strip()
        except Exception as e:
            print(f"Gemini Generation Error: {e}")
            return "" # 에러 발생 시 빈 문자열 반환 (프로그램 죽지 않게)
    
    def get_provider_name(self) -> str:
        return "gemini"


class OllamaLLM(BaseLLM):
    """Ollama local LLM implementation"""
    
    def __init__(
        self,
        model_name: str = "llama3.1",
        host: Optional[str] = None,
        temperature: float = 0.1,
        top_p: float = 0.95,
        top_k: int = 40,
        num_ctx: int = 4096,
        num_predict: int = 2048
    ):
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "ollama is not installed. "
                "Install it with: pip install ollama"
            )
        
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
        else:
            self.client = ollama.Client()
    
    def generate(self, prompt: str) -> str:
        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=self.options
        )
        return response['response'].strip()
    
    def get_provider_name(self) -> str:
        return "ollama"


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
            # 문자열로 들어온 경우 Enum으로 변환 시도
            try:
                provider = LLMProvider(provider.lower())
            except ValueError:
                # LLMProvider에 없는 문자열이면 예외 처리 (혹은 기본값)
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
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")