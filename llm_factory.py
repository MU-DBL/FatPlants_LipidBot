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
        
        else:
            raise ValueError(f"Unsupported provider: {provider}")