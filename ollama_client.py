"""
Ollama API client for LLM agent communication
"""
import requests
import json
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# Constants
DEFAULT_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "llama3.2"
DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 200
DEFAULT_REPEAT_PENALTY = 1.1
DEFAULT_REPEAT_LAST_N = 128
DEFAULT_MIN_P = 0.05
API_TIMEOUT = 60
CONNECTION_CHECK_TIMEOUT = 5


class OllamaClient:
    """Client for interacting with Ollama API"""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        repeat_penalty: float = DEFAULT_REPEAT_PENALTY,
        repeat_last_n: int = DEFAULT_REPEAT_LAST_N,
        min_p: float = DEFAULT_MIN_P,
        think: bool = True
    ):
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.repeat_penalty = repeat_penalty
        self.repeat_last_n = repeat_last_n
        self.min_p = min_p
        self.think = think  # False = thinking無効（qwen3等でJSONを直接出力させる）
        self.chat_url = f"{self.base_url}/api/chat"

    def generate(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None
    ) -> str:
        if temperature is None:
            temperature = self.temperature
        if max_tokens is None:
            max_tokens = self.max_tokens
        if timeout is None:
            timeout = API_TIMEOUT

        try:
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False,
                "think": self.think,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "repeat_penalty": self.repeat_penalty,
                    "repeat_last_n": self.repeat_last_n,
                    "min_p": self.min_p
                }
            }

            response = requests.post(
                self.chat_url,
                json=payload,
                timeout=timeout
            )
            response.raise_for_status()

            result = response.json()
            content = result.get("message", {}).get("content", "").strip()
            return content

        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return ""
        except Exception as e:
            logger.error(f"Unexpected error in Ollama client: {e}")
            return ""

    def check_connection(self) -> bool:
        """Check if Ollama server is accessible"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=CONNECTION_CHECK_TIMEOUT
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """List all available models in Ollama"""
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=CONNECTION_CHECK_TIMEOUT
            )
            response.raise_for_status()
            data = response.json()
            models = [model['name'] for model in data.get('models', [])]
            return models
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []

    def check_model_exists(self) -> bool:
        """Check if the specified model exists"""
        available_models = self.list_models()
        return self.model in available_models
