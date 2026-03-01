"""
LLM Client — Supports OpenRouter, DeepSeek, and AWS Bedrock

Set LLM_PROVIDER in .env:
  - "deepseek"   → DeepSeek native API (cheap, recommended)
  - "openrouter" → OpenRouter (multi-model)
  - "bedrock"    → AWS Bedrock (Claude, Llama, Titan)
"""

import json
import logging
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .config import LLMConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


class LLMClient:
    """Unified LLM client supporting OpenRouter and AWS Bedrock."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client.
        
        Auto-detects provider from LLM_PROVIDER env var.
        """
        self.config = config or DEFAULT_CONFIG.llm
        
        import os
        self.provider = os.getenv("LLM_PROVIDER", "deepseek").lower().strip()
        
        if self.provider == "bedrock":
            self._init_bedrock()
        elif self.provider == "deepseek":
            self._init_deepseek()
        else:
            self._init_openrouter()
    
    # ── OpenRouter Setup ───────────────────────────────────────────
    
    def _init_openrouter(self):
        """Initialize OpenRouter client."""
        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/finmem",
            "X-Title": "FinMEM Trading Agent"
        }
        logger.info(f"LLM Provider: OpenRouter | Model: {self.config.model}")
    
    # ── DeepSeek Setup ─────────────────────────────────────────────
    
    def _init_deepseek(self):
        """Initialize DeepSeek native API client."""
        import os
        self.deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "")
        
        if not self.deepseek_api_key:
            raise ValueError(
                "DeepSeek API key not found. "
                "Set DEEPSEEK_API_KEY in your .env file. "
                "Get one at https://platform.deepseek.com/api_keys"
            )
        
        self.deepseek_base_url = "https://api.deepseek.com/v1"
        self.deepseek_model = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
        
        self.headers = {
            "Authorization": f"Bearer {self.deepseek_api_key}",
            "Content-Type": "application/json",
        }
        logger.info(f"LLM Provider: DeepSeek | Model: {self.deepseek_model}")
    
    # ── AWS Bedrock Setup ──────────────────────────────────────────
    
    def _init_bedrock(self):
        """Initialize AWS Bedrock client."""
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for AWS Bedrock. Install it: pip install boto3"
            )
        
        import os
        region = os.getenv("AWS_REGION", "us-east-1")
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        
        # Build boto3 client — uses explicit keys if provided, else default credential chain
        client_kwargs = {
            "service_name": "bedrock-runtime",
            "region_name": region,
        }
        
        if access_key and secret_key:
            client_kwargs["aws_access_key_id"] = access_key
            client_kwargs["aws_secret_access_key"] = secret_key
            logger.info(f"LLM Provider: AWS Bedrock | Region: {region} | Auth: Access Key")
        else:
            logger.info(f"LLM Provider: AWS Bedrock | Region: {region} | Auth: Default chain (aws configure)")
        
        self.bedrock_client = boto3.client(**client_kwargs)
        
        # Default to DeepSeek R1 on Bedrock
        self.bedrock_model = os.getenv(
            "BEDROCK_MODEL_ID",
            "us.deepseek.r1-v1:0"
        )
        logger.info(f"Bedrock Model: {self.bedrock_model}")
    
    # ── Chat Method (Unified) ──────────────────────────────────────
    
    def chat(
        self,
        messages: List[ChatMessage] | str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a chat request to the LLM.
        
        Routes to the correct provider automatically.
        """
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        
        if self.provider == "bedrock":
            return self._chat_bedrock(messages, temperature, max_tokens)
        elif self.provider == "deepseek":
            return self._chat_deepseek(messages, temperature, max_tokens)
        else:
            return self._chat_openrouter(messages, model, temperature, max_tokens)
    
    # ── OpenRouter Chat ────────────────────────────────────────────
    
    def _chat_openrouter(
        self,
        messages: List[ChatMessage],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat to OpenRouter API."""
        payload = {
            "model": model or self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens
        }
        
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(
                f"OpenRouter API error: {response.status_code} - {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    # ── DeepSeek Chat ──────────────────────────────────────────────
    
    def _chat_deepseek(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat to DeepSeek native API (OpenAI-compatible)."""
        payload = {
            "model": self.deepseek_model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens
        }
        
        response = requests.post(
            f"{self.deepseek_base_url}/chat/completions",
            headers=self.headers,
            json=payload,
            timeout=60
        )
        
        if response.status_code != 200:
            raise Exception(
                f"DeepSeek API error: {response.status_code} - {response.text}"
            )
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    # ── AWS Bedrock Chat (Converse API — works with ALL models) ───
    
    def _chat_bedrock(
        self,
        messages: List[ChatMessage],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send chat via AWS Bedrock Converse API.
        
        Works with ALL Bedrock models: DeepSeek R1, Claude, Llama, Titan, etc.
        """
        
        # Separate system messages from conversation messages
        system_parts = []
        conversation = []
        
        for m in messages:
            if m.role == "system":
                system_parts.append({"text": m.content})
            else:
                conversation.append({
                    "role": m.role,
                    "content": [{"text": m.content}]
                })
        
        # Ensure at least one user message
        if not conversation:
            conversation = [{"role": "user", "content": [{"text": "Hello"}]}]
        
        # Build Converse API request
        converse_kwargs = {
            "modelId": self.bedrock_model,
            "messages": conversation,
            "inferenceConfig": {
                "maxTokens": min(max_tokens or self.config.max_tokens, 8192),
                "temperature": temperature or self.config.temperature,
            },
        }
        
        # Add system prompt if present
        if system_parts:
            converse_kwargs["system"] = system_parts
        
        try:
            response = self.bedrock_client.converse(**converse_kwargs)
            
            # Extract text from Converse API response
            output_message = response.get("output", {}).get("message", {})
            content_blocks = output_message.get("content", [])
            
            # Collect all text blocks (DeepSeek R1 may have reasoning + answer)
            text_parts = []
            for block in content_blocks:
                if "text" in block:
                    text_parts.append(block["text"])
            
            if text_parts:
                # Return the last text block (final answer, not reasoning)
                return text_parts[-1]
            
            raise Exception(f"Unexpected Bedrock Converse response: {response}")
            
        except Exception as e:
            error_str = str(e)
            if "ThrottlingException" in error_str:
                raise Exception(f"AWS Bedrock rate limited. Wait and retry: {e}")
            elif "AccessDeniedException" in error_str:
                raise Exception(
                    f"AWS Bedrock access denied for model {self.bedrock_model}. "
                    f"Enable it in AWS Console → Bedrock → Model access: {e}"
                )
            elif "ValidationException" in error_str:
                raise Exception(f"AWS Bedrock validation error: {e}")
            else:
                raise Exception(f"Bedrock Converse API failed: {e}")
    
    # ── Convenience Methods ────────────────────────────────────────
    
    def chat_with_system(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """Convenience method for system + user message chat."""
        messages = [
            ChatMessage(role="system", content=system_prompt),
            ChatMessage(role="user", content=user_message)
        ]
        return self.chat(messages, **kwargs)
    
    def generate_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate a JSON response from the LLM."""
        json_system = (system_prompt or "") + """

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, just pure JSON."""
        
        response = self.chat_with_system(json_system, prompt)
        
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")
    
    def test_connection(self) -> bool:
        """Test the API connection."""
        try:
            response = self.chat("Say 'Hello' in one word.")
            logger.info(f"Connection test passed: {response[:50]}")
            return bool(response)
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
