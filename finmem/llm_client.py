"""
OpenRouter LLM Client

Handles all LLM interactions via the OpenRouter API.
"""

import json
import requests
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from .config import LLMConfig, DEFAULT_CONFIG


@dataclass
class ChatMessage:
    """Represents a chat message."""
    role: str  # "system", "user", or "assistant"
    content: str


class LLMClient:
    """Client for OpenRouter API interactions."""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        """Initialize the LLM client.
        
        Args:
            config: LLM configuration. Uses default if not provided.
        """
        self.config = config or DEFAULT_CONFIG.llm
        
        if not self.config.api_key:
            raise ValueError(
                "OpenRouter API key not found. "
                "Set OPENROUTER_API_KEY in your .env file."
            )
        
        self.headers = {
            "Authorization": f"Bearer {self.config.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/finmem",  # Required by OpenRouter
            "X-Title": "FinMEM Trading Agent"
        }
    
    def chat(
        self,
        messages: List[ChatMessage] | str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Send a chat request to the LLM.
        
        Args:
            messages: Either a string (user message) or list of ChatMessage objects.
            model: Override the default model.
            temperature: Override the default temperature.
            max_tokens: Override the default max tokens.
            
        Returns:
            The assistant's response text.
        """
        # Convert string to message list
        if isinstance(messages, str):
            messages = [ChatMessage(role="user", content=messages)]
        
        # Build request payload
        payload = {
            "model": model or self.config.model,
            "messages": [{"role": m.role, "content": m.content} for m in messages],
            "temperature": temperature or self.config.temperature,
            "max_tokens": max_tokens or self.config.max_tokens
        }
        
        # Make API request
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
    
    def chat_with_system(
        self,
        system_prompt: str,
        user_message: str,
        **kwargs
    ) -> str:
        """Convenience method for system + user message chat.
        
        Args:
            system_prompt: The system prompt to set context.
            user_message: The user's message.
            **kwargs: Additional arguments passed to chat().
            
        Returns:
            The assistant's response text.
        """
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
        """Generate a JSON response from the LLM.
        
        Args:
            prompt: The user prompt.
            system_prompt: Optional system prompt.
            
        Returns:
            Parsed JSON dictionary.
        """
        json_system = (system_prompt or "") + """

IMPORTANT: Respond ONLY with valid JSON. No markdown, no explanation, just pure JSON."""
        
        response = self.chat_with_system(json_system, prompt)
        
        # Try to extract JSON from response
        response = response.strip()
        if response.startswith("```"):
            # Remove markdown code blocks
            lines = response.split("\n")
            response = "\n".join(lines[1:-1])
        
        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON response: {e}\nResponse: {response}")
    
    def test_connection(self) -> bool:
        """Test the API connection.
        
        Returns:
            True if connection successful.
        """
        try:
            response = self.chat("Say 'Hello' in one word.")
            return bool(response)
        except Exception:
            return False
