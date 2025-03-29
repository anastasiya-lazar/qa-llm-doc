from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, List, Optional

from pydantic import BaseModel, ConfigDict


class LLMConfig(BaseModel):
    """Base configuration for LLM providers."""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    temperature: float = 0.7
    max_tokens: Optional[int] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    additional_params: Dict[str, Any] = {}


class LLMResponse(BaseModel):
    """Standardized response from LLM providers."""
    model_config = ConfigDict(protected_namespaces=())
    
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: Optional[str] = None
    additional_info: Dict[str, Any] = {}


class BaseLLMConnector(ABC):
    """Abstract base class for LLM connectors."""
    
    def __init__(self, config: LLMConfig):
        self.config = config
    
    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        """Generate a response from the LLM."""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        """Generate a streaming response from the LLM."""
        pass
    
    @abstractmethod
    async def get_embeddings(self, text: str) -> List[float]:
        """Get embeddings for the given text."""
        pass 