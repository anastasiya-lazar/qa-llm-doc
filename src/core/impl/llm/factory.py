from enum import Enum
from typing import Dict, Type

from .base import BaseLLMConnector, LLMConfig
from .openai_connector import OpenAIConnector


class LLMProvider(str, Enum):
    """Supported LLM providers."""
    OPENAI = "openai"
    # Add more providers here as they are implemented


class LLMFactory:
    """Factory class for creating LLM connectors."""
    
    _providers: Dict[LLMProvider, Type[BaseLLMConnector]] = {
        LLMProvider.OPENAI: OpenAIConnector,
        # Add more providers here as they are implemented
    }
    
    @classmethod
    def create(cls, provider: LLMProvider, config: LLMConfig) -> BaseLLMConnector:
        """Create a new LLM connector instance."""
        if provider not in cls._providers:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        
        connector_class = cls._providers[provider]
        return connector_class(config)
    
    @classmethod
    def register_provider(cls, provider: LLMProvider, connector_class: Type[BaseLLMConnector]) -> None:
        """Register a new LLM provider."""
        if not issubclass(connector_class, BaseLLMConnector):
            raise ValueError(f"Connector class must inherit from BaseLLMConnector")
        cls._providers[provider] = connector_class 