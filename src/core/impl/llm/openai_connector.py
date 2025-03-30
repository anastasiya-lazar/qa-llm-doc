from typing import Any, AsyncGenerator, Dict, List, Optional

from langchain_openai import ChatOpenAI
from openai import AsyncOpenAI

from .base import BaseLLMConnector, LLMConfig, LLMResponse


class OpenAIConnector(BaseLLMConnector):
    """OpenAI implementation of the LLM connector."""

    def __init__(self, config: LLMConfig):
        super().__init__(config)
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.api_base,
        )
        self.langchain_llm = ChatOpenAI(
            model_name=config.model_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            openai_api_key=config.api_key,
            openai_api_base=config.api_base,
            **config.additional_params
        )

    async def generate(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> LLMResponse:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            **kwargs
        )

        return LLMResponse(
            content=response.choices[0].message.content,
            model=response.model,
            usage=response.usage.model_dump(),
            finish_reason=response.choices[0].finish_reason,
            additional_info={
                "role": response.choices[0].message.role,
                "index": response.choices[0].index,
            },
        )

    async def generate_stream(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs: Any
    ) -> AsyncGenerator[LLMResponse, None]:
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True,
            **kwargs
        )

        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield LLMResponse(
                    content=chunk.choices[0].delta.content,
                    model=chunk.model,
                    usage={},  # Usage is not available for streaming responses
                    finish_reason=chunk.choices[0].finish_reason,
                    additional_info={
                        "role": chunk.choices[0].delta.role,
                        "index": chunk.choices[0].index,
                    },
                )

    async def get_embeddings(self, text: str) -> List[float]:
        response = await self.client.embeddings.create(
            model="text-embedding-ada-002", input=text
        )
        return response.data[0].embedding
