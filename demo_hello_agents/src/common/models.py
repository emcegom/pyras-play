from dataclasses import dataclass, field
import os
from dotenv import load_dotenv
from openai import OpenAI
load_dotenv()

@dataclass
class LLMConfig:
    api_key: str = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    model_id: str = field(default_factory=lambda: os.getenv("LLM_MODEL_ID"))
    base_url: str = field(default_factory=lambda: os.getenv("LLM_BASE_URL"))
    timeout: int = field(default_factory=lambda: int(os.getenv("LLM_TIMEOUT", 60)))

    def __post_init__(self):
        if not all([self.api_key, self.model_id, self.base_url]):
            raise ValueError(
                "LLM configuration is incomplete. Please check environment variables."
            )



def build_openapi_client(config: LLMConfig) -> OpenAI:
    client = OpenAI(
        api_key=config.api_key, base_url=config.base_url, timeout=config.timeout
    )
    return client


@dataclass
class Configuration:
    search_api_key: str = field(default_factory=lambda: os.getenv("SEARCH_API_KEY"))

configuration = Configuration()