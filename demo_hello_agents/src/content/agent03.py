from typing import Dict, List
from src.agent.models import LLMConfig
from src.agent.utils import build_openapi_client


class HelloAgentsLLM:
    def __init__(self, config: LLMConfig):
        self.config = config
        print(f"Initialized HelloAgentsLLM with model_id: {self.config.model_id}")
        self.client = build_openapi_client(config)
        
    def thinking(self, messages: List[Dict[str, str]], temperature: float = 0.) -> str:
        print(f"invoke model: {self.config.model_id} with temperature: {temperature}")
        try:
            response = self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                temperature=temperature,
                stream=True
            )
            print("Model invocation successful.")
            collected_content = []
            for chunk in response:
                chunk_content = chunk.choices[0].delta.content or ""
                print(chunk_content, end="", flush=True)
                collected_content.append(chunk_content)
            print()  # for newline after streaming
            return "".join(collected_content)
        except Exception as e:
            print(f"Error invoking LLM API: {e}")
            return None



if __name__ == "__main__":
    config = LLMConfig()
    llm = HelloAgentsLLM(config=config)
    
    system_prompt = "You are a helpful assistant."
    user_prompt = "你好，请介绍你自己。"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    response = llm.thinking(messages=messages, temperature=0.5)
    print("Final Response:")
    print(response)