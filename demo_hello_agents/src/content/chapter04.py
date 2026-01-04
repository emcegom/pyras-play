from typing import Dict, List, Any

from serpapi import SerpApiClient

from demo_hello_agents.src.common.models import LLMConfig, build_openapi_client, configuration


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


def search(query: str) -> str:
    print(f"Searching for '{query}'")
    if not configuration.search_api_key:
        return "No search API key provided."
    params = {
        "engine": "google",
        "q": query,
        "api_key": configuration.search_api_key,
        "gl": "cn",
        "hl": "zh-cn"
    }
    try:
        client = SerpApiClient(params)
        results = client.get_dict()
        if "answers_box_list" in results:
            return "\n".join(results["answers_box_list"])
        if "answers_box" in results:
            return results["answers_box"]["answer"]
        if "knowlege_graph" in results and "description" in results["knowlege_graph"]:
            return results["knowlege_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                f"[{i + 1}] {res.get('title', '')}\n{res.get('snippet', '')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"Sorry for no relevant information abount {query} found."
    except Exception as e:
        return f"Error searching: {e}"


class ToolExecutor:
    def __init__(self):
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: callable):
        if name in self.tools:
            print(f"Warning: Tool '{name}' is already registered. Overwriting it.")
        self.tools[name] = {"description": description, "func": func}
        print(f"Registered tool: {name}")

    def getTool(self, name: str) -> callable:
        return self.tools.get(name, {}).get("func")

    def getAvailableTools(self) -> str:
        return "\n".join([
            f"- {name}: {info['description']}" for name, info in self.tools.items()
        ])


def test_agents():
    llm = HelloAgentsLLM(config=LLMConfig())

    system_prompt = "You are a helpful assistant."
    user_prompt = "你好，请介绍你自己。"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.thinking(messages=messages, temperature=0.5)
    print("Final Response:")
    print(response)


def test_search():
    print(search("今天的天气如何"))

def test_tool_executor():
    # 1. 初始化工具执行器
    toolExecutor = ToolExecutor()

    # 2. 注册我们的实战搜索工具
    search_description = "一个网页搜索引擎。当你需要回答关于时事、事实以及在你的知识库中找不到的信息时，应使用此工具。"
    toolExecutor.registerTool("Search", search_description, search)

    # 3. 打印可用的工具
    print("\n--- 可用的工具 ---")
    print(toolExecutor.getAvailableTools())

    # 4. 智能体的Action调用，这次我们问一个实时性的问题
    print("\n--- 执行 Action: Search['英伟达最新的GPU型号是什么'] ---")
    tool_name = "Search"
    tool_input = "英伟达最新的GPU型号是什么"

    tool_function = toolExecutor.getTool(tool_name)
    if tool_function:
        observation = tool_function(tool_input)
        print("--- 观察 (Observation) ---")
        print(observation)
    else:
        print(f"错误:未找到名为 '{tool_name}' 的工具。")


if __name__ == "__main__":
    # unittest.main()
    test_tool_executor()
    print()
