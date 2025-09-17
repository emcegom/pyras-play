import os

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient
from datetime import datetime

load_dotenv()


def check_env_vars():
    missing_vars = []
    for var in ['TAVILY_API_KEY', 'OPENAI_API_KEY']:
        if not os.getenv(var):
            missing_vars.append(var)
    if missing_vars:
        raise ValueError(f"缺少必要的环境变量: {', '.join(missing_vars)}")


class Translator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    def translate(self, text):
        prompt = f"请将以下英文文本翻译成中文：{text}"
        response = self.client.chat.completions.create(
            model="qwen3-max",  # 使用经济实惠的 GPT-4o-mini 模型
            messages=[
                {"role": "system", "content": "你是一个专业的英译中翻译器。请将英文准确翻译成中文，保持专业性。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3  # 降低随机性，使翻译更稳定
        )
        return response.choices[0].message.content.strip()


class TavilyNewsSearch:
    def __init__(self, tavily_api_key, openai_api_key):
        self.client = TavilyClient(api_key=tavily_api_key)
        self.translator = Translator(api_key=openai_api_key)

    def search_news(self, query, days=7, max_results=5):
        response = self.client.search(
            query=query,
            topic="news",  # 指定搜索新闻
            days=days,  # 最近几天的新闻
            max_results=max_results,
            include_answer=True
        )
        return response

    def process_results(self, response):
        processed_data = {"news": []}
        for result in response.get("results", []):
            translated_title = self.translator.translate(result["title"])
            translated_content = self.translator.translate(result["content"])
            processed_data["news"].append({
                "title": result["title"],
                "title_zh": translated_title,
                "content": result["content"],
                "content_zh": translated_content,
                "url": result["url"]
            })
        return processed_data

    def save_to_markdown(self, processed_data, query):
        # 创建输出目录
        output_dir = "news_summaries"
        os.makedirs(output_dir, exist_ok=True)

        # 生成文件名（日期_查询词.md）
        date_str = datetime.now().strftime("%Y%m%d")
        filename = f"{date_str}_{query.replace(' ', '_')}.md"
        filepath = os.path.join(output_dir, filename)

        # 生成Markdown内容
        content = [f"# {query}相关新闻总结({date_str})\n"]

        for i, news in enumerate(processed_data["news"], 1):
            content.extend([
                f"## {i}. {news['title_zh']}",
                f"原标题：{news['title']}\n",
                f"###中文摘要",
                f"{news['content_zh']}\n",
                f"###原文",
                f"{news['content']}\n",
                f"来源：{news['url']}\n",
                "---\n"
            ])

        # 写入文件
        with open(filepath, "w", encoding="utf-8") as f:
            f.write("\n".join(content))

        return filepath


if __name__ == '__main__':
    # 检查环境变量
    check_env_vars()
    tavily_api_key = os.getenv('TAVILY_API_KEY')
    openai_api_key = os.getenv('OPENAI_API_KEY')

    # 创建搜索器实例
    searcher = TavilyNewsSearch(tavily_api_key, openai_api_key)

    # 设置搜索参数
    query = "AI Agent"
    days = 7
    max_results = 5

    # 执行搜索和处理
    print(f"正在搜索最近{days}天的{query}相关新闻...")
    response = searcher.search_news(query=query, days=days, max_results=max_results)

    print("正在翻译和处理结果...")
    processed_data = searcher.process_results(response)

    # 保存为Markdown文件
    print("正在生成Markdown文件...")
    filepath = searcher.save_to_markdown(processed_data, query)

    print(f"\n✅处理完成！")
    print(f" 结果已保存到：{filepath}")
    print(f" 共整理了{len(processed_data['news'])}条新闻")
