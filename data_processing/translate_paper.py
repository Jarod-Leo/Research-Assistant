import requests
import time
import json
from tqdm import tqdm

DEEPSEEK_API_KEY = "sk-cc40a1764a6c4115b33ce9b9b91e7024"
API_URL = "https://api.deepseek.com/v1/chat/completions"

def translate_abstract(abstract):
    headers = {
        "Authorization": f"Bearer {DEEPSEEK_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {
                "role": "system", 
                "content": "你是一个专业的学术翻译助手。用户会提供英文论文摘要，请仅返回翻译后的中文内容，不要添加任何标题、解释或其他无关信息。"
            },
            {
                "role": "user", 
                "content": abstract
            }
        ],
        "temperature": 0.1  # 进一步降低随机性
    }
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"翻译失败: {e}")
        return "翻译失败"
        
# 加载之前的摘要数据
with open("llm_related_with_abstracts.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

# 翻译并添加字段
for paper in tqdm(papers, desc="Translating"):
    if "abstract" in paper:
        paper["abstract_cn"] = translate_abstract(paper["abstract"])
        #time.sleep(1.5)  # 避免触发 API 速率限制
    else:
        paper["abstract_cn"] = "无摘要"

# 保存含中文摘要的新 JSON
with open("llm_papers_translated.json", "w", encoding="utf-8") as f:
    json.dump(papers, f, ensure_ascii=False, indent=2)