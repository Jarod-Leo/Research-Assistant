import argparse
import requests
import time
import json
from tqdm import tqdm
import random

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

def preview_translations(papers, num_samples=3):
    """随机预览多个文档的翻译结果"""
    print(f"\n=== 翻译预览模式 (随机选择 {num_samples} 个文档) ===")
    
    for i in range(num_samples):
        index = random.randint(0, len(papers) - 1)
        paper = papers[index]
        
        print(f"\n文档 #{index}: {paper.get('title', '无标题')}")
        print("-" * 50)
        print("原文摘要:")
        print(paper["abstract"])
        
        # 执行翻译
        translation = translate_abstract(paper["abstract"])
        print("\n翻译结果:")
        print(translation)
        
        # 等待用户确认
        input("\n按Enter继续预览下一个文档... ")
    
    # 询问是否继续完整翻译
    confirm = input("\n是否继续执行完整翻译？(y/n): ").lower().strip()
    return confirm == 'y'

def main():
    parser = argparse.ArgumentParser(description="使用DeepSeek API翻译论文摘要")
    parser.add_argument("--preview", action="store_true", help="预览模式：随机展示几个翻译结果")
    parser.add_argument("--samples", type=int, default=3, help="预览模式下的样本数量")
    args = parser.parse_args()
    
    # 加载数据
    with open("llm_related_with_abstracts.json", "r", encoding="utf-8") as f:
        papers = json.load(f)
    
    # 预览模式
    if args.preview:
        if not preview_translations(papers, args.samples):
            print("程序已取消。")
            return
    
    # 正常翻译模式
    print("\n开始批量翻译...")
    for paper in tqdm(papers, desc="Translating"):
        if "abstract" in paper:
            paper["abstract_cn"] = translate_abstract(paper["abstract"])
        else:
            paper["abstract_cn"] = "无摘要"
    
    # 保存结果
    with open("llm_papers_translated.json", "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    print("翻译完成！结果已保存到 llm_papers_translated.json")

if __name__ == "__main__":
    main()
