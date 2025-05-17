import json
import re
import arxiv
from tqdm import tqdm  # 用于显示进度条

file_path = "./nlp-arxiv-daily.json"
with open(file_path, "r", encoding="utf-8") as f:
    data = json.load(f)

nlp_data = data.get("NLP", {})

llm_keywords = [
    "language model", "gpt", "llm", "chatgpt", "transformer",
    "few-shot", "zero-shot", "prompt", "pretrain", "autoregressive",
    "instruction", "alignment", "openai"
]

llm_related = []
for arxiv_id, entry in tqdm(nlp_data.items(), desc="Processing papers"):
    if any(keyword.lower() in entry.lower() for keyword in llm_keywords):
        # 提取第二个加粗内容作为标题
        title_matches = re.findall(r"\*\*(.*?)\*\*", entry)
        title = title_matches[1].strip() if len(title_matches) > 1 else "未知标题"

        link_match = re.search(r"\[([0-9.]+v[0-9]+)\]\((.*?)\)", entry)
        url = link_match.group(2) if link_match else f"https://arxiv.org/abs/{arxiv_id}"
        
        # 获取摘要
        try:
            search = arxiv.Search(id_list=[arxiv_id])
            paper = next(search.results())
            abstract = paper.summary
        except Exception as e:
            print(f"Error fetching abstract for {arxiv_id}: {e}")
            abstract = "摘要获取失败"
        
        llm_related.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "url": url,
            "abstract": abstract
        })

# 可选：将结果保存到文件
with open("llm_related_with_abstracts.json", "w", encoding="utf-8") as f:
    json.dump(llm_related, f, ensure_ascii=False, indent=2)