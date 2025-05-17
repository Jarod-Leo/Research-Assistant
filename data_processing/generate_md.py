import os
import json
with open("llm_papers_translated.json", "r", encoding="utf-8") as f:
    papers = json.load(f)

os.makedirs("rag_paper_md", exist_ok=True)

for i, paper in enumerate(papers, 1):
    filename = f"rag_paper_md/{i:03d}_{paper['arxiv_id']}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# {paper['title']}\n\n")
        f.write(f"链接: {paper['url']}\n\n")
        f.write("原文摘要:\n")
        f.write(paper['abstract'] + "\n\n")
        f.write("中文翻译:\n")
        f.write(paper.get("abstract_cn", "暂无翻译") + "\n")
