from huggingface_hub import snapshot_download

# 下载模型
snapshot_download(
    repo_id="OpenRLHF/Llama-3-8b-rm-mixture",
    local_dir="./models/Llama-3-8b-rm-mixture",
    resume_download=True
)

from datasets import load_dataset

# 下载并保存数据集
dataset = load_dataset("OpenRLHF/prompt-collection-v0.1")
dataset.save_to_disk("./data/prompt-collection-v0.1")