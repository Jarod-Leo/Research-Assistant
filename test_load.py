from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("./models/Llama-3-8b-rm-mixture", device_map="auto")
print("模型加载成功！")

from datasets import load_from_disk

dataset = load_from_disk("./data/prompt-collection-v0.1")
print("数据集加载成功！样本数:", len(dataset))