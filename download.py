from datasets import load_dataset
#加载数据集的 firefly 文件夹
dataset = load_dataset("QingyiSi/Alpaca-CoT", data_dir="firefly")
#打印数据集的一些信息
print(dataset)
print(dataset["train"] [0])
from transformers import AutoModelForCausalLM, AutoTokenizer
# 加载模型和tokenizer
model_name = "Qwen/Qwen2-0.5B"
model = AutoModelForCausalLM. from_pretrained(model_name, trust_remote_code=True)
tokenizer = AutoTokenizer. from_pretrained(model_name, trust_remote_code=True)
#打印模型的一些信息
print(model)