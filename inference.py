from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 加载你保存的checkpoint路径
checkpoint_path = "./checkpoint/qwen2-0.5b-firefly-sft"

# 加载tokenizer和模型
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
model = AutoModelForCausalLM.from_pretrained(
    checkpoint_path,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 测试一个示例
def generate_response(instruction):
    input_text = f"Instruction: {instruction}\nOutput:"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=2048, truncation=True)
    inputs = inputs.to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# 测试示例
examples = [
    "解释一下量子计算的基本原理",
    "写一首关于春天的诗",
    "如何用Python实现快速排序?"
]

# 打开文件准备写入（'w'模式会覆盖原有内容，如需追加请用'a'模式）
with open("output_sft.txt", "w", encoding="utf-8") as f:
    for example in examples:
        # 控制台输出
        print(f"输入: {example}")
        print("输出:")
        response = generate_response(example)
        print(response)
        print("\n" + "="*50 + "\n")
        
        # 文件写入
        f.write(f"输入: {example}\n")
        f.write("输出:\n")
        f.write(response + "\n")
        f.write("\n" + "="*50 + "\n\n")

print("结果已保存到 output.txt")