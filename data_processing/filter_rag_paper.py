import re
import os

def clean_markdown(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 定义要删除的内容模式
    patterns_to_remove = [
        r'\（注：.*?\）',  # 移除中文括号内的注释
        r'代码已开源：.*',  # 移除代码开源声明
        r'https://github.com/.*',  # 移除github链接
        # 可以添加更多模式...
    ]
    
    # 应用正则表达式移除内容
    for pattern in patterns_to_remove:
        content = re.sub(pattern, '', content, flags=re.DOTALL)
    
    # 保存清理后的内容
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"已清理文件并保存至: {output_file}")

def process_directory(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.endswith('.md'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            clean_markdown(input_path, output_path)
            print(f"已处理: {filename}")

# 批量处理目录中的所有Markdown文件
process_directory('translation_paper_md', 'cleaned_translation_md')