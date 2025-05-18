# 创建文档加载器
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# 加载MD文件
loader = DirectoryLoader('data_processing/rag_paper_md/', glob="**/*.md")
docs = loader.load()

# 中文文本分割器
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    separators=["\n#", "\n##", "\n###", "\n\n", "\n", "。", "！", "？"]
)

splits = text_splitter.split_documents(docs)

# 选择嵌入模型
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={'device': 'cuda'}  
)

# 构建向量数据库
from langchain.vectorstores import FAISS

vector_db = FAISS.from_documents(splits, embedding_model)
vector_db.save_local("faiss_db")  # 保存以备后用

# 加载SFT调好的模型
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "/workspace/openrlhf_sft/checkpoint/qwen2-0.5b-firefly-sft"  # 或 "qwen/Qwen1.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

# 创建Langchain llm包装器
from langchain.llms import HuggingFacePipeline
from transformers import pipeline

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
)

llm = HuggingFacePipeline(pipeline=pipe)
# 构建查询接口
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFacePipeline  # 或 OpenAI

retriever = vector_db.as_retriever()
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,  # 你微调的 Qwen 模型或 ChatGPT API
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# test_output = llm("你好，你是谁？")
# print(test_output)
response = qa_chain.invoke({"query": "请总结近期LLM在跨语言应用方面有哪些进展？"})
print(response["result"])  # 获取答案
print(response["source_documents"])  # 获取来源文档
