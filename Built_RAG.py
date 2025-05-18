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
import os
db_path = "faiss_db"

if not os.path.exists(db_path):
    # 如果向量库文件夹不存在，说明需要创建
    print("未检测到向量数据库，正在创建...")
    vector_db = FAISS.from_documents(splits, embedding_model)
    vector_db.save_local(db_path)
else:
    # 否则直接加载已有数据库
    print("已检测到向量数据库，直接加载...")
    vector_db = FAISS.load_local(db_path, embedding_model, allow_dangerous_deserialization=True)


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

# 创建检索器
retriever = vector_db.as_retriever(search_kwargs={"k": 3}) # 每次检索时返回 最相似的 3 个文本片段

# 设计提示词模板
from langchain.prompts import PromptTemplate

template = """你是一个专业的中文科研助手，请根据以下上下文信息回答问题。
如果不知道答案，就回答不知道，不要编造答案。

上下文：
{context}

问题：{question}
专业回答："""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
) 
# 构建查询接口
from langchain.chains import RetrievalQA

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

question = "Transformer在文本摘要中有哪些应用？"
result = qa_chain({"query": question})
print(result["result"])