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

# 1. 查询改写功能
def query_rewrite(original_query: str) -> str:
    """
    对用户原始查询进行语义优化，使其更利于检索
    :param original_query: 用户原始查询
    :return: 改写后的查询字符串，若失败则返回原始查询
    """
    try:
        # 使用LLM进行查询改写
        rewrite_prompt = f"""请将以下用户查询改写为更专业、更利于信息检索的形式，保持原意不变。
只需输出改写后的查询，不要添加任何解释。

原始查询：{original_query}
改写后的查询："""
        
        rewritten_query = llm(rewrite_prompt)
        # 清理输出，去除可能的额外内容
        rewritten_query = rewritten_query.strip().split('\n')[0]
        return rewritten_query if rewritten_query else original_query
    except Exception as e:
        print(f"查询改写失败: {e}")
        return original_query

# 2. 检索&重排功能
def retrieve_and_rerank(query: str, k: int = 3):
    """
    检索并重排结果
    :param query: 查询文本
    :param k: 返回的结果数量
    :return: 包含距离分数的结果列表
    """
    try:
        # 获取查询文本的嵌入向量
        query_embedding = embedding_model.embed_query(query)
        
        # 归一化处理
        import numpy as np
        query_embedding = np.array(query_embedding)
        query_embedding = query_embedding / np.linalg.norm(query_embedding)
        
        # 使用FAISS搜索
        scores, indices = vector_db.index.search(np.array([query_embedding]), k)
        
        # 构造结果列表
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示没有足够的结果
                continue
            doc = vector_db.index_to_docstore_id[idx]
            doc = vector_db.docstore.search(doc)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(1 - score)  # 将距离转换为相似度分数(0-1)
            })
        
        # 按相似度分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    except Exception as e:
        print(f"检索失败: {e}")
        return []

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

# 增强的问答函数
def enhanced_qa(question: str):
    # 1. 查询改写
    rewritten_query = query_rewrite(question)
    print(f"原始查询: {question}")
    print(f"改写后的查询: {rewritten_query}")
    
    # 2. 检索&重排
    retrieved_results = retrieve_and_rerank(rewritten_query)
    print("\n检索结果(带相似度分数):")
    for i, res in enumerate(retrieved_results, 1):
        print(f"\n结果 {i} (相似度: {res['score']:.4f}):")
        print(res["content"][:200] + "...")  # 只打印前200字符
    
    # 3. 使用改写后的查询进行问答
    result = qa_chain({"query": rewritten_query})
    
    # 4. 返回结果
    return {
        "original_query": question,
        "rewritten_query": rewritten_query,
        "answer": result["result"],
        "source_documents": result["source_documents"]
    }

# 示例使用
question = "Transformer在文本摘要中有哪些应用？"
result = enhanced_qa(question)
print("\n最终答案:")
print(result["answer"])