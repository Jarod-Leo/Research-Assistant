# 创建文档加载器
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np
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
from langchain_huggingface import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(
    model_name="GanymedeNil/text2vec-large-chinese",
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}  # 强制归一化  
)

# 构建向量数据库
from langchain_community.vectorstores import FAISS
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

model_path = "/workspace/Research-Assistant/qwen2-0.5b-firely-sft"  # 或 "qwen/Qwen1.5-0.5B"

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto"
)

# 创建Langchain llm包装器
from langchain_huggingface import HuggingFacePipeline
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

def query_rewrite(original_query: str) -> str:
    """
    对用户原始查询进行语义优化，使其更利于检索
    :param original_query: 用户原始查询
    :return: 改写后的查询字符串（确保不包含提示词）
    """
    try:
        # 使用更严格的提示词模板
        rewrite_prompt = f"""请将以下查询改写成更专业的检索问题，保持原意不变。
只需输出改写后的查询，不要包含任何解释、提示词或额外文本。

原始查询：{original_query}
改写后查询："""
        
        # 调用模型并清理输出
        rewritten_query = llm.invoke(rewrite_prompt).strip()
        
        # 彻底移除提示词残留（三种清理方式）
        if "改写后查询：" in rewritten_query:
            rewritten_query = rewritten_query.split("改写后查询：")[-1].strip()
        if "\n" in rewritten_query:
            rewritten_query = rewritten_query.split("\n")[0].strip()
        if rewritten_query.startswith("原始查询："):
            rewritten_query = original_query  # 失败时回退
            
        return rewritten_query if rewritten_query else original_query
    except Exception as e:
        print(f"查询改写失败: {e}")
        return original_query

# 2. 检索&重排功能
def retrieve_and_rerank(query: str, k: int = 2):
    """
    检索并重排结果
    :param query: 查询文本
    :param k: 返回的结果数量
    :return: 包含距离分数的结果列表
    """
    try:
        # 获取查询文本的嵌入向量
        query_embedding = embedding_model.embed_query(query)

        
        # 使用FAISS搜索
        scores, indices = vector_db.index.search(query_embedding, k)
        
        # 构造结果列表
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:  # FAISS返回-1表示没有足够的结果
                continue
            doc_idx = vector_db.index_to_docstore_id[idx]
            doc = vector_db.docstore.search(doc_idx)
            results.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score)  # 将距离转换为相似度分数(0-1)
            })
        
        # 按相似度分数降序排序
        results.sort(key=lambda x: x["score"], reverse=True)
        return results
    
    except Exception as e:
        print(f"检索失败: {e}")
        return []

# 创建检索器
retriever = vector_db.as_retriever(search_kwargs={"k": 2})

# 设计提示词模板
from langchain.prompts import PromptTemplate

# 构建查询接口
from langchain.chains import RetrievalQA

# 更新提示词模板，确保输入变量名匹配
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

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    chain_type_kwargs={
        "prompt": prompt
    },
    return_source_documents=True,
    input_key="question",  # 明确指定输入键
    output_key="result"    # 明确指定输出键
)

# 增强的问答函数
def enhanced_qa(question: str):
    # 1. 查询改写
    rewritten_query = query_rewrite(question)
    # print(f"原始查询: {question}")
    # print(f"改写后的查询: {rewritten_query}")
    
    # 2. 检索&重排 - 使用最新的invoke方法
    rerank_results = retrieve_and_rerank(rewritten_query, k=5)
    
    # 提取得分列表
    scores = [result["score"] for result in rerank_results]
    # print("\n召回文档得分列表:", scores)

    retrieved_docs = retriever.invoke(rewritten_query)
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    # print(f"\n构建的上下文: {context[:200]}...")  # 打印上下文前200个字符
    
    # 3. 使用改写后的查询进行问答 - 确保使用正确的输入键
    response = qa_chain.invoke({"question": rewritten_query})  # 注意这里使用"question"作为键

    # 4. 提取"专业回答："后的内容
    full_answer = response["result"].strip()
    if "专业回答：" in full_answer:
        # 找到"专业回答："的位置并提取后面的内容
        answer = full_answer.split("专业回答：")[-1].strip()
    else:
        answer = full_answer  # 如果没有找到标记，返回完整回答
    
    # 5. 返回精简后的结果
    return {
        "original_query": question,
        "rewritten_query": rewritten_query,
        "answer": answer,
        "source_documents": [doc["content"] for doc in rerank_results],  # 使用rerank结果的文档
        "retrieval_scores": scores  # 得分列表
    } 
    
# 示例使用
question = "Transformer在文本摘要中有哪些应用?"
result = enhanced_qa(question)
print(f"原始查询: {result["original_query"]}")
print(f"改写后的查询: {result["rewritten_query"]}")
print("\n=== 最终回答 ===")
print(result["answer"])
print("\n=== 召回得分 ===")
print(result["retrieval_scores"])  # 打印得分列表
print("\n=== 来源文档 ===")
for i, doc in enumerate(result["source_documents"][:3], 1):
    print(f"\n文档{i} [相似度:{result['retrieval_scores'][i-1]:.2f}]: {doc[:200]}...")