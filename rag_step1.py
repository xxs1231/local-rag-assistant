# ====== 第 1 步：加载知识 ======

from langchain_core.documents import Document

raw_knowledege = """
LangChain 是一个用于开发大语言模型应用的框架。
它支持链式调用（Chains）、智能代理（Agents）、长期记忆（Memory）等功能。
RAG（检索增强生成）是一种结合外部知识库与语言模型的技术。
通过先检索相关文档，再让大模型基于这些文档生成答案，可以显著提高回答准确性。
Chroma 是一个轻量级向量数据库，适合嵌入到 Python 应用中。
"""
docs = [Document(page_content = raw_knowledege)]
print("知识加载完成！共一个文档")

# ====== 第 2步：切分文本 ======
from langchain_text_splitters import  RecursiveCharacterTextSplitter
# 创建切分器：按字符递归切分（优先按 "\n\n" → "\n" → " "）
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,# 每块最多 100 个字符
    chunk_overlap = 20 # 相邻块重叠 20 个字符（避免断句）
)
#执行切分
chunks = text_splitter.split_documents(docs)
print("文本切分为{}个块".format(len(chunks)))

#打印每块内容
for i,chunk in enumerate(chunks):
    print(f"[块 {i + 1}] {repr(chunk.page_content)}")

# ====== 第 3步：向量化+ 存入数据库
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from  langchain_chroma import  Chroma
print("正在将文本转换为向量（首次运行会下载模型）")

#创建Embedding 模型（本地运行，无需网络）
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

#创建 chroma向量数据库，并把chunks存进去
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings
)
print("向量数据库创建成功")

# ====== 第 4步,检索测试
#创建检索器，每次返回最相似的2个文本块
retriever = vectorstore.as_retriever(search_kwargs={"k":2})

#用户提问
question = "什么是RAG"

#执行搜索
results = retriever.invoke(question)

#输出结果
print("用户问{}?".format(question))
print("检索到的相关的内容：")
for  i,doc in enumerate(results):
    print(f"[{i + 1}] {doc.page_content.strip()}")

# ====== 第 5 步：交互式问答 ======
print("\n💬 现在进入问答模式！输入 '退出' 结束程序。")

while True:
    question = input("\n❓ 请输入你的问题: ").strip()
    if question in ["退出", "exit", "quit"]:
        print("👋 再见！")
        break
    if not question:
        continue

    results = retriever.invoke(question)
    print("📚 最相关的知识:")
    for i, doc in enumerate(results):
        print(f"[{i + 1}] {doc.page_content.strip()}")