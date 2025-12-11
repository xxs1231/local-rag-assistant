# rag_bot.py - 适配 LangChain 1.1.3
print("🔍 正在测试 LangChain 1.x 环境...")

try:
    # ✅ LangChain 1.x 中 Document 来自 langchain_core
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_community.embeddings import SentenceTransformerEmbeddings
    from langchain_chroma import Chroma
    print("✅ 所有模块导入成功！")
except Exception as e:
    print("❌ 导入失败:", e)
    exit()

# 测试 Embedding 模型
print("📥 首次运行：正在下载 embedding 模型（all-MiniLM-L6-v2）...")
try:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    test_vector = embeddings.embed_query("Hello, world!")
    print(f"✅ 模型加载成功！向量维度: {len(test_vector)}")
except Exception as e:
    print("❌ 模型加载失败:", e)
    exit()

print("\n🎉 恭喜！你的 LangChain 1.x 开发环境已完美就绪！")