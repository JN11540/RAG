from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langchain_community.document_loaders import TextLoader
from langchain_ollama import OllamaEmbeddings  # <-- Updated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS  # <-- Updated
from dotenv import load_dotenv
import pprint
import re
import os

load_dotenv()

# 清理文本
def clean_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.lower()

# 載入文本
documents = TextLoader("./doc/dream.txt").load()

# 分割
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
texts = text_splitter.split_documents(documents)
texts = [clean_text(text.page_content) for text in texts]

# 使用 Ollama 模型產生 embedding
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 建立 FAISS 檢索器
retriever = FAISS.from_texts(texts, embeddings).as_retriever(search_kwargs={"k": 3})

# 詢問
query = "what did Martin Luther King Jr. dream about"
docs = retriever.invoke(query)

pprint.pprint(f" => DOCS: {docs}")

# Create the chat prompt
prompt = ChatPromptTemplate.from_template(
    "Please use the following docs {docs}, and answer the following question {query}",
)
# Create a chat model
model = ChatOllama(model="llama3")

# Chain the prompt, model and output parser
chain = prompt | model | StrOutputParser()

response = chain.invoke({"docs": docs, "query": query})
print(f"Model Response: \n\n{response}")

