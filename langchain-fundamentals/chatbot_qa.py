from langchain_ollama import OllamaEmbeddings  # <-- Updated
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from typing import Dict, List
from langchain.schema import Document

load_dotenv()

model_name="llama3"

documents = [
    "https://beebom.com/what-is-nft-explained/",
    "https://beebom.com/how-delete-servers-discord/",
    "https://beebom.com/how-list-groups-linux/",
    "https://beebom.com/linux-vs-windows/",
]

def scrape_docs(urls: List[str]) -> List[Document]:
    """Scrape content from URLs using SeleniumURLLoader"""
    try:
        loader = SeleniumURLLoader(urls=urls)
        raw_docs = loader.load()
        print(f"\nSuccessfully loaded {len(raw_docs)} documents")

        # Print some information about the loaded documents
        for doc in raw_docs:
            print(f"\nSource: {doc.metadata.get('source', 'No source')}")
            print(f"Content length: {len(doc.page_content)} characters")

        return raw_docs

    except Exception as e:
        print(f"Error during document loading: {str(e)}")
        return []

def create_vector_store(texts: List[str], metadatas: List[Dict]):
    """Create vector store using ChromaDB"""

    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_texts(texts=texts, metadatas=metadatas, embedding=embeddings)

    return db

def setup_qa_chain(db, model_name="llama3"):
    """Set up QA chain with polite response template"""
    llm = ChatOllama(model=model_name)
    retriever = db.as_retriever()

    # Create a custom prompt template
    prompt = ChatPromptTemplate.from_template(
        """
        Please provide a polite and helpful response to the following question, utilizing the provided context.
        Ensure that the tone remains professional, courteous, and empathetic.

        ### Context:
        {context}

        ### Question:
        {question}

        ### Polite Response:
        In your response, consider including:
        - Acknowledge the user's query and express gratitude for the opportunity.
        - Provide a clear and concise answer that directly addresses the question.
        - Use positive language and maintain a supportive tone throughout.
        - If applicable, include relevant information or resources.
        - Conclude by inviting any follow-up questions or encouragement.
        """
    )

    # Create the chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain, retriever

def process_query(chain_and_retriever, query):
    """Prcoess a query and return response"""
    try:
        chain, retriever = chain_and_retriever

        # Get the response
        response = chain.invoke(query)

        # Get the sources using the retriever
        docs = retriever.invoke(query)
        sources_str = ", ".join([doc.metadata.get("source", "") for doc in docs])

        return {"answer": response, "sources": sources_str}

    except Exception as e:
        print(f"Error processing query: {str(e)}")
        return {
            "answer": "I apologize, but I encountered an error while processing your question",
            "sources": "",
        }

def split_documents(pages_content: List[Dict]) -> tuple:
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

    all_texts, all_metadatas = [], []
    for document in pages_content:
        text = document.page_content
        source = document.metadata.get("source", "")

        chunks = text_splitter.split_text(text)
        for chunk in chunks:
            all_texts.append(chunk)
            all_metadatas.append({"source": source})

    print(f"Created {len(all_texts)} chunks of text")
    return all_texts, all_metadatas

def main():

    print("Scraping documnets...")
    pages_content = scrape_docs(documents)

    print(pages_content)

    print("Splitting documents")
    all_texts, all_metadatas = split_documents(pages_content)

    print("Creating vector store")
    db = create_vector_store(all_texts, all_metadatas)

    print("Setting up QA chain...")
    qa_chain = setup_qa_chain(db)

    print("\nReady for questions! (Type 'quit' to exit)")
    while True:
        query = input("\nEnter your question: ").strip()

        if not query:
            continue

        if query.lower() == "quit":
            break

        result = process_query(qa_chain, query)

        print("\nResponse")
        print(result["answer"])

        if result["sources"]:
            print("\nSources:")
            for source in result["sources"].split(","):
                print("- " + source.strip())

if __name__ == "__main__":
    main()


