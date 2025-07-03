import os
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain_ollama import ChatOllama
from langchain.schema import Document
from newspaper import Article
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from typing import Optional, List
from langchain.prompts import PromptTemplate

class NewsArticleSummarizer:
    def __init__(self, model_name="llama3", model_type="ollama"):
        self.model_type = model_type
        self.model_name = model_name

        # Using ChatOllama with proper configuration
        if model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                format="json",
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=200,
            length_function=len
        )

    def fetch_article(self, url) -> Optional[Article]:
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article
        except Exception as e:
            print(f"Error fetching article: {e}")

    # create Langchain documnets from text
    def create_documents(self, text: str) -> List[Document]:
        texts = self.text_splitter.split_text(text)
        docs = [Document(page_content=t) for t in texts]
        return docs

    def summarize(self, url, summary_type="detailed") -> dict:
        # main summarization pipeline

        # fetch article
        article = self.fetch_article(url)
        if not article:
            return {"error": "Failed to fetch article"}

        # create documents
        docs = self.create_documents(article.text)

        # Define prompts based on summary type
        if summary_type == "detailed":
            map_prompt_template = """Write a detailed summary of the following text:
                '{text}'
                DETAILED SUMMARY:
            """

            combine_prompt_template = """Write a detailed summary of the following the
            '{text}'
            FINAL DETAILED SUMMARY:
            """

        else: # consise summary
            map_prompt_template = """Write a concise summary of the following text:
                '{text}'
                CONSISE SUMMARY:
            """

            combine_prompt_template = """Write a concise summary of the following the
            '{text}'
            FINAL CONSISE SUMMARY:
            """

        # create prompt
        map_prompt = PromptTemplate(
            template=map_prompt_template, input_variables=["text"]
        )

        combine_prompt = PromptTemplate(
            template=combine_prompt_template, input_variables=["text"]
        )

        # create and run chain
        chain = load_summarize_chain(
            llm=self.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True
        )

        # generate summary
        summary = chain.invoke(docs)

        return {
            "title": article.title,
            "authors": article.authors,
            "publish_date": article.publish_date,
            "summary": summary,
            "url": url,
            "model_info": {"type": self.model_type, "name": self.model_name},
        }


def main():

    url = "https://www.artificialintelligence-news.com/news/us-china-ai-chip-race-ci"

    ollama_summarizer = NewsArticleSummarizer(
        model_type="ollama", model_name="llama3"
    )

    print("\nGenerating Llama Summary...")
    llama_summary = ollama_summarizer.summarize(url, summary_type="detailed")

    # print result
    for summary, model in [(llama_summary, "Llama")]:
        print(f"\n{model} Summary:")
        print("-"*50)
        print(f"Title: {summary["title"]}")
        print(f"Authors: {', '.join(summary['authors'])}")
        print(f"Published: {summary['publish_date']}")
        print(
            f"Model: {summary['model_info']['type']} - {summary['model_info']['name']}"
        )
        print(f"Summary:\n{summary['summary']}")

        print("\nFirst Document Content:")
        print(summary['summary']['input_documents'][0].page_content)

if __name__ == "__main__":
    main()

