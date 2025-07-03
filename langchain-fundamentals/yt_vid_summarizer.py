import yt_dlp
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.summarize import load_summarize_chain
from langchain.schema import Document
import whisper
from typing import List, Dict
import os

from dotenv import load_dotenv

load_dotenv()

class EmbeddingModel:
    def __init__(self, model_type="nomic"):
        self.model_type = model_type
        if model_type == "chroma":
            from langchain.embeddings import HuggingFaceEmbeddings

            self.embedding_fn = HuggingFaceEmbeddings()
        elif model_type == "nomic":
            self.embedding_fn = OllamaEmbeddings(
                model="nomic-embed-text", base_url="http://localhost:11434"
            )
        else:
            raise ValueError(f"Unsupported embedding type: {model_type}")

class LLMModel:
    def __init__(self, model_type="ollama", model_name="llama3"):
        self.model_type = model_type
        self.model_name = model_name

        # Using ChatOllama with proper configuration
        if model_type == "ollama":
            self.llm = ChatOllama(
                model=model_name,
                temperature=0,
                timeout=120,
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

class YoutubeVideoSummarizer:
    def __init__(
        self, llm_type="ollama", llm_model_name="llama3", embedding_type="nomic"
    ):
        self.embedding_model = EmbeddingModel(embedding_type)
        self.llm_model = LLMModel(llm_type, llm_model_name)

        self.whisper_model = whisper.load_model("base")

    def get_model_info(self) -> Dict:
        return {
            "llm_type": self.llm_model.model_type,
            "llm_model": self.llm_model.model_name,
            "embedding_type": self.embedding_model.model_type,
        }

    def download_video(self, url: str) -> tuple[str,str]:
        """Download video and extract audio"""
        print("Downloading video...")
        ydl_opts = {
            "format": "bestaudio/best",
            "postprocessors": [
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",
                }
            ],
            "outtmpl": "downloads/%(title)s.%(ext)s",
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(info).replace(".webm", ".mp3")
            video_title = info.get("title", "Unknown Title")
            return audio_path, video_title

    def transcribe_audio(self, audio_path: str) -> str:
        """Transcribe audio using Whisper"""
        print("Transcribing audio...")

        result = self.whisper_model.transcribe(audio_path)
        return result["text"]

    def create_documents(self, text: str, video_title: str) -> List[Document]:
        print("Creating Documents...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", ", ", ", ", " "]
        )
        texts = text_splitter.split_text(text)
        return [
            Document(page_content=chunk, metadata={"source": video_title})
            for chunk in texts
        ]

    def create_vector_store(self, documents: List[Document]) -> Chroma:
        print(f"Creating vector store using {self.embedding_model.model_type} embeddings...")
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embedding_model.embedding_fn,
            collection_name=f"youtube_summary_{self.embedding_model.model_type}",
        )

    def generate_summary(self, documents: List[Document]) -> str:
        print("Generating summary...")
        map_prompt = ChatPromptTemplate.from_template(
            """Write a concise summary of the following transcript section:
            "{text}"
            CONCISE SUMMARY:
            """
        )

        combine_prompt = ChatPromptTemplate.from_template(
            """Write a detailed summary of the following video transcript section:
            "{text}"

            Include:
            - Main topics and key points
            - Important details and examples
            - Any conclusions or call to action


            DETAILED SUMMARY:
            """
        )

        summary_chain = load_summarize_chain(
            llm=self.llm_model.llm,
            chain_type="map_reduce",
            map_prompt=map_prompt,
            combine_prompt=combine_prompt,
            verbose=True,
        )

        return summary_chain.invoke(documents)

    def setup_qa_chain(self, vector_store: Chroma):
        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm_model.llm,
            retriever=vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )

    def process_video(self, url: str) -> Dict:
        try:
            os.makedirs("downloads", exist_ok=True)

            audio_path, video_title = self.download_video(url)
            transcript = self.transcribe_audio(audio_path)
            documents = self.create_documents(transcript, video_title)
            summary = self.generate_summary(documents)
            vector_store = self.create_vector_store(documents)
            qa_chain = self.setup_qa_chain(vector_store)

            os.remove(audio_path)
            return {
                "summary": summary,
                "qa_chain": qa_chain,
                "title": video_title,
                "full_transcript": transcript,
            }

        except Exception as e:
            print(f"Error processing video: {str(e)}")
            return None

def main():
    # Get Model preferences
    print("\nAvailable LLM Models")
    print("1. Ollama Llama3")

    llm_choice = input("Choose LLM Model: ").strip()

    print("\nAvailable Embeddings: ")
    print("1. Chroma Default")
    print("2. Nomic (via Ollama)")
    embedding_choice = input("Choose embeddings: ").strip()

    # Configure model settings
    llm_type = "ollama"
    llm_model_name = "llama3"

    if embedding_choice == "1":
        embedding_type = "chroma"
    else:
        embedding_type = "nomic"

    try:
        # Initialize summarizer
        summarizer = YoutubeVideoSummarizer(
            llm_type=llm_type,
            llm_model_name=llm_model_name,
            embedding_type=embedding_type
        )

        # Display configuration
        model_info = summarizer.get_model_info()
        print("\nCurrent Configuration:")
        print(f"LLM {model_info['llm_type']} ({model_info['llm_model']})")
        print(f"Embeddings: {model_info['embedding_type']}")

        url = input("\nEnter Youtube URL: ")
        print(f"\n Processing video...")
        result = summarizer.process_video(url)

        if result:
            print(f"\nVideo Title: {result['title']}")
            print("\nSummary:")

            # Interactive Q&A
            print("\nYou can ask questions about the video (type 'quit' to exit)")
            while True:
                query = input("\nYour question: ").strip()
                if query.lower() == "quit":
                    break
                if query:
                    response = result["qa_chain"].invoke({"question": query})
                    print("response: ", response)
                    print("\nAnswer:", response["answer"])

            if input("\nWant to see the full transcript? (y/n): ").lower() == "y":
                print("\nFull Transcript:")
                print(result["full_transcript"])

    except Exception as e:
        print(f"Error: {str(e)}")
        print("Make sure required model and APIs are properly configured.")

if __name__ == "__main__":
    main()