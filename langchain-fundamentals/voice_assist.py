import streamlit as st
import whisper
import sounddevice as sd
import soundfile as sf
from elevenlabs.client import ElevenLabs

from langchain_community.vectorstores import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)

import tempfile
from dotenv import load_dotenv
import os
from typing import List
from langchain_core.documents import Document

load_dotenv()

class DocunmentProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200, separators=["\n\n", "\n", ", ", ", ", " "]
        )
        self.embeddings = OllamaEmbeddings()

    def load_documents(self, directory: str) -> List[Document]:
        loaders = {
            ".pdf": DirectoryLoader(directory, glob="**/*.pdf", loader_cls=PyPDFLoader),
            ".txt": DirectoryLoader(directory, glob="**/*.txt", loader_cls=TextLoader),
            ".md": DirectoryLoader(directory, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader),
        }

        documents = []
        for file_type, loader in loaders.items():
            try:
                documents.extend(loader.load())
                print(f"Loaded {file_type} documents")
            except Exception as e:
                print(f"Error loading {file_type} documnets: {str(e)}")

        return documents

    def process_documents(self, documents: List[Document]) -> List[Document]:
        return self.text_splitter.split_documents(documents)

    def create_vector_store(
        self, documents: List[Document], persist_directory:str
    ) -> Chroma:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            print(f"Loading existing vector store from {persist_directory}")

            vector_store = Chroma(
                persist_directory=persist_directory, embedding_fucntion=self.embeddings
            )
        else:
            os.makedirs(persist_directory, exist_ok=True)

            vector_store = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=persist_directory,
            )
            vector_store.persist()

        return vector_store

class VoiceGenerator:
    def __init__(self, api_key):
        self.client = ElevenLabs(api_key=api_key)

        self.available_voice = [
            "Rachel",
            "Domi",
            "Bella",
            "Antoni",
            "Elli",
            "Josh",
            "Arnold",
            "Adam",
            "Sam"
        ]
        self.default_voice = "Rachel"

    def generate_voice_response(self, text: str, voice_name: str = None) -> str:
        try:
            selected_voice = voice_name or self.default_voice
            audio_generator = self.client.generate(
                text=text, voice=selected_voice, model="eleven_multilingual_v2"
            )

            audio_bytes = b"".join(audio_generator)

            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
                temp_audio.write(audio_bytes)
                return temp_audio.name

        except Exception as e:
            print(f"Error generating voice response: {e}")
            return None

class VoiceAssistantRAG:
    def __init__(self, elevenlabs_api_key)
        self.whisper_model = whisper.load_model("base")
        self.llm = ChatOllama(model="llama3", temperature=0)
        self.embeddings = OllamaEmbeddings()
        self.vector_space = None
        self.qa_chain = None
        self.sample_rate = 44100
        self.voice_generator = VoiceGenerator(elevenlabs_api_key)

    def setup_vector_space(self, vector_store):
        self.vector_space = vector_space

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )

        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm = self.llm,
            retriever=self.vector_store.as_retriever(),
            memory=memory,
            verbose=True
        )

    def record_audio(self, duration=5):
        recording = sd.rec(
            int(duration * self.sample_rate), samplerate=self.sample_rate, channels=1
        )
        sd.wait()
        return recording

    def transcribe_audio(self, audio_array):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, audio_array, self.sample_rate)
            result = self.whisper_model.transcribe(temp_audio.name)
            os.unlink(temp_audio.name)
        return result["text"]

    def generate_response(self, query):
        if self.qa_chain is None:
            return "Error: Vector store not initialized"

        response = self.qa_chain.invoke({"question": query})
        return response["answer"]