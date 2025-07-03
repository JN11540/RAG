import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv

# Suppress tokenizer warnings
os.environ["TOKENIZER_PARALLELISM"] = "false"
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class EmbeddingModel:
    def __init__(self, model_type="chroma"):
        self.model_type = model_type
        if model_type == "chroma":
            self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        elif model_type == "nomic":
            # using ollama nomic-embed-text model
            self.embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
                api_key="ollama",
                api_base="http://localhost:11434/v1",
                model_name="nomic-embed-text"
            )

class LLMModel:
    def __init__(self, model_type="gemma"):
        self.model_type = model_type
        if model_type == "gemma":
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            self.model_name = "gemma"
        else:
            self.client = OpenAI(
                base_url="http://localhost:11434/v1",
                api_key="ollama"
            )
            self.model_name = "llama3"

    def generate_completion(self, messages):
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.7
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error generating response: {e}"

def select_models():
    # select LLM model
    print("\nSelect LLM Model:")
    print("1. OpenAI GPT-4")
    print("2. Ollama Llama2")
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice in ["1", "2"]:
            llm_type = "openai" if choice == "1" else "ollama"
            break
        print("Please enter either 1 or 2")

    # select Embedding model
    print("\nSelect Embedding Model:")
    print("1. OpenAI Embedding")
    print("2. Chroma Default")
    print("3. Nomic Embed Text (Ollama)")
    while True:
        choice = input("Enter choice (1 or 2 or 3): ").strip()
        if choice in ["1", "2", "3"]:
            embedding_type = {"1": "openai", "2": "chroma", "3": "nomic"}[choice]
            break
        print("Please enter either 1 or 2 or 3")

    return llm_type, embedding_type

def generate_csv():
    facts = [
        {"id": 1, "fact": "Yuri Gagarin was the first human to orbit Earth in 1961."},
        {"id": 2, "fact": "Apollo 11 was the first mission to land humans on the Moon in 1969."},
        {"id": 3, "fact": "The Hubble Space Telescope has captured stunning images of deep space since its launch in 1990."},
        {"id": 4, "fact": "Mars is often called the Red Planet due to its iron-rich, dusty surface."},
        {"id": 5, "fact": "The International Space Station (ISS) is a habitable satellite orbiting Earth since 1998."},
        {"id": 6, "fact": "Voyager 1 is the farthest human-made object from Earth, now traveling through interstellar space."},
        {"id": 7, "fact": "SpaceX is a private aerospace company revolutionizing space travel with reusable rockets."},
        {"id": 8, "fact": "The James Webb Space Telescope observes the universe in infrared to study its earliest galaxies."},
        {"id": 9, "fact": "Black holes are regions of spacetime with gravity so strong that nothing, not even light, can escape."},
    ]

    with open("space_facts.csv", mode="w", newline="") as file:
        writter = csv.DictWriter(file, fieldnames=["id", "fact"])
        writter.writeheader()
        writter.writerows(facts)

    print("CSV file 'space_facts'.csv' created successfully!")
    return facts

def load_csv():
    df = pd.read_csv("space_facts.csv")
    documents = df["fact"].tolist()
    print("\nLoaded documents")
    for doc in documents:
        print(f"- {doc}")
    return documents

def setup_chromadb(documents, embedding_model):
    client = chromadb.Client()

    try:
        client.delete_collection("space_facts")
    except Exception as e:
        pass

    collection = client.create_collection(
        name="space_facts", embedding_function=embedding_model.embedding_fn
    )

    collection.add(documents=documents, ids=[str(i) for i in range(len(documents))])

    print("\nDocuments add to ChromaDB collection successfully!")
    return collection

def find_related_chunks(query, collection, top_k=2):
    results = collection.query(query_texts=[query], n_results=top_k)

    print("\nRelated chunks found:")
    for doc in results["documents"][0]:
        print(f"- {doc}")

    return list(
        zip(
            results["documents"][0],
            (
                results["metadatas"][0]
                if results["metadatas"][0]
                else [{}] * len(results["documents"][0])
            ),
        )
    )

def augment_prompt(query, related_chunks):
    context = "\n".join(chunk[0] for chunk in related_chunks)
    augmented_prompt = f"Context\n{context}\n\nQuestion: {query}\nAnswer:"

    print("\nAugmented prompt: ")
    print(augmented_prompt)

    return augmented_prompt

def rag_pipeline(query, collection, llm_model, top_k=2):
    print(f"\nProcessing query: {query}")

    related_chunks = find_related_chunks(query, collection, top_k)
    augmented_prompt = augment_prompt(query, related_chunks)

    response = llm_model.generate_completion(
        [
            {
                "role": "system",
                "content": "You are a helpful assistant who can answer questions about space but only answers questions that are directly related to the sources/documents given.",
            },
            {
                "role": "user","content": augmented_prompt
            },
        ]
    )

    print("\nGenerated response: ")
    print(response)

    references = [chunk[0] for chunk in related_chunks]
    return response, references, augmented_prompt

def streamlit_app():
    st.set_page_config(page_title="Space Facts RAG", layout="wide")
    st.title("Space Facts RAG System")

    # Sidebar for model selection
    st.sidebar.title("Model Configuration")

    llm_type = st.sidebar.radio(
        "Select LLM Model:",
        ["gemma", "ollama"],
        format_func=lambda x: "Ollama Gemma" if x == "gemma" else "Ollama Llama3",
    )

    embedding_type = st.sidebar.radio(
        "Select Embedding Model:",
        ["chroma", "nomic"],
        format_func=lambda x: {
            "chroma": "Chroma Default",
            "nomic": "Nomic Embed Text (Ollama)",
        }[x],
    )

    # Initialize session state
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
        st.session_state.facts = generate_csv()

        # Initialize models
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

        # Setup ChromaDB
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )
        st.session_state.initialized = True

    # Reinitialize models if selections changed
    if (
        st.session_state.llm_model.model_name != llm_type
        or st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)
        st.session_state.embedding_model = EmbeddingModel(embedding_type)
        documents = [fact["fact"] for fact in st.session_state.facts]
        st.session_state.collection = setup_chromadb(
            documents, st.session_state.embedding_model
        )

    # Display available facts
    with st.expander("Available Space Facts", expanded=False):
        for fact in st.session_state.facts:
            st.write(f"- {fact['fact']}")

    # Query Input
    query = st.text_input(
        "Enter your question about space:",
        placeholder="e.g., What is the Hubble Space Telescope?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            # Display results in columns
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### Response")
                st.write(response)

            with col2:
                st.markdown("## References Used")
                for ref in references:
                    st.write(f"- {ref}")

            # Show technical details in expander
            with st.expander("Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")

if __name__ == "__main__":
    streamlit_app()