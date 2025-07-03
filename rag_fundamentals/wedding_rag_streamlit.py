import streamlit as st
import csv
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import os
from dotenv import load_dotenv
import uuid

# Suppress tokenizer warnings
os.environ["TOKENIZER_PARALLELISM"] = "false"
load_dotenv()

CSV_URL = os.getenv("CSV_URL")

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
        else:
            raise ValueError(f"Unsupported embedding model type: {model_type}")

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

class WeddingTableArranger:
    def __init__(self, sheet_url):
        self.sheet_url = sheet_url

    def _get_sheet_data(self):
        """
        Reads the Google Sheet content from the provided URL.
        """
        try:
            df = pd.read_csv(self.sheet_url)
            return df.to_dict(orient='records')
        except Exception as e:
            print(f"Error reading Google Sheet: {e}")
            return []

    def arrange_table_numbers(self):
        """
        Reads the spreadsheet content, adds a 'table_num' column,
        and assigns table numbers based on attendance and relationship.
        """
        csv_json = self._get_sheet_data()
        if not csv_json:
            return []

        for row in csv_json:
            contact_number = row.get("Your contact number")
            can_attend = row.get("Can you attend the wedding?")
            relationship = row.get("What is your relationship with the newcomer?")

            # Call method2 to get the table number
            table_num = self._assign_table_number(can_attend, relationship)
            row["table_num"] = table_num

        self._generate_csv(csv_json)

        return csv_json

    def _generate_csv(self, csv_json):
        try:
            with open("wedding.csv", mode="w", newline="", encoding="utf-8") as file:
                writer = csv.DictWriter(file, fieldnames=[
                    "時間戳記", "Your name", "Your contact number", "Your Email",
                    "What is your relationship with the newcomer?", "Can you attend the wedding?", "table_num"
                ])
                writer.writeheader()
                writer.writerows(csv_json)

            print("CSV file 'wedding.csv' created successfully!")
        except Exception as e:
            print(f"Error creating CSV file 'wedding.csv': {e}")

    def _assign_table_number(self, can_attend, relationship):
        """
        Assigns a table number based on attendance and relationship.
        """
        if can_attend == "I want to participate! Witness the happy moment together":
            if relationship == "Groom's relatives":
                table_num = 1
            elif relationship == "Groom's friend":
                table_num = 2
            elif relationship == "Groom's colleague":
                table_num = 3
            elif relationship == "Bride's relatives":
                table_num = 4
            elif relationship == "Bride's friend":
                table_num = 5
            elif relationship == "Bride's colleague":
                table_num = 6
            else:
                table_num = "Not attending"
        else:
            table_num = "Not attending"
        return table_num

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

def setup_chromadb(documents, embedding_model):
    client = chromadb.PersistentClient(path="./chroma_db")

    collection_name = f"wedding_facts_{embedding_model.model_type}" # 用 embedding model type 來當作 collection name

    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_model.embedding_fn
    )

    existing_count = collection.count() # 取得目前已存在的資料筆數（避免重複加入）

    new_documents = documents[existing_count:]
    new_ids = [str(uuid.uuid4()) for _ in new_documents]

    if new_documents:
        collection.add(documents=new_documents, ids=new_ids)
        print("\nDocuments add to ChromaDB collection successfully!")
    else:
        print("No new documents to add.")

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
    st.set_page_config(page_title="Wedding Arrange TABLE RAG", layout="wide")
    st.title("Wedding Arrange TABLE RAG System")

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

    # Always fetch latest facts
    arranger = WeddingTableArranger(CSV_URL)
    new_facts = arranger.arrange_table_numbers()

    # Save to session
    if "facts" not in st.session_state:
        st.session_state.facts = new_facts
    elif len(new_facts) > len(st.session_state.facts):
        st.session_state.facts = new_facts

    # Generate documents
    documents = [
        f"{f['Your name']} ({f['Your contact number']}, {f['Your Email']}) is a {f['What is your relationship with the newcomer?']} and says: "
        f"\"{f['Can you attend the wedding?']}\" - Assigned to table {f['table_num']}"
        for f in st.session_state.facts
    ]

    # Initialize embedding and LLM models if not present or changed
    if (
        "llm_model" not in st.session_state
        or st.session_state.llm_model.model_name != llm_type
    ):
        st.session_state.llm_model = LLMModel(llm_type)

    if (
        "embedding_model" not in st.session_state
        or st.session_state.embedding_model.model_type != embedding_type
    ):
        st.session_state.embedding_model = EmbeddingModel(embedding_type)

    # Setup or incrementally update ChromaDB
    st.session_state.collection = setup_chromadb(documents, st.session_state.embedding_model)

    # Query Input
    query = st.text_input(
        "Enter your question about wedding table number:",
        placeholder="e.g., What is edward table number?",
    )

    if query:
        with st.spinner("Processing your query..."):
            response, references, augmented_prompt = rag_pipeline(
                query, st.session_state.collection, st.session_state.llm_model
            )

            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### Response")
                st.write(response)

            # with col2:
            #     st.markdown("## References Used")
            #     for ref in references:
            #         st.write(f"- {ref}")

            with st.expander("Technical Details", expanded=False):
                st.markdown("#### Augmented Prompt")
                st.code(augmented_prompt)

                st.markdown("#### Model Configuration")
                st.write(f"- LLM Model: {llm_type.upper()}")
                st.write(f"- Embedding Model: {embedding_type.upper()}")

if __name__ == "__main__":
    streamlit_app()