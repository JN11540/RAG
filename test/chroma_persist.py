import chromadb

from chromadb.utils import embedding_functions

default_df = embedding_functions.DefaultEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="./db/chroma_persist")

collection = chroma_client.get_or_create_collection(
    "my_story", embedding_function=default_df
)

documents = [
    {"id": "doc1", "text": "Hello, world!"},
    {"id": "doc2", "text": "How are you today?"},
    {"id": "doc3", "text": "Goodbye, see you later"},
    {"id": "doc4", "text": "That's good to see you"},
]

for doc in documents:
    collection.upsert(ids=doc["id"], documents=[doc["text"]])

# define a query text
query_text = "Hello, world!"

results = collection.query(
    query_texts=[query_text],
    n_results=2
)