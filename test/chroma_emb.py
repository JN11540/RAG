from chromadb.utils import embedding_functions

default_df = embedding_functions.DefaultEmbeddingFunction()

name = "Paulo"

emb = default_df(name)

print(emb)