from langchain_community.document_loaders import (
    TextLoader,
    PyPDFLoader,
    CSVLoader,
    DirectoryLoader,
)

dir_loader = DirectoryLoader("./doc/", glob="**/*.txt")
dir_documents = dir_loader.load()

print("Directiry Text Documents", dir_documents)
