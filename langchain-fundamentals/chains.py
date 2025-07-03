from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# Define a prompt template
prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")

# Create a chat model
model = ChatOllama(model="llama3")

# Chain the prompt, model and output parser
chain = prompt | model | StrOutputParser()

# Run the chain
response = chain.invoke({"topic": "bears"})
print(response)

