import os
import streamlit as st
import pickle
import time
import langchain
import faiss
from langchain.llms import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
# from langchain.vectorstores import FaissFlatIndexStore

# initialize variables from .env file
from dotenv import load_dotenv
load_dotenv()

# load the model
llm = OpenAI(temperature=0.9, max_tokens=500)

# load the chain (data)
loaders = UnstructuredURLLoader(urls=[
    "https://www.bbc.co.uk/news/science-environment-67383755",
    "https://www.bbc.co.uk/news/business-67284936",
    "https://www.bbc.co.uk/news/uk-67302048",
])
data = loaders.load()

# split the data into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
docs = text_splitter.split_documents(data)

# create the embeddings
embeddings = OpenAIEmbeddings()

# create the vector store
vector_index = FAISS.from_documents(docs, embeddings)
time.sleep(15)

# Store vector index locally
try:
    file_path = "vector_index.pkl"
    with open(file_path, "wb") as f:
        pickle.dump(vector_index, f)

    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vector_index = pickle.load(f)
except:
    pass

# create retrieval chain
r_chain = RetrievalQAWithSourcesChain(llm=llm, retriever=vector_index.as_retriever())

# create retrieval query
query = "Is AI good for the world?"

r_chain({"question": query}, return_only_outputs=True)
print(r_chain)



