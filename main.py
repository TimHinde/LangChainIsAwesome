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
# load_dotenv()

# # load the model
# llm = OpenAI(temperature=0.9, max_tokens=500)

# # load the chain (data)
# loaders = UnstructuredURLLoader(urls=[
#     "https://www.bbc.co.uk/news/science-environment-67383755",
#     "https://www.bbc.co.uk/news/business-67284936",
#     "https://www.bbc.co.uk/news/uk-67302048",
# ])
# data = loaders.load()

# # split the data into chunks
# text_splitter = RecursiveCharacterTextSplitter(
#     chunk_size=500,
#     chunk_overlap=100,
# )
# docs = text_splitter.split_documents(data)

# # create the embeddings
# embeddings = OpenAIEmbeddings()

# # create the vector store
# vector_index = FAISS.from_documents(docs, embeddings)
# time.sleep(15)

# # Store vector index locally
# try:
#     file_path = "vector_index.pkl"
#     with open(file_path, "wb") as f:
#         pickle.dump(vector_index, f)

#     if os.path.exists(file_path):
#         with open(file_path, "rb") as f:
#             vector_index = pickle.load(f)
# except:
#     pass

# # create retrieval chain
# r_chain = RetrievalQAWithSourcesChain(llm=llm, retriever=vector_index.as_retriever())

# # create retrieval query
# query = "Is AI good for the world?"

# r_chain({"question": query}, return_only_outputs=True)
# print(r_chain)

import streamlit as st

load_dotenv()

st.title("Website Summary")
st.sidebar.title("Website URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i + 1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Summarise")

# set up for loading bar
main_placeholder = st.empty()
# set up llm
llm = OpenAI(temperature=0.9, max_tokens=500)

# Set up button actiom
if process_url_clicked:
    # show loading bar
    main_placeholder.text("Loading...")
    main_placeholder.progress(0.1)
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()
    # update progress bar
    main_placeholder.progress(0.4)
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '. '],
        chunk_size=1000,
        chunk_overlap=100
    )
    docs = text_splitter.split_documents(data)
    # update progress bar
    main_placeholder.progress(0.6)
    # create embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    # create FAISS
    vecorstore_openai = FAISS.from_documents(docs, embeddings)
    # using FAISS save_local to save db
    vecorstore_openai.save_local("vector_index.db")
    # keeping this in memory as pickle not working.
    # update progress bar
    main_placeholder.progress(1.0)

# set main page to query options 
query = main_placeholder.text_input("Enter your question: ")

if query:
    if os.path.exists("vector_index.db"):
        # load vector store with FAISS
        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        vectorstore = FAISS.load_local("vector_index.db", embeddings)
        # set up llm
        llm = OpenAI(temperature=0.9, max_tokens=500, model="gpt-3.5-turbo-instruct")
        # setup langchain
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        response = chain({"question": query}, return_only_outputs=True)
        # parse response dict and rpesent to user
        if isinstance(response, dict):
            if "answer" in response:
                st.header("Answer: ")
                st.write(response["answer"])
            if "context" in response:
                st.subheader("Context: ")
                st.write(response["context"])
            if "sources" in response:
                sources = response["sources"]
                st.subheader("Sources: ")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)

