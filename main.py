from langchain.document_loaders import UnstructuredURLLoader, TextLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter

from sentence_transformers import SentenceTransformer

import faiss
import numpy as np

# loader = UnstructuredURLLoader(urls=["https://www.bbc.co.uk/news/live/uk-67390343",
                                    #  "https://www.bbc.co.uk/news/live/uk-67418363",])
loader = TextLoader(file_path="ex_text.txt")
data = loader.load()
# print(len(data))
# print(data[0])

splitter = RecursiveCharacterTextSplitter(
    separators=['\n', '\n', '.', '?', '!'],
    chunk_size=500,
    chunk_overlap=100
)
chunks = splitter.split_text(data[0].page_content)
# print(chunks)

sentence = "Rishi Sunak says the government is working on a new treaty with Rwanda, after the government's asylum seeker plan was ruled unlawful"

encoder = SentenceTransformer("all-mpnet-base-v2")
vectors = encoder.encode(sentence)
# print(vectors.shape)

vec_arr = np.array([vectors]).reshape(1, -1)
# print(vec_arr.shape)
dim = vec_arr.shape[1]

index = faiss.IndexFlatL2(dim)
index.add(vec_arr)

search_query = "How many hours per week watched?"
query_vector = encoder.encode(search_query)
# print(query_vector.shape)

q_arr = np.array([query_vector]).reshape(1, -1)

search = index.search(q_arr, 2)
print(search)