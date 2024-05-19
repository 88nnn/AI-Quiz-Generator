import streamlit as st
import getpass
import os
import io
from pymongo import MongoClient
from PyPDF2 import PdfReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

def main():
    st.title("DB 연결 검증")

    # Get OpenAI API Key
    os.environ["OPENAI_API_KEY"]
    st.write("OpenAI API Key has been set.")

    # MongoDB Atlas Cluster URI
    mongodb_atlas_cluster_uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    st.write("MongoDB Atlas Cluster URI has been set.")

    # Initialize MongoDB python client
    client = MongoClient(mongodb_atlas_cluster_uri)

    # Define database and collection names
    DB_NAME = "sample_mflix"
    COLLECTION_NAME = "embedded_movies"
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # Access the MongoDB collection
    mongodb_collection = client[DB_NAME][COLLECTION_NAME]

    st.write(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")

    # Load the PDF
    uploaded_file = st.file_uploader("Upload PDF", type="pdf")
    if uploaded_file is not None:
        data = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in data.pages:
            text_content += page.extract_text()

        if text_content:
            documents = [{"page_content": text_content}]
            st.write(documents)
        else:
            st.warning("파일에 텍스트가 없습니다.")
          text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
      docs = text_splitter.split_documents(data)
      print(docs[0])

# insert the documents in MongoDB Atlas with their embedding

    vector_search = MongoDBAtlasVectorSearch.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection=MONGODB_COLLECTION,
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

# Perform a similarity search between the embedding of the query and the embeddings of the documents

     results = vector_search.similarity_search(data=data, k=5, pre_filter={"page": {"$eq": 1}})
     for result in results:
       st.write(result)
       results = vector_search.similarity_search_with_score(
         data=data, k=5,)
       for result in results:
         st.write(result)
       print(results[0].page_content)
       vector_search = MongoDBAtlasVectorSearch.from_connection_string(
         MONGODB_ATLAS_CLUSTER_URI,
         DB_NAME + "." + COLLECTION_NAME,
         OpenAIEmbeddings(disallowed_special=()),
         index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
       )


if __name__ == "__main__":
   main()디버깅해줘
