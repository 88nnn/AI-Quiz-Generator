import streamlit as st
import getpass
import os
from pymongo import MongoClient

st.title("DB 연결 검증")

# Get OpenAI API Key and MongoDB Atlas Cluster URI
#openai_api_key = getpass.getpass("OpenAI API Key:")
os.environ["OPENAI_API_KEY"] #= OPENAI_API_KEY  # Set the API key in the environment
st.write("OpenAI API Key has been set.")

# MongoDB Atlas Cluster URI
mongodb_atlas_cluster_uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
#mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0
#mongodb_atlas_cluster_uri = getpass.getpass("MongoDB Atlas Cluster URI:")
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
