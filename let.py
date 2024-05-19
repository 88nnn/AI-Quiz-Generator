import streamlit as st
from pymongo import MongoClient

def main():
    st.title("DB 연결 검증")

    # MongoDB Atlas Cluster URI
    mongodb_atlas_cluster_uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"
    st.write("MongoDB Atlas Cluster URI has been set.")

    # Initialize MongoDB python client
    client = MongoClient(mongodb_atlas_cluster_uri)

    # Define database and collection names
    DB_NAME = "sample_mflix"
    COLLECTION_NAME = "embedded_movies"

    # Access the MongoDB collection
    mongodb_collection = client[DB_NAME][COLLECTION_NAME]

    st.write(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")

    # 벡터 저장소 콜렉션에서 처음 5개 문서 가져오기
    documents = mongodb_collection.find().limit(5)

    # 가져온 문서 출력
    st.subheader("벡터 저장소 콜렉션의 처음 5개 문서:")
    for document in documents:
        st.write(document)

if __name__ == "__main__":
    main()
