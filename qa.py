import streamlit as st
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings

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
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # Access the MongoDB collection
    mongodb_collection = client[DB_NAME][COLLECTION_NAME]

    st.write(f"Connected to MongoDB database: {DB_NAME}, collection: {COLLECTION_NAME}")

    # 사용자가 검색할 쿼리 입력
    query = st.text_input("유사한 파일을 검색할 쿼리를 입력하세요:")

    if query:
        # MongoDB Atlas Vector Search 인스턴스 생성
        vector_search = MongoDBAtlasVectorSearch.from_connection_string(
            mongodb_atlas_cluster_uri,
            f"{DB_NAME}.{COLLECTION_NAME}",
            OpenAIEmbeddings(disallowed_special=()),  # OpenAI API Key는 사용하지 않습니다.
            index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
        )

        # 유사한 문서 검색
        results = search_similar_documents(query, vector_search)

        # 검색 결과 출력
        if results:
            st.subheader("검색 결과:")
            for result in results:
                st.write(result)
        else:
            st.warning("검색 결과가 없습니다.")

if __name__ == "__main__":
    main()
