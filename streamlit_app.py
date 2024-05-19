import streamlit as st
from pymongo import MongoClient
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
import numpy as np

# 검색 횟수 세기
if 'search_count' not in st.session_state:
    st.session_state['search_count'] = 0

def search_similar_documents(query, vector_search):
    embeddings = OpenAIEmbeddings().embed_query(query)
    
    # MongoDB aggregation pipeline for vector search
    pipeline = [
        {
            "$vectorSearch": {
                "index": "vector_index",
                "path": "embedding",
                "queryVector": embeddings,
                "numCandidates": 10,
                "limit": 5
            }
        }
    ]
    
    results = list(vector_search.collection.aggregate(pipeline))
    return results

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

    # 검색 버튼
    if st.button("검색하기"):
        st.session_state['search_count'] += 1
        
        if query:
            # MongoDB Atlas Vector Search 인스턴스 생성
            vector_search = MongoDBAtlasVectorSearch(
                collection=mongodb_collection,
                embedding=OpenAIEmbeddings(),  # OpenAIEmbeddings 인스턴스 생성
                index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
            )

            # 쿼리를 임베딩하여 유사한 문서 검색
            results = search_similar_documents(query, vector_search)

            # 검색 결과 출력
            if results:
                st.subheader("검색 결과:")
                for result in results:
                    st.write(result)
            else:
                st.warning("검색 결과가 없습니다.")
        
        st.write(f"검색 횟수: {st.session_state['search_count']}")

if __name__ == "__main__":
    main()
