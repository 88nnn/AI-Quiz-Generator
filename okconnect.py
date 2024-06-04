import streamlit as st
import json
from pymongo import MongoClient, UpdateOne
from pymongo.errors import OperationFailure
from langchain_openai import OpenAIEmbeddings

# MongoDB 연결 설정
def connect_db():
    client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    return client["db1"]

# 몽고DB에 문서 삽입
def insert_documents(collection_name, documents):
    if not documents:
        raise ValueError("documents must be a non-empty list")
    
    db = connect_db()
    collection = db[collection_name]
    collection.insert_many(documents)

# 벡터화하고 몽고DB에 저장
def vectorize_and_store(data, collection_name):
    embeddings = OpenAIEmbeddings()
    vector_operations = []
    page_counter = 1

    for document in data:
        if '_id' not in document:
            raise ValueError("Each document must contain an '_id' field")
        
        text = document.get('Problem', '')  # 'Problem' 필드가 없으면 빈 문자열 사용
        vector = embeddings.embedding(text)  # 임베딩 생성 메서드 사용
        
        # 'page' 필드가 없는 경우 자동 생성
        if 'page' not in document:
            document['page'] = page_counter
            page_counter += 1
        
        operation = UpdateOne(
            {'_id': document['_id']},
            {'$set': {
                'embedding': vector.tolist(),
                'page': document['page']
            }},
            upsert=True
        )
        vector_operations.append(operation)

    db = connect_db()
    collection = db[collection_name]
    collection.bulk_write(vector_operations)

# Streamlit UI
st.title("JSON 파일 업로드 및 MongoDB 저장")

upload_option = st.radio("입력 유형을 선택하세요", ("JSON 파일",))

uploaded_file = st.file_uploader("JSON 파일을 업로드하세요.", type=["json"])

if uploaded_file is not None and upload_option == "JSON 파일":
    try:
        # JSON 파일 읽기
        json_content = json.load(uploaded_file)
        
        # json_content가 리스트가 아니면 리스트로 감싸기
        if not isinstance(json_content, list):
            json_content = [json_content]
        
        # 문서 삽입
        insert_documents("study", json_content)
        
        # 벡터화 및 저장
        vectorize_and_store(json_content, "study")

        st.success("JSON 파일이 성공적으로 저장되었습니다.")
    except Exception as e:
        st.error(f"파일 저장 중 오류가 발생했습니다: {e}")
