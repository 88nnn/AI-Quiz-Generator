import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.output_parsers import PydanticOutputParser
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from pymongo import MongoClient

# MongoDB 연결
client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")


def process_file(uploaded_file, upload_option):
    """
    업로드된 파일을 처리하여 텍스트로 변환하는 함수

    Args:
        uploaded_file (file object): 업로드된 파일 객체
        upload_option (str): 업로드된 파일의 유형

    Returns:
        str: 변환된 텍스트 내용
    """
    # 파일 유형에 따라 처리
    if uploaded_file.type == "text/plain":
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None

    # 텍스트 분리 및 처리
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text_content])

    return texts


def generate_quiz(quiz_type, is_topic, retriever_chain):
    """
    퀴즈를 생성하는 함수

    Args:
        quiz_type (str): 퀴즈 유형
        is_topic (str): 선택된 토픽
        retriever_chain: 텍스트 검색 및 퀴즈 생성을 위한 검색 체인

    Returns:
        dict: 생성된 퀴즈
    """
    # 토픽에 따라 퀴즈 생성
    if is_topic is None:
        input_text = f"Create one {quiz_type} question focusing on important concepts, following the given format, referring to the following context"
    else:
        input_text = f"Create one {is_topic} {quiz_type} question focusing on important concepts, following the given format, referring to the following context"

    response = retriever_chain.invoke({"input": input_text})
    quiz_questions = response

    return quiz_questions


def quiz_creation_page():
    """
    퀴즈 생성 페이지를 렌더링하는 함수
    """
    st.title("AI 퀴즈 생성기")
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

    st.header("파일 업로드")
    uploaded_file = None
    text_content = None
    topic = None

    if upload_option == "토픽 선택":
        topic = st.selectbox(
           "토픽을 선택하세요",
           ("수학", "문학", "비문학", "과학", "test", "langchain", "vector_index"),
           index=None,
           placeholder="토픽을 선택하세요",
        )
        

        if st.button('토픽에 따른 벡터 검색'):
            # MongoDB 연결 및 설정
            db_name = "db1"
            collection_name = "PythonDatascienceinterview"
            atlas_collection = client[db_name][collection_name]
            vector_search_index = "vector_index"

            # 벡터 검색기 생성
            embeddings = OpenAIEmbeddings()
            retriever = MongoDBAtlasVectorSearch.from_connection_string(
                "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
                collection_name,
                embeddings,
                vector_search_index
            )

            # 토픽에 따른 벡터 검색 결과 출력
            docs = WikipediaLoader(query=topic, load_max_docs=3).load()
            text_splitter = RecursiveCharacterTextSplitter()
            documents = text_splitter.split_documents(docs)
            vector_search = MongoDBAtlasVectorSearch.from_documents(
                documents=documents,
                embedding=embeddings,
                collection=atlas_collection,
                index_name=vector_search_index
            )
            st.write(vector_search.search_results())

    if st.button('퀴즈 생성'):
        # MongoDB 연결 및 설정
        db_name = "db1"
        collection_name = "PythonDatascienceinterview"
        atlas_collection = client[db_name][collection_name]

        # 토픽을 임베딩합니다.
        topic_embedding = embeddings.embed_text(topic)

        # MongoDB에서 벡터 검색을 수행합니다.
        results = search_vectors(collection_name, topic_embedding)

        quiz_questions = []
        for doc in results:
            quiz_questions.append({
                "quiz": doc["quiz"],
                "options1": doc["options1"],
                "options2": doc["options2"],
                "options3": doc["options3"],
                "options4": doc["options4"],
                "correct_answer": doc["correct_answer"]
            })

        st.success('퀴즈 생성이 완료되었습니다!')
        st.write(quiz_questions)
        st.session_state['quiz_created'] = True

    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()

    else:
        uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])
        if uploaded_file:
            text_content = process_file(uploaded_file, upload_option)

    if text_content is not None or (topic is not None and upload_option == "토픽 선택"):
        if st.button('퀴즈 생성'):
            # MongoDB 연결 및 설정
            db_name = "db1"
            collection_name = "PythonDatascienceinterview"
            atlas_collection = client[db_name][collection_name]
            vector_search_index = "vector_index"

            # 벡터 검색기 생성
            embeddings = OpenAIEmbeddings()
            retriever = MongoDBAtlasVectorSearch.from_connection_string(
                client,
                collection_name,
                embeddings,
                vector_search_index
            )

            # 텍스트 검색 및 퀴즈 생성 체인 설정
            retriever_chain = create_retrieval_chain(retriever)

            # 토픽 선택 여부에 따라 퀴즈 생성
            is_topic = topic if upload_option == "토픽 선택" else None

            quiz_questions = []
            for _ in range(num_quizzes):
                quiz_questions.append(
                    generate_quiz(
                        quiz_type,
                        is_topic,
                        retriever_chain
                    )
                )

            st.success('퀴즈 생성이 완료되었습니다!')
            st.write(quiz_questions)
            st.session_state['quiz_created'] = True

    elif topic is not None:
        st.warning("토픽을 선택하고 '토픽에 따른 벡터 검색' 버튼을 눌러주세요.")

def search_vectors(collection_name, query_vector, top_k=10):
    """
    MongoDB에서 벡터 검색을 수행하는 함수
    """
    db = connect_db()
    collection = db[collection_name]
    results = collection.aggregate([
        {
            '$search': {
                'vector': {
                    'query': query_vector,
                    'path': 'vector',
                    'cosineSimilarity': True,
                    'topK': top_k
                }
            }
        }
    ])

    return list(results)


def quiz_page():
    """
    퀴즈 풀이 페이지를 렌더링하는 함수
    """
    st.title("AI 퀴즈 생성기")
    num_quizzes = st.session_state.quiz_questions
    st.write(f"퀴즈 개수: {len(num_quizzes)}")
    for i, quiz in enumerate(num_quizzes):
        st.write(f"문제 {i + 1}: {quiz['quiz']}")
        if quiz_type == "다중 선택 (객관식)":
            st.write(f"보기 1: {quiz['options1']}")
            st.write(f"보기 2: {quiz['options2']}")
            st.write(f"보기 3: {quiz['options3']}")
            st.write(f"보기 4: {quiz['options4']}")
        st.write(f"정답: {quiz['correct_answer']}")
        st.write("=" * 50)


def main():
    """
    메인 애플리케이션 함수
    """
    st.title("AI 퀴즈 생성기")
    page = st.sidebar.selectbox(
        "페이지를 선택하세요:",
        ["퀴즈 생성", "퀴즈 풀이"]
    )

    if page == "퀴즈 생성":
        quiz_creation_page()
    elif page == "퀴즈 풀이":
        quiz_page()


if __name__ == "__main__":
    main()
