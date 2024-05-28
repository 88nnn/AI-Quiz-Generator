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


def grade_quiz_answer(user_answer, quiz_answer):
    """
    퀴즈 답변을 평가하는 함수

    Args:
        user_answer (str): 사용자 답변
        quiz_answer (str): 정답

    Returns:
        str: 평가 결과
    """
    if user_answer.lower() == quiz_answer.lower():
        grade = "정답"
    else:
        grade = "오답"
    return grade


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
    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()
    else:
        uploaded_file = st.file_uploader(f"{upload_option} 파일을 업로드하세요.", type=[upload_option.lower()])

    quiz_questions = []

    if text_content is not None or uploaded_file is not None:
        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                embeddings = OpenAIEmbeddings()

                # MongoDB 연결 및 설정
                db_name = "db1"
                collection_name = "PythonDatascienceinterview"
                atlas_collection = client[db_name][collection_name]
                vector_search_index = "vector_index"

                retriever = MongoDBAtlasVectorSearch.from_documents(
                    documents=text_content if upload_option != "토픽 선택" else WikipediaLoader(query=topic, load_max_docs=3).load(),
                    embedding=embeddings,
                    collection=atlas_collection,
                    index_name=vector_search_index
                )

                retriever_chain = create_retrieval_chain(retriever)

                for i in range(num_quizzes):
                    quiz_questions.append(
                        generate_quiz(
                            quiz_type,
                            topic if upload_option == "토픽 선택" else None,
                            retriever_chain
                        )
                    )
                st.success('퀴즈 생성이 완료되었습니다!')
                st.write(quiz_questions)
                st.session_state['quiz_created'] = True

    if st.session_state.get('quiz_created', False):
        if st.button('퀴즈 풀기'):
            st.session_state.page = 1


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
