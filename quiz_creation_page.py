#quiz_creation_page.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader

class CreateQuizoub(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

class CreateQuizsub(BaseModel):
    quiz = ("quiz =The created problem")
    correct_answer = ("correct_answer =The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz = ("The created problem")
    options1 = ("The true or false option of the created problem")
    options2 = ("The true or false option of the created problem")
    correct_answer = ("One of the options1 or options2")

def make_model(pages):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    # Rag
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)

    # PydanticOutputParser 생성
    parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
    parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
    parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

    prompt = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN."

        "CONTEXT:"
        "{context}."

        "FORMAT:"
        "{format}"
    )
    promptoub = prompt.partial(format=parseroub.get_format_instructions())
    promptsub = prompt.partial(format=parsersub.get_format_instructions())
    prompttf = prompt.partial(format=parsertf.get_format_instructions())

    document_chainoub = create_stuff_documents_chain(llm, promptoub)
    document_chainsub = create_stuff_documents_chain(llm, promptsub)
    document_chaintf = create_stuff_documents_chain(llm, prompttf)

    retriever = vector.as_retriever()

    retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
    retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
    retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

    # chainoub = promptoub | chat_model | parseroub
    # chainsub = promptsub | chat_model | parsersub
    # chaintf = prompttf | chat_model | parsertf
    return 0

@st.cache_data
def process_file(uploaded_file, text_area_content, url_area_content):
    text_content = None

    if uploaded_file is not None:
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
    elif text_area_content:
        text_content = text_area_content
    elif url_area_content:
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()

    if text_content:
        documents = [{"page_content": text_content}]
        text_splitter = RecursiveCharacterTextSplitter()
        documents = text_splitter.split_documents(documents)
        return documents
    else:
        st.warning("파일, 텍스트 또는 URL을 입력하세요.")
        return None

    return text_content


# 파일 처리 함수
def process_file(uploaded_file):

    uploaded_file = None
    text_area_content = None
    url_area_content = None
    selected_topic = None
    
    # 파일 업로드 옵션 선택
    upload_option = st.radio("입력 유형을 선택하세요", ("이미지 파일", "PDF 파일", "직접 입력", "URL", "토픽 선택"))

    # 선택된 옵션에 따라 입력 방식 제공
    if upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    else:
        uploaded_file = None

    # 텍스트 입력 영역
    if upload_option == "직접 입력":
        text_area_content = st.text_area("텍스트를 입력하세요.")
    else:
        text_area_content = None

    # URL 입력 영역
    if upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
    else:
        url_area_content = None

    # 토픽 선택 영역
    if upload_option == "토픽 선택":
        selected_topic = "수학"
        selected_topic = st.selectbox(
            "토픽을 선택하세요.",
            ("토픽 선택", "수학", "물리학", "역사", "화학"))
    else:
        url_area_content = None
    
    if uploaded_file is None:
        if url_area_content is None:
            if selected_topic == "토픽 선택":
                if text_area_content is None:
                    st.warning("입력이 필요합니다.")
                    return None

    # 업로드된 파일 처리
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    if uploaded_file.type.startswith("image/"):
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
        
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    if text_area_content is not None:
        text_content = process_file(uploaded_file, text_area_content)
    texts = text_splitter.create_documents([text_content])
    return texts

    return texts

# 퀴즈 생성 함수
@st.experimental_fragment
def generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub, retrieval_chaintf):
    # Generate quiz prompt based on selected quiz type
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chainoub.invoke(
            {
                "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "주관식":
        response = retrieval_chainsub.invoke(
            {
                "input": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = retrieval_chaintf.invoke(
            {
                "input": "Create one true or false question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    quiz_questions = response

    return quiz_questions

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    if user_answer.lower() == quiz_answer.lower():
        grade = "정답"
    else:
        grade = "오답"
    return grade

# 메인 함수
def quiz_creation_page():
    placeholder = st.empty()
    st.session_state.page = 0
    if st.session_state.page == 0:
        with placeholder.container():
            st.title("AI 퀴즈 생성기")
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = ""

            # 퀴즈 유형 선택
            quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

            # 퀴즈 개수 선택
            num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

            # 파일 업로드 옵션
            st.header("파일 업로드")
            uploaded_file = None
            #uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])
            text_content = process_file(uploaded_file)

            quiz_questions = []

            if text_content is not None:

                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                        embeddings = OpenAIEmbeddings()

                        # Rag
                        text_splitter = RecursiveCharacterTextSplitter()
                        documents = text_splitter.split_documents(text_content)
                        vector = FAISS.from_documents(documents, embeddings)

                        # PydanticOutputParser 생성
                        parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
                        parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
                        parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

                        prompt = PromptTemplate.from_template(
                            "{input}, Please answer in KOREAN."

                            "CONTEXT:"
                            "{context}."

                            "FORMAT:"
                            "{format}"
                        )
                        promptoub = prompt.partial(format=parseroub.get_format_instructions())
                        promptsub = prompt.partial(format=parsersub.get_format_instructions())
                        prompttf = prompt.partial(format=parsertf.get_format_instructions())

                        document_chainoub = create_stuff_documents_chain(llm, promptoub)
                        document_chainsub = create_stuff_documents_chain(llm, promptsub)
                        document_chaintf = create_stuff_documents_chain(llm, prompttf)

                        retriever = vector.as_retriever()

                        retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
                        retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
                        retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

                        for i in range(num_quizzes):
                            quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub,retrieval_chaintf))
                            st.session_state['quizs'] = quiz_questions
                        st.session_state.selected_page = "퀴즈 풀이"
                        st.session_state.selected_type = quiz_type
                        st.session_state.selected_num = num_quizzes

                        st.success('퀴즈 생성이 완료되었습니다!')
                        st.write(quiz_questions)
                        st.session_state['quiz_created'] = True

                if st.session_state.get('quiz_created', False):
                    if st.button('퀴즈 풀기'):
                        st.switch_page("pages/quiz_solve_page.py")


if __name__ == "__main__":
    quiz_creation_page()
