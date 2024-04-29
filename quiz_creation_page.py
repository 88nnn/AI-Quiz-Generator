#quiz_creation_page.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io

chat_model = ChatOpenAI(model="gpt-3.5-turbo-0125")


class CreateQuizoub(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    options1: str = Field(description="만들어진 문제의 첫 번째 보기")
    options2: str = Field(description="만들어진 문제의 두 번째 보기")
    options3: str = Field(description="만들어진 문제의 세 번째 보기")
    options4: str = Field(description="만들어진 문제의 네 번째 보기")
    correct_answer: str = Field(description="만들어진 문제의 보기중 하나")


class CreateQuizsub(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    correct_answer: str = Field(description="만들어진 문제의 답")


class CreateQuizTF(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    options1: str = Field(description="만들어진 문제의 참 또는 거짓인 보기")
    options2: str = Field(description="만들어진 문제의 참 또는 거짓인 보기")
    correct_answer: str = Field(description="만들어진 보기중 하나")


# PydanticOutputParser 생성
parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

prompt = PromptTemplate.from_template(
    "{instruction}, Please answer in KOREAN."

    "CONTEXT:"
    "{input}."

    "FORMAT:"
    "{format}"
)
promptoub = prompt.partial(format=parseroub.get_format_instructions())
promptsub = prompt.partial(format=parsersub.get_format_instructions())
prompttf = prompt.partial(format=parsertf.get_format_instructions())

chainoub = promptoub | chat_model | parseroub
chainsub = promptsub | chat_model | parsersub
chaintf = prompttf | chat_model | parsertf


# 퀴즈 채점 함수
@st.experimental_fragment
def grade_quiz_answers(user_answers, quiz_answers):
    graded_answers = []
    for user_answer, quiz_answer in zip(user_answers, quiz_answers):
        if user_answer.lower() == quiz_answer.lower():
            graded_answers.append("정답")
        else:
            graded_answers.append("오답")
    st.session_state['ganswer'] = graded_answers
    return graded_answers


# 파일 처리 함수
def process_file(uploaded_file):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    # 업로드된 파일 처리
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

    return text_content


# 퀴즈 생성 함수
@st.experimental_fragment
def generate_quiz(quiz_type, text_content):
    # Generate quiz prompt based on selected quiz type
    if quiz_type == "다중 선택 (객관식)":
        response = chainoub.invoke(
            {
                "instruction": "다음 글을 이용해 객관식 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
            }
        )
    elif quiz_type == "주관식":
        response = chainsub.invoke(
            {
                "instruction": "다음 글을 이용해 주관식 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = chaintf.invoke(
            {
                "instruction": "다음 글을 이용해 참과 거짓, 2개의 보기를 가지는 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
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
            st.write(st.session_state.selected_page)

            # 퀴즈 유형 선택
            quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

            # 퀴즈 개수 선택
            num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

            # 파일 업로드 옵션
            st.header("파일 업로드")
            uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

            text_content = process_file(uploaded_file)

            quiz_questions = []
            # if 'gene' not in st.session_state:
            #     st.session_state.gene = None

            if text_content is not None:

                if st.button('문제 생성 하기'):
                    for i in range(num_quizzes):
                        quiz_questions.append(generate_quiz(quiz_type, text_content))
                        st.session_state['quizs'] = quiz_questions
                    st.session_state.selected_page = "퀴즈 풀이"
                    st.session_state.selected_type = quiz_type
                    st.session_state.selected_num = num_quizzes
                    # st.session_state.gene = 1
            # if st.session_state.gene is not None:
            #     st.rerun()
