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


class CreateQuiz(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    options: str = Field(description="만들어진 문제의 보기")
    correct_answer: str = Field(description="만들어진 문제의 답")

class CreateQuizsub(BaseModel):
    quiz: str = Field(description="만들어진 문제")
    correct_answer: str = Field(description="만들어진 문제의 답")


parser = PydanticOutputParser(pydantic_object=CreateQuiz)
parser2 = PydanticOutputParser(pydantic_object=CreateQuizsub)


prompt = PromptTemplate.from_template(
    "{instruction}, Please answer in KOREAN."

    "CONTEXT:"
    "{input}."

    "FORMAT:"
    "{format}"
)
prompt2 = prompt.partial(format=parser2.get_format_instructions())
prompt1 = prompt.partial(format=parser.get_format_instructions())

chain = prompt1 | chat_model | parser
chain2 = prompt2 | chat_model | parser2

@st.experimental_fragment
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

@st.experimental_fragment
def generate_quiz(quiz_type, text_content):
    # Generate quiz prompt based on selected quiz type
    if quiz_type == "다중 선택 (객관식)":
        response = chain.invoke(
            {
                "instruction": "다음 글을 이용해 객관식 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
            }
        )
    elif quiz_type == "주관식":
        response = chain2.invoke(
            {
                "instruction": "다음 글을 이용해 주관식 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = chain.invoke(
            {
                "instruction": "다음 글을 이용해 참과 거짓, 2개의 보기를 가지는 퀴즈를 1개 만들어 주세요",
                "input": str({text_content}),
            }
        )
    quiz_questions = response

    return quiz_questions

def quiz_creation_page():
    # 퀴즈 유형 선택
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

    # 퀴즈 개수 선택
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

    # 파일 업로드 옵션
    st.header("파일 업로드")
    uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

    text_content = process_file(uploaded_file)

    quiz_questions = []
    quiz_answers = []
    user_answers = []

    if text_content is not None:
        if 'quizs' not in st.session_state:
            st.session_state.quizs = None
        if st.button('문제 생성 하기'):
            for i in range(num_quizzes):
                quiz_questions.append(generate_quiz(quiz_type, text_content))
                quiz_answers.append(quiz_questions[i].correct_answer)
            st.session_state['quizs'] = quiz_questions
            st.session_state['canswer'] = quiz_answers
            if st.button("퀴즈 채점 페이지로 이동"):  # New button
                st.experimental_rerun()

        if st.session_state.quizs is not None:
            st.header("생성된 퀴즈")
            for j, question in enumerate(st.session_state.quizs):
                if quiz_type == "주관식":
                    st.write(f"주관식 문제{j+1}: {question.quiz}")
                    st.write("\n")
                else:
                    if quiz_type == "다중 선택 (객관식)":
                        st.write(f"객관식 문제{j+1}: {question.quiz}")
                    else:
                        st.write(f"OX퀴즈 문제{j+1}: {question.quiz}")
                    st.write("\n")
                    st.write(f"{question.options}")
                    st.write("\n")
                    st.write(f"여기는 퀴즈의 구조입니다: {question}")
                    st.write("\n")
                user_answer = st.text_input(f"질문{j + 1}에 대한 답변 입력", "1")
                user_answers.append(user_answer)
                st.session_state['uanswer'] = user_answers
                j += 1
                st.write("-----------------------------------------")
                st.write("\n")
