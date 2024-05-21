import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
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
from db_connect import create_quiz_retrieval_chain, retrieve_results

def process_file(uploaded_file, upload_option):
    text_content = ""
    if upload_option == "이미지 파일":
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif upload_option == "PDF 파일":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    else:
        st.error("지원하지 않는 파일 형식입니다.")
    return text_content

@st.experimental_fragment
def generate_quiz(quiz_type, text_content, retrieval_chain_mc, retrieval_chain_subj, retrieval_chain_tf):
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chain_mc.invoke(
            {
                "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "주관식":
        response = retrieval_chain_subj.invoke(
            {
                "input": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = retrieval_chain_tf.invoke(
            {
                "input": "Create one true or false question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    return response

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    return "정답" if user_answer.lower() == quiz_answer.lower() else "오답"

def quiz_creation_page():
    st.title("AI 퀴즈 생성기")
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)
    upload_option = st.radio("입력 유형을 선택하세요", ("이미지 파일", "PDF 파일", "직접 입력", "URL", "토픽 선택"))

    if upload_option == "직접 입력":
        text_content = st.text_area("텍스트를 입력하세요.")
    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        loader = RecursiveUrlLoader(url=url_area_content)
        text_content = loader.load()
    else:
        uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["jpg", "jpeg", "png", "pdf"])
        text_content = process_file(uploaded_file, upload_option) if uploaded_file else None

    quiz_questions = []

    if text_content and st.button('문제 생성 하기'):
        with st.spinner('퀴즈를 생성 중입니다...'):
            # Try to retrieve results from the vector search
            results = retrieve_results(text_content)

            if results:
                pages = [{"content": result['text']} for result in results]
                retrieval_chain_mc, retrieval_chain_subj, retrieval_chain_tf = create_quiz_retrieval_chain(pages)
            else:
                # If no results, use the user's input directly
                pages = [{"content": text_content}]
                retrieval_chain_mc, retrieval_chain_subj, retrieval_chain_tf = create_quiz_retrieval_chain(pages)

            for _ in range(num_quizzes):
                quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chain_mc, retrieval_chain_subj, retrieval_chain_tf))

            st.success('퀴즈 생성이 완료되었습니다!')
            st.write(quiz_questions)

            if st.button('퀴즈 풀기'):
                st.session_state['quiz_questions'] = quiz_questions
                st.session_state['quiz_type'] = quiz_type
                st.session_state['num_quizzes'] = num_quizzes
                st.switch_page("quiz_solve_page")

if __name__ == "__main__":
    quiz_creation_page()
