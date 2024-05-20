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
from db_connect import retrieve_results
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader


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
