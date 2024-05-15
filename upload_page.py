#upload_page.py

import streamlit as st
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io

@st.cache(allow_output_mutation=True)
def process_file(uploaded_file, text_area_content):
    if uploaded_file is not None:
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
    elif text_area_content is not None:
        text_content = text_area_content
    else:
        st.warning("파일 또는 텍스트를 업로드하세요.")
        return None

    return text_content

def upload_page():
    st.title("파일 업로드 및 텍스트 입력")

    # 파일 업로드 옵션 선택
    upload_option = st.radio("입력 유형을 선택하세요:", ("텍스트 파일", "이미지 파일", "PDF 파일", "텍스트 직접 입력"))

    # 선택된 옵션에 따라 입력 방식 제공
    if upload_option == "텍스트 파일":
        uploaded_file = st.file_uploader("텍스트 파일을 업로드하세요.", type=["txt"])
    elif upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    else:
        uploaded_file = None

    # 텍스트 입력 영역
    if upload_option == "텍스트 직접 입력":
        text_area_content = st.text_area("텍스트를 입력하세요.")
    else:
        text_area_content = None

    text_content = process_file(uploaded_file, text_area_content)

    if text_content is not None:
        st.success("파일 처리 완료!")
        st.text("파일 내용:")
        st.write(text_content)

# 메인 함수 실행
if __name__ == "__main__":
    upload_page()
