#main.py

import streamlit as st
from upload_page import upload_page
from quiz_creation_page import quiz_creation_page
from quiz_grading_page import quiz_grading_page
#from sign import sign

# 페이지 타이틀 설정
st.set_page_config(page_title="AI 퀴즈 생성기")

# 메인 함수
def main():
    # 페이지 상태 초기화
    page_state = st.session_state

    # 사이드바 메뉴 설정
    menu_options = ["로그인", "파일 업로드", "퀴즈 생성", "퀴즈 채점"]
    selected_page = st.sidebar.radio("메뉴", menu_options)

    # 선택된 페이지 표시
    if selected_page == "파일 업로드":
        upload_page()
    elif selected_page == "퀴즈 생성":
        quiz_creation_page()
    elif selected_page == "퀴즈 채점":
        quiz_grading_page(quiz_questions=page_state.get("quiz_questions", []), page_state=page_state)
    #elif selected_page == "로그인":
        sign()

# 메인 함수 실행
if __name__ == "__main__":
    main()
