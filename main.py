import streamlit as st
import quiz_creation_page
import quiz_solve_page
import quiz_grading_page
import awscog
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
load_dotenv()

def main():
    st.sidebar.title("Navigation")
    selected_page = st.sidebar.radio("Go to", ["퀴즈 생성", "퀴즈 풀기", "퀴즈 리뷰", 
                                               "로그인", "회원가입", "퀴즈 저장"])

    if selected_page == "퀴즈 생성":
        quiz_creation_page.quiz_creation_page()
    elif selected_page == "퀴즈 풀기":
        quiz_solve_page.quiz_solve_page()
    elif selected_page == "퀴즈 리뷰":
        quiz_grading_page.quiz_grading_page()
    elif selected_page == "로그인":
        awscog.login()
    elif selected_page == "회원가입":
        awscog.sign()  # 회원가입 함수 호출
    #elif selected_page == "로그인":
     #   awscog.login()  # 로그인 함수 호출


if __name__ == "__main__":
    main()
