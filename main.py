import streamlit as st
from pages import quiz_creation_page, quiz_solve_page, quiz_grading_page, awscog, sign
from streamlit_option_menu import option_menu
import os
from dotenv import load_dotenv
load_dotenv()
from memo import save_memo, display_memos



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
        awscog.start()
    elif selected_page == "회원가입":
        sign.sign()

if __name__ == "__main__":
    main()
