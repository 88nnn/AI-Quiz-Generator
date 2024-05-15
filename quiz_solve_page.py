#quiz_grading_page.py
import streamlit as st
from quiz_solve_page import grade_quiz_answers

def quiz_grading_page():
    # 세션 상태에서 사용자 답안과 정답 가져오기
    user_answers = st.session_state.get("user_answers", [])
    correct_answers = st.session_state.get("correct_answers", [])

    # 퀴즈 채점
    graded_answers = grade_quiz_answers(user_answers, correct_answers)

    # 채점 결과 표시
    st.title("퀴즈 채점 결과")
    for i, result in enumerate(graded_answers, start=1):
        st.subheader(f"문제 {i}")
        st.write(f"사용자 답변: {user_answers[i-1]}")
        st.write(f"정답: {correct_answers[i-1]}")
        if result == "정답":
            st.success("정답입니다!")
        else:
            st.error("오답입니다.")

    # 퀴즈 생성 페이지로 이동 버튼
    if st.button("퀴즈 생성 페이지로 이동"):
        st.session_state["page"] = "quiz_creation_page"

if __name__ == "__main__":
    quiz_grading_page()
