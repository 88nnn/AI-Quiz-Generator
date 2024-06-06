import streamlit as st

st.set_page_config(
    page_title="memo",
)

# 세션에서 닉네임 가져오기
if 'user' not in st.session_state:
    st.session_state.user = None

def get_username():
    if st.session_state.user is None:
        st.session_state.user = st.text_input("닉네임을 입력하세요", "unknown")
    return st.session_state.user

def save_memo(user, memo_text):
    with open("memoRecord.txt", "a") as file:
        file.write(f"{user} : {memo_text}\n")

def display_memos():
    con = st.container()
    con.caption("Result")
    try:
        with open("memoRecord.txt", "r") as file:
            for line in file:
                con.write(line)
    except FileNotFoundError:
        con.write("메모가 없습니다.")

def memo():
    user = get_username()
    st.title(f"닉네임: {user}")
    
    st.sidebar.header("메모 남기기")
    memo_text = st.text_input(label="메모 남기기", value="메모 입력")

    if user:
        display_memos()

        if st.button("메모 등록"):
            if memo_text:
                save_memo(user, memo_text)
                st.experimental_rerun()
    else:
        st.warning("닉네임을 입력해주세요.")

