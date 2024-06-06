import streamlit as st

st.set_page_config(
    page_title="memo",
)

# 세션에서 닉네임 가져오기
if 'user' not in st.session_state:
    st.session_state.user = None

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
        con.write("오답노트가 없습니다.")

def memo():
    user = get_username()
    if user == "unknown" or user == "" or user == None : 
        st.stop()
    else:
        st.title(f"닉네임: {user}")

        st.sidebar.header("오답노트 남기기")
        memo_text = st.text_input(label="오답노트 남기기", value="오답노트 입력")

        display_memos()

        if st.button("메모 등록"):
            if memo_text:
                save_memo(user, memo_text)
                st.experimental_rerun()

