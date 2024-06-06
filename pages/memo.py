import streamlit as st

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
