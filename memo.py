if('user_name' not in st.session_state):
    st.session_state['user_name'] = 'unknown'
    
name = st.session_state['user_name']

st.title("닉네임: " + name)

#sideBar에 설정해 주어야 이동가능
st.sidebar.header("메모 남기기")

memo = st.text_input(label="메모 남기기", value="메모 입력")

con = st.container()
con.caption("Result")

# 기존의 메모 데이터 읽어와서 출력
file = open("memoRecord.txt", "r")
while True:
    line = file.readline()
    if not line:
        break
    con.write(line)
    
file.close()

#메모 기록하기
if st.button("메모 등록"):
    con.write(f"{name} : {str(memo)}")
    
    file = open("memoRecord.txt", "a")
    file.write(f"{name} : {str(memo)}\n")
    file.close()
