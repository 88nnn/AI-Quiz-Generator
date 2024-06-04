import streamlit as st
import boto3

def start():
    placeholder = st.empty()
    if 'user' not in st.session_state:
        st.session_state.user = None  # 초기화
    
    if st.session_state.user:
        with placeholder.container():
            st.title("퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.experimental_set_query_params(page="quiz_creation_page")  # 페이지 전환
            st.title("여기는 로그인한 가입자 전용 서비스입니다.")
            if st.button('로그아웃'):
                st.session_state.user = None
                st.experimental_rerun()
            if st.button("퀴즈 저장"):
                st.write("저장되셨습니다: 결과")
                if st.button('퀴즈 생성 바로가기'):
                    st.experimental_set_query_params(page="quiz_creation_page")  # 페이지 전환
    else:
        with placeholder.container():
            st.title("비회원으로 퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.experimental_set_query_params(page="quiz_creation_page")  # 페이지 전환
            
            # AWS Cognito 설정
            region_name = 'us-east-1'
            client_id = '57gm5vjnk9p3ehk9hn5s97ropu'
            user_pool_id = 'us-east-1_TXA2Lha1Y'

            # Streamlit UI
            st.header("로그인")
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")

            if st.button("Login"):
                cognito_client = boto3.client('cognito-idp', region_name=region_name)
                try:
                    response = cognito_client.initiate_auth(
                        ClientId=client_id,
                        AuthFlow='USER_PASSWORD_AUTH',
                        AuthParameters={
                            'USERNAME': username,
                            'PASSWORD': password
                        }
                    )
                    user = response['AuthenticationResult']
                    st.write(f"Welcome, {username}")
                    st.session_state.user = username  # 유저네임 저장
                    st.experimental_rerun()
                except cognito_client.exceptions.NotAuthorizedException:
                    st.error("인증 실패: 사용자 이름 또는 비밀번호가 올바르지 않습니다.")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

            if st.button('회원가입'):
                st.experimental_set_query_params(page="sign")  # 페이지 전환

if __name__ == "__main__":
    start()
