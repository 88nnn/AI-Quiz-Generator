import streamlit as st
import boto3

def logout_cognito():
    # 코그니토 클라이언트 초기화
    region_name = 'us-east-1'
    cognito_client = boto3.client('cognito-idp', region_name=region_name)

    # Access Token 가져오기
    access_token = st.session_state.get('access_token')
    
    # 코그니토에서 사용자 세션 무효화
    try:
        cognito_client.global_sign_out(
            AccessToken=access_token
        )
        st.write("로그아웃되었습니다.")
    except cognito_client.exceptions.NotAuthorizedException:
        st.error("로그아웃 실패: 사용자 인증 토큰이 올바르지 않습니다.")
    except Exception as e:
        st.error(f"로그아웃 중 오류 발생: {str(e)}")

    # Streamlit 세션에서 사용자 정보 제거
    st.session_state.user = None
    st.session_state.access_token = None

def start():
    placeholder = st.empty()
    if 'user' not in st.session_state:
        st.session_state.user = None  # 초기화
    
    if st.session_state.user:
        with placeholder.container():
            st.title("퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.switch_page("pages/quiz_creation_page.py")  # 페이지 전환
            st.title("여기는 로그인한 가입자 전용 서비스입니다.")
            if st.button('로그아웃'):
                logout_cognito()  # 코그니토 로그아웃 수행
                st.experimental_rerun()
            if st.header("준비 중입니다."):
                if st.button("퀴즈 저장"):
                    st.write("저장되셨습니다: 결과")
                    if st.button('퀴즈 생성 바로가기'):
                        st.switch_page("pages/quiz_creation_page.py")   # 페이지 전환
    else:
        with placeholder.container():
            st.title("비회원으로 퀴즈 이용하러 돌아가기")
            if st.button('퀴즈 생성 바로가기'):
                st.switch_page("pages/quiz_creation_page.py")   # 페이지 전환
            
            # AWS Cognito 설정
            region_name = 'us-east-1'
            client_id = '7bdv436rrb0l7nhbsva60t7242'

            # Streamlit UI
            st.header("로그인 | ID: admin / PW: Admin12!")
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
                    #authentication_result = response['AuthenticationResult']
                    access_token = authentication_result['AccessToken']
                    st.write(f"Welcome, {username}")
                    st.session_state.user = username  # 유저네임 저장
                    st.session_state.access_token = access_token  # 액세스 토큰 저장
                    st.experimental_rerun()
                except cognito_client.exceptions.NotAuthorizedException:
                    st.error("인증 실패: 사용자 이름 또는 비밀번호가 올바르지 않습니다.")
                except Exception as e:
                    st.error(f"오류 발생: {str(e)}")

            if st.button('회원가입'):
                st.switch_page("pages/sign.py")  # 페이지 전환

if __name__ == "__main__":
    start()
