import streamlit as st
import boto3

# AWS Cognito 설정
region_name = 'us-east-1'
client_id = '57gm5vjnk9p3ehk9hn5s97ropu'
user_pool_id = 'us-east-1_TXA2Lha1Y'

# Streamlit UI
st.title("AWS Cognito 인증")

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
        # 사용자 정보 추출 (예시)
        user = response['AuthenticationResult']
        st.write(f"Welcome, {username}")
        # 이후 사용자 전용 서비스를 운영하기 위한 동시성 코드 추가
        st.session_state['user'] = user

    except cognito_client.exceptions.NotAuthorizedException:
        st.error("인증 실패: 사용자 이름 또는 비밀번호가 올바르지 않습니다.")
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")

# 사용자 전용 서비스 코드 예시
if 'user' in st.session_state:
    st.write("여기는 로그인한 사용자 전용 서비스입니다.")
    # 사용자 전용 서비스 코드 추가
