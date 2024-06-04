import streamlit as st
import boto3
import hashlib
import bcrypt

def register_cognito_user(username, email, password):
    # 코그니토 클라이언트 생성
    client = boto3.client('cognito-idp', region_name='your_region_name')

    # 사용자 등록 요청
    try:
        response = client.sign_up(
            ClientId='your_client_id',
            Username=username,
            Password=password,
            UserAttributes=[
                {
                    'Name': 'email',
                    'Value': email
                },
                {
                    'Name': 'custom:username',
                    'Value': username
                },
            ],
        )
        return True  # 사용자 등록에 성공하면 True 반환
    except client.exceptions.InvalidPasswordException:
        return "비밀번호가 적합하지 않습니다."
    except client.exceptions.UsernameExistsException:
        return "이미 사용 중인 사용자 이름입니다."
    except client.exceptions.InvalidParameterException as e:
        return f"오류 발생: {e}"

    return False  # 그 외의 경우는 사용자 등록 실패로 간주하여 False 반환


def sign():
    st.title("회원가입")

    # 회원가입 양식
    st.header("회원가입")
    new_username = st.text_input("새로운 사용자 이름 입력:")
    new_email = st.text_input("이메일 입력:")
    new_password = st.text_input("새로운 비밀번호 입력:", type="password")

    if st.button("가입하기"):
        registration_result = register_cognito_user(new_username, new_email, new_password)
        if registration_result == True:
            st.success("계정이 성공적으로 생성되었습니다!")
            
            # 코그니토 로그인 처리
            cognito_client = boto3.client('cognito-idp', region_name='your_region_name')
            try:
                response = cognito_client.initiate_auth(
                    ClientId='your_client_id',
                    AuthFlow='USER_PASSWORD_AUTH',
                    AuthParameters={
                        'USERNAME': new_username,
                        'PASSWORD': new_password
                    }
                )
                user = response['AuthenticationResult']
                st.success(f"{new_username}님, 환영합니다!")
                if st.button('퀴즈 생성 바로가기'):
                    st.switch_page("pages/quiz_creation_page.py")   # 페이지 전환
                    
                # 여기서 자동으로 로그인된 것으로 처리하거나, 로그인 관련 작업을 추가할 수 있습니다.
            except cognito_client.exceptions.NotAuthorizedException:
                st.error("로그인 실패: 사용자 이름 또는 비밀번호가 올바르지 않습니다.")
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
            
        else:
            st.error(registration_result)

if __name__ == "__main__":
    sign()
