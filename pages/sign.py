import streamlit as st
import boto3

def register_cognito_user(username, password):
    """
    코그니토를 사용하여 새로운 사용자를 등록합니다.

    Args:
        username (str): 사용자 이름.
        password (str): 비밀번호.

    Returns:
        bool: 회원가입 성공 여부.
    """
    # 코그니토 클라이언트 초기화
    region_name = 'us-east-1'
    user_pool_id = 'us-east-1_TXA2Lha1Y'
    client_id = '57gm5vjnk9p3ehk9hn5s97ropu'
    cognito_client = boto3.client('cognito-idp', region_name=region_name)

    # 코그니토를 통한 회원가입 시도
    try:
        response = cognito_client.sign_up(
            ClientId=client_id,
            Username=username,
            Password=password
        )
        st.success("회원가입이 성공적으로 완료되었습니다!")
        return True
    except cognito_client.exceptions.UsernameExistsException:
        st.error("이미 사용 중인 사용자 이름입니다.")
        return False
    except Exception as e:
        st.error(f"오류 발생: {str(e)}")
        return False

def sign():
    st.title("회원가입User Registration")
    
    # User registration
    st.header("Register")
    new_username = st.text_input("Enter a new username:")
    new_password = st.text_input("Enter a new password:", type="password")
    if st.button("Register"):
        if register_cognito_user(new_username, new_password):
            st.success("회원가입이 성공적으로 완료되었습니다!")
            st.write("로그인 페이지로 이동하려면 아래 버튼을 클릭하세요.")
            if st.button("로그인 페이지로 이동"):
                st.experimental_set_query_params(page="login")  # 페이지 전환

if __name__ == "__main__":
    sign()
