import streamlit as st
import boto3

def register_cognito_user(username, password):
    region_name = 'us-east-1'
    client_id = '7bdv436rrb0l7nhbsva60t7242'
    user_pool_id = 'us-east-1_TXA2Lha1Y'
    
    cognito_client = boto3.client('cognito-idp', region_name=region_name)

    try:
        response = cognito_client.sign_up(
            ClientId=client_id,
            Username=username,
            Password=password
        )
        return True
    except cognito_client.exceptions.UsernameExistsException:
        st.error("이미 존재하는 사용자 이름입니다.")
        return False
    except Exception as e:
        st.error(f"회원가입 중 오류 발생: {str(e)}")
        return False

def sign():
    st.title("회원가입")
    
    new_username = st.text_input("새 사용자 이름을 입력하세요:")
    new_password = st.text_input("새 비밀번호를 입력하세요:", type="password")

    if st.button("회원가입"):
        if register_cognito_user(new_username, new_password):
            st.success("회원가입이 완료되었습니다! 자동으로 로그인됩니다...")
            # 자동 로그인 시도
            region_name = 'us-east-1'
            client_id = '57gm5vjnk9p3ehk9hn5s97ropu'
            cognito_client = boto3.client('cognito-idp', region_name=region_name)
            try:
                response = cognito_client.initiate_auth(
                    ClientId=client_id,
                    AuthFlow='USER_PASSWORD_AUTH',
                    AuthParameters={
                        'USERNAME': new_username,
                        'PASSWORD': new_password
                    }
                )
                authentication_result = response['AuthenticationResult']
                access_token = authentication_result['AccessToken']
                st.session_state.user = new_username  # 유저네임 저장
                st.session_state.access_token = access_token  # 액세스 토큰 저장
                st.experimental_rerun()
            except Exception as e:
                st.error(f"자동 로그인 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    sign()
