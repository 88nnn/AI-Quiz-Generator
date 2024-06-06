import streamlit as st
import boto3

def register_cognito_user(username, password):
    region_name = 'us-east-1'
    client_id = '7bdv436rrb0l7nhbsva60t7242'
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
            
            region_name = 'us-east-1'
            client_id = '7bdv436rrb0l7nhbsva60t7242'
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
                st.write(response)  # 디버깅을 위해 전체 응답 출력
                if 'AuthenticationResult' in response:
                    authentication_result = response['AuthenticationResult']
                    access_token = authentication_result['AccessToken']
                    st.session_state.user = new_username
                    st.session_state.access_token = access_token
                    st.experimental_rerun()
                else:
                    st.error("자동 로그인에 실패했습니다. 로그인 페이지로 이동하세요.")
            except Exception as e:
                st.error(f"자동 로그인 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    sign()
