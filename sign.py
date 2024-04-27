#sign.py

import yaml
import streamlit as st
import streamlit_authenticator as auth


# 샘플 사용자 데이터베이스 (실제로는 데이터베이스나 API 호출로 대체해야 합니다)
#authenticator = auth.Authenticate(
 #   config['credentials'],
  #  config['cookie']['name'],
   # config['cookie']['key'],
    #config['cookie']['expiry_days'],
    #config['preauthorized']
#)

#names = ["Smith Lang", "sample id"]
#usernames = ["lsmith", "samid"]
#passwords = ["123","456"] # yaml 파일 생성하고 비밀번호 지우기!

def register_user(name, username, email, password):
    """
    새로운 사용자를 등록합니다.

    Args:
        username (str): 원하는 사용자 이름.
        password (str): 원하는 비밀번호.

    Returns:
        str: 성공적인 등록 또는 오류 메시지.
    """
    new_data = {
        "credentials": {
            "usernames": {
                username: {
                    "email": email
                    "name": name,
                    "password": auth.Hasher([password]).generate()[0]  # 비밀번호 해싱
                }
            }
        }
    }
    # 기존 데이터 읽기
    with open('config.yaml', 'r') as file:
        existing_data = yaml.safe_load(file)
    
    # 새로운 계정 정보 추가
    existing_data['credentials']['usernames'].update(new_data['credentials']['usernames'])
    
    # YAML 파일에 쓰기
    with open('config.yaml', 'w') as file:
        yaml.dump(existing_data, file, default_flow_style=False)
    
    st.success("계정이 성공적으로 생성되었습니다!")

def login_user(name, authentication_status, username, password):
    """
    기존 사용자를 로그인합니다.

    Args:
        username (str): 사용자 이름.
        password (str): 비밀번호.

    Returns:
        str: 성공적인 로그인 또는 오류 메시지.
    """

    name, authentication_status, username = authenticator.login(username, password)
    # authentication_status : 인증 상태 (실패=>False, 값없음=>None, 성공=>True)
    if authentication_status == False:
        st.error("잘못된 사용자 이름 또는 비밀번호입니다.")
    if authentication_status == None:
        st.warning("사용자 이름 또는 비밀번호를 입력해주세요.")
    if authentication_status:
        return f"'{username}' 사용자가 성공적으로 로그인되었습니다."
    
    """if username in users and users[username]["password"] == password:
        return f"'{username}' 사용자가 성공적으로 로그인되었습니다."
    else:
        return "오류: 잘못된 사용자 이름 또는 비밀번호입니다."
        """

def sign():
    st.title("User Registration & Login")
    with open('config.yaml') as file:
        config = yaml.load(file, Loader=auth.SafeLoader)
     
     authenticator = auth.Authenticate(
        config['credentials'],
        config['cookie']['name'],
        config['cookie']['key'],
        config['cookie']['expiry_days'],
        config['preauthorized']
    )

    
    # User registration
    st.header("Register")
    new_name = st.text_input("Enter your name:")
    new_username = st.text_input("Enter a new username:")
    new_email = st.text_input("Enter your email:")
    new_password = st.text_input("Enter a new password:", type="password")
    if st.button("Register"):
        result = register_user(new_name, new_username, new_email, new_password)
        st.success(result)
        
    # User login
    st.header("Login")
    existing_username = st.text_input("Enter your username:")
    existing_password = st.text_input("Enter your password:", type="password")
    if st.button("Login"):
        result = login_user(existing_username, existing_password)
        st.success(result)

