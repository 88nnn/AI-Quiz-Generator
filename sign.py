import re
import yaml
import streamlit as st
import bcrypt

def is_valid_username(username):
    """
    올바른 형식의 사용자 이름인지 확인합니다.
    """
    # 영어 5글자 이상인지 확인
    if len(username) < 5 or not username.isalpha():
        return False
    return True

def is_valid_password(password):
    """
    올바른 형식의 비밀번호인지 확인합니다.
    """
    # 영어, 숫자, 특수문자(!@#$%^&*-_+=)가 모두 하나 이상 포함되어 있는지 확인
    if len(password) < 8:
        return False
    if not re.search("[a-zA-Z]", password):
        return False
    if not re.search("[0-9]", password):
        return False
    if not re.search("[!@#$%^&*_\-+=]", password):
        return False
    return True

def is_valid_email(email):
    """
    올바른 형식의 이메일 주소인지 확인합니다.
    """
    # 이메일 주소의 패턴을 확인하는 정규식
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if re.match(email_pattern, email):
        return True
    else:
        return False


def register_user(name, username, email, password):
    """
    새로운 사용자를 등록합니다.

    Args:
        username (str): 원하는 사용자 이름.
        password (str): 원하는 비밀번호.

    Returns:
        str: 성공적인 등록 또는 오류 메시지.
    """
    # 사용자 이름, 비밀번호 및 이메일 형식 확인
    if not is_valid_username(username):
        return "계정 생성 실패: 사용자 이름은 영어로 5글자 이상이어야 합니다."
    if not is_valid_password(password):
        return "계정 생성 실패: 비밀번호는 영어, 숫자, 특수문자(!@#$%^&*_\-+=)가 모두 하나 이상 포함된 8글자 이상이어야 합니다."
    if not is_valid_email(email):
        return "계정 생성 실패: 올바른 이메일 주소를 입력해주세요."
    
    # 비밀번호 해싱
    hashed_password = bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()

    # 기존 데이터 읽기
    with open('config.yaml', 'r') as file:
        existing_data = yaml.safe_load(file)
    
    # 새로운 계정 정보 추가
    if username in existing_data['credentials']['usernames']:
        return "계정 생성 실패: 이미 사용 중인 사용자 이름입니다."
    else:
        new_data = {
            "credentials": {
                "usernames": {
                    username: {
                        "email": email,
                        "name": name,
                        "password": hashed_password
                    }
                }
            }
        }

        existing_data['credentials']['usernames'].update(new_data['credentials']['usernames'])
        
        # YAML 파일에 쓰기
        with open('config.yaml', 'w') as file:
            yaml.dump(existing_data, file, default_flow_style=False)
        
        # 회원가입 성공 메시지 출력
        message = f"계정이 성공적으로 생성되었습니다!\n- 이름: {name}\n- 사용자 이름: {username}\n- 이메일 주소: {email}"
        return message





def login_user(username, password):
    """
    기존 사용자를 로그인합니다.

    Args:
        username (str): 사용자 이름.
        password (str): 비밀번호.

    Returns:
        str: 성공적인 로그인 또는 오류 메시지.
    """
    # 사용자 정보 확인 및 인증
    with open('config.yaml', 'r') as file:
        existing_data = yaml.safe_load(file)
    
    user_info = existing_data['credentials']['usernames'].get(username)
    if user_info is None:
        st.error("잘못된 사용자 이름 또는 비밀번호입니다.")
        return
    if user_info["password"] == hashlib.sha256(password.encode()).hexdigest():
        return f"'{username}' 사용자가 성공적으로 로그인되었습니다."
    else:
        st.error("잘못된 사용자 이름 또는 비밀번호입니다.")

def sign():
    st.title("User Registration & Login")
    
    # User login
    st.header("Login")
    existing_username = st.text_input("Enter your username:", key="username_input")
    existing_password = st.text_input("Enter your password:", type="password", key="password_input")
    if st.button("Login", key="login_button"):
        result = login_user(existing_username, existing_password)
        if result:
            st.success(result)
    
    # Ask for registration
    st.write("Would you like to register?")
    if st.button("Register"):
        show_registration_form()

def show_registration_form():
    # User registration form
    st.subheader("Register")
    with st.form(key="registration_form"):
        new_name = st.text_input("Enter your name:")
        new_username = st.text_input("Enter a new username: it must be over 5 alphabets")
        new_email = st.text_input("Enter your email:")
        new_password = st.text_input("Enter a new password: it must have an alphabet, number, and one of !@#$%^&*_\-+= more than one", type="password")
        submit_button = st.form_submit_button(label="Sign Up")
        if submit_button:
            result = register_user(new_name, new_username, new_email, new_password)
            if "성공적으로" in result:
                st.success(result)
            else:
                st.error(result)


