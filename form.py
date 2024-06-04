#가져오기
import streamlit as st
#사이드 바와 선택 박스
page = st.sidebar.selectbox(‘Choose your page’, [‘INPUT FORM’, ‘RESULT’])
#정보 입력 후 함수
def update_page():
st.balloons()
st.markdown(‘# Thank you for information’)
st.json(customer_information)
#혹시 선택 박스에서선택한 페이지가INPUT FORM이라면
if page == ‘INPUT FORM’:
st.title(‘INPUT FORMATION’)
#각종 입력 폼
with st.form(key=’customer’):
customer_name: str = st.text_input(‘NAME’, max_chars=15)
customer_age: int = st.text_input(‘AGE’, max_chars=3)
customer_gender = st.radio(“GENDER”,(‘MEN’, ‘Women’))
customer_address = st.selectbox(‘COUNTRY’,
(‘Hokkaido’, ‘Tohoku’, ‘Kanto’, ‘Chubu’, ‘Kinki’, ‘Kansai’, ‘Chugoku’, ‘Shikoku’, ‘Kyusyu’, ‘Okinawa’))
customer_mail: str = st.text_input(‘Mail Address’, max_chars = 30)
#폼에 입력 결과를 정리
customer_information = {
‘customer_name’: customer_name,
‘customer_age’ : customer_age,
‘customer_gender’: customer_gender,
‘customer_address’: customer_address,
‘customer_mail’: customer_mail
}
#폼에 입력 결과를 송신
submit_button = st.form_submit_button(label=’Send’)
#submit_button가 송신 되면 함수를 실행
if submit_button:
update_page()
