#quiz_creation_page.py

import io
import re
import pytesseract
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
#from topic_creation import topic_select, topic_list
#from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient

class CreateQuizoub(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

class CreateQuizsub(BaseModel):
    quiz = ("quiz =The created problem")
    correct_answer = ("correct_answer =The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz = ("The created problem")
    options1 = ("The true or false option of the created problem")
    options2 = ("The true or false option of the created problem")
    correct_answer = ("One of the options1 or options2")
def make_model(pages):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    # Rag
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)

    # PydanticOutputParser 생성
    parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
    parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
    parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

    prompt = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN."

        "CONTEXT:"
        "{context}."

        "FORMAT:"
        "{format}"
    )
    promptoub = prompt.partial(format=parseroub.get_format_instructions())
    promptsub = prompt.partial(format=parsersub.get_format_instructions())
    prompttf = prompt.partial(format=parsertf.get_format_instructions())

    document_chainoub = create_stuff_documents_chain(llm, promptoub)
    document_chainsub = create_stuff_documents_chain(llm, promptsub)
    document_chaintf = create_stuff_documents_chain(llm, prompttf)

    retriever = vector.as_retriever()

    retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
    retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
    retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

    # chainoub = promptoub | chat_model | parseroub
    # chainsub = promptsub | chat_model | parsersub
    # chaintf = prompttf | chat_model | parsertf
    return 0


    # chainoub = promptoub | chat_model | parseroub
    # chainsub = promptsub | chat_model | parsersub
    # chaintf = prompttf | chat_model | parsertf

def load_data_from_mongodb():
    # MongoDB 연결 및 데이터 불러오기
    client = MongoClient("mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
    db = client["sample_mflix"]
    collection = db["movies"]
    data = collection.find({}, {"_id": 0, "text": 1})  # 필요한 필드만 선택하여 가져오기
    return data

def prepare_data_for_rag(data):
    # 불러온 데이터를 RAG 모델의 입력 형식에 맞게 변환
    return "\n".join([doc["text"] for doc in data])

@st.cache(allow_output_mutation=True)
def process_file(uploaded_file, text_area_content):
    if uploaded_file is not None:
        # 업로드된 파일 처리
        if uploaded_file.type == "text/plain":
            text_content = uploaded_file.read().decode("utf-8")
        elif uploaded_file.type.startswith("image/"):
            image = Image.open(uploaded_file)
            text_content = pytesseract.image_to_string(image)
        elif uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
            text_content = ""
            for page in pdf_reader.pages:
                text_content += page.extract_text()
        else:
            st.error("지원하지 않는 파일 형식입니다.")
            return None
    elif text_area_content is not None:
        text_content = text_area_content
    else:
        st.warning("파일 또는 텍스트를 업로드하세요.")
        return None

    return text_content


# 파일 처리 함수
def process_file(uploaded_file):
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

    # 업로드된 파일 처리
    if uploaded_file.type == "text/plain":
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=100,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    texts = text_splitter.create_documents([text_content])
    return texts

# 퀴즈 생성 함수
@st.experimental_fragment
def generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub, retrieval_chaintf):
    # Generate quiz prompt based on selected quiz type
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chainoub.invoke(
            {
                "input": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "주관식":
        response = retrieval_chainsub.invoke(
            {
                "input": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    elif quiz_type == "OX 퀴즈":
        response = retrieval_chaintf.invoke(
            {
                "input": "Create one true or false question focusing on important concepts, following the given format, referring to the following context"
            }
        )
    quiz_questions = response

    return quiz_questions

@st.experimental_fragment
def grade_quiz_answer(user_answer, quiz_answer):
    if user_answer.lower() == quiz_answer.lower():
        grade = "정답"
    else:
        grade = "오답"
    return grade


# 메인 함수
def quiz_creation_page():
    placeholder = st.empty()
    st.session_state.page = 0
    if st.session_state.page == 0:
        with placeholder.container():
            st.title("AI 퀴즈 생성기")
            if 'selected_page' not in st.session_state:
                st.session_state.selected_page = ""

            # 퀴즈 유형 선택
            quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

            # 퀴즈 개수 선택
            num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

            # 퀴즈 주제 선택
            language = "언어"
            english = "영어"
            korean = "한국어"
            language_topic = [english, korean]
            
            mathematic = "수리"
            algebra = "대수학"
            geometry = "기하학"
            calculus = "미적분학"
            statistics = "통계학"
            mathematic_topic = [algebra, geometry, calculus, statistics]
            
            social_science = "사회과학"
            psychology = "심리학"
            sociology = "사회학"
            economics = "경제학"
            political_science = "정치학"
            social_science_topic = [psychology, sociology, economics, political_science]
            
            natural_science = "자연과학"
            physics = "물리학"
            chemistry = "화학"
            biology = "생물학"
            astronomy = "천문학"
            natural_science_topic = [physics, chemistry, biology, astronomy]
            
            humanity = "인문학"
            philosophy = "철학"
            history = "역사학"
            literature = "문학"
            art_history = "미술사"
            humanity_topic = [philosophy, history, literature, art_history]
            
            engineering = "공학"
            computer_engineering = "컴퓨터 공학"
            architectural_engineering = "건축공학"
            engineering_topic = [computer_engineering, architectural_engineering]
            
            art = "예술"
            film = "영화"
            novel = "소설"
            art_topic = [film, novel]
            
            topic = [language + "(미지원)", mathematic + "(미지원)",
             social_science + "(미지원)", natural_science + "(미지원)",
             humanity + "(미지원)", engineering + "(미지원)",
             art + "(영화 분야 지원)"]

            selected_topics = st.multiselect("생성할 퀴즈의 주제를 선택하세요. (중복 선택 가능)", topic)

            # 주제 직접 입력
            def subtopic_select(selected_topics):
                sub_topics = []
                for now_topic in selected_topics:
                    if now_topic.startswith(language):
                        sub_topics.extend(language_topic)
                    elif now_topic.startswith(mathematic):
                        sub_topics.extend(mathematic_topic)
                    elif now_topic.startswith(social_science):
                        sub_topics.extend(social_science_topic)
                    elif now_topic.startswith(natural_science):
                        sub_topics.extend(natural_science_topic)
                    elif now_topic.startswith(humanity):
                        sub_topics.extend(humanity_topic)
                    elif now_topic.startswith(engineering):
                        sub_topics.extend(engineering_topic)
                    elif now_topic.startswith(art):
                        sub_topics.extend(art_topic)
                return sub_topics

            sub_topics = subtopic_select(selected_topics)

            selected_sub_topics = st.multiselect("선택한 주제의 하위 분류를 선택하세요. (중복 선택 가능)", sub_topics)

            st.write("선택한 주제:", selected_topics)
            st.write("선택한 하위 분류:", selected_sub_topics)

            if selected_sub_topics == "film":
                def display_prepared_data(prepared_data):
    # 준비된 데이터를 출력
                    print("RAG 모델 입력 형식에 맞게 변환된 데이터:")
                    print(prepared_data) # MongoDB에서 데이터 불러오기
                    
                data = load_data_from_mongodb()
                # RAG 모델 입력 형식에 맞게 데이터 준비
                prepared_data = prepare_data_for_rag(data)
            # 준비된 데이터 출력
                display_prepared_data(prepared_data)
            
            # 파일 업로드 옵션
            st.header("파일 업로드")
            uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

            text_content = process_file(uploaded_file)

            quiz_questions = []

            if text_content is not None:

                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                        embeddings = OpenAIEmbeddings()

                        # Rag
                        text_splitter = RecursiveCharacterTextSplitter()
                        documents = text_splitter.split_documents(text_content)
                        vector = FAISS.from_documents(documents, embeddings)

                        # PydanticOutputParser 생성
                        parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
                        parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
                        parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

                        prompt = PromptTemplate.from_template(
                            "{input}, Please answer in KOREAN."

                            "CONTEXT:"
                            "{context}."

                            "FORMAT:"
                            "{format}"
                        )
                        promptoub = prompt.partial(format=parseroub.get_format_instructions())
                        promptsub = prompt.partial(format=parsersub.get_format_instructions())
                        prompttf = prompt.partial(format=parsertf.get_format_instructions())

                        document_chainoub = create_stuff_documents_chain(llm, promptoub)
                        document_chainsub = create_stuff_documents_chain(llm, promptsub)
                        document_chaintf = create_stuff_documents_chain(llm, prompttf)

                        retriever = vector.as_retriever()

                        retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
                        retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
                        retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

                        for i in range(num_quizzes):
                            quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub,retrieval_chaintf))
                            st.session_state['quizs'] = quiz_questions
                        st.session_state.selected_page = "퀴즈 풀이"
                        st.session_state.selected_type = quiz_type
                        st.session_state.selected_num = num_quizzes

                        st.success('퀴즈 생성이 완료되었습니다!')
                        st.write(quiz_questions)

                if st.button('퀴즈 풀기'):
                    st.switch_page("pages/quiz_solve_page.py")


if __name__ == "__main__":
    quiz_creation_page()
