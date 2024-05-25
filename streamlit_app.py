# quiz_creation_page.py

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains import create_retrieval_chain, create_stuff_documents_chain
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
from pymongo import MongoClient
import topic_creation as tc

class CreateQuizMCQ(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option")
    options2: str = Field(description="The second option")
    options3: str = Field(description="The third option")
    options4: str = Field(description="The fourth option")
    correct_answer: str = Field(description="One of the options")

class CreateQuizOpenEnded(BaseModel):
    quiz: str = Field(description="The created problem")
    correct_answer: str = Field(description="The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="True option")
    options2: str = Field(description="False option")
    correct_answer: str = Field(description="One of the options")

def connect_db():
    client = MongoClient('mongodb+srv://<username>:<password>@cluster0.mongodb.net/?retryWrites=true&w=majority')
    return client['sample_mflix']

def process_text(text_area_content):
    text_content = st.text_area("텍스트를 입력하세요.")
    return text_content

def process_file(uploaded_file, upload_option):
    if upload_option == "텍스트 파일":
        text_content = uploaded_file.read().decode("utf-8")
    elif upload_option == "이미지 파일":
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif upload_option == "PDF 파일":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = ""
        for page in pdf_reader.pages:
            text_content += page.extract_text()
    return text_content

def retrieve_results(user_query, vector_search):
    response = vector_search.similarity_search_with_score(
        user_query=user_query, k=3
    )
    template = """
    Use the following pieces of context to answer the question at the end.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    {context}
    Question: {question}
    """
    custom_rag_prompt = PromptTemplate.from_template(template)
    llm = ChatOpenAI()
    parser = PydanticOutputParser(pydantic_object=CreateQuizMCQ)
    rag_chain = (
        {"context": response, "question": user_query}
        | custom_rag_prompt
        | llm
        | parser
    )
    answer = rag_chain.invoke(user_query)
    return answer

def generate_quiz(quiz_type, text_content, retrieval_chainMCQ, retrieval_chainOpenEnded, retrieval_chainTF):
    if quiz_type == "다중 선택 (객관식)":
        response = retrieval_chainMCQ.invoke(
            {"input": "Create one multiple-choice question focusing on important concepts."}
        )
    elif quiz_type == "주관식":
        response = retrieval_chainOpenEnded.invoke(
            {"input": "Create one open-ended question focusing on important concepts."}
        )
    elif quiz_type == "OX 퀴즈":
        response = retrieval_chainTF.invoke(
            {"input": "Create one true or false question focusing on important concepts."}
        )
    return response

def quiz_creation_page():
    st.title("AI 퀴즈 생성기")
    if 'quiz_created' not in st.session_state:
        st.session_state.quiz_created = False

    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

    text_content = None
    if upload_option in ["텍스트 파일", "이미지 파일", "PDF 파일"]:
        uploaded_file = st.file_uploader(f"{upload_option}을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])
        if uploaded_file:
            text_content = process_file(uploaded_file, upload_option)
    elif upload_option == "URL":
        url_area_content = st.text_area("URL을 입력하세요.")
        if url_area_content:
            loader = RecursiveUrlLoader(url=url_area_content)
            text_content = loader.load()
    elif upload_option == "토픽 선택":
        topic = st.selectbox("토픽을 선택하세요", tc.topic_list())
        if topic:
            subtopics = tc.subtopic_select(topic)
            selected_subtopics = st.multiselect("세부 토픽을 선택하세요", subtopics)
            text_content = " ".join(selected_subtopics)

    if text_content:
        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                embeddings = OpenAIEmbeddings()
                text_splitter = RecursiveCharacterTextSplitter()
                documents = text_splitter.split_documents([text_content])
                vector = FAISS.from_documents(documents, embeddings)

                parserMCQ = PydanticOutputParser(pydantic_object=CreateQuizMCQ)
                parserOpenEnded = PydanticOutputParser(pydantic_object=CreateQuizOpenEnded)
                parserTF = PydanticOutputParser(pydantic_object=CreateQuizTF)

                prompt = PromptTemplate.from_template(
                    "{input}, Please answer in KOREAN."
                    "CONTEXT: {context}."
                    "FORMAT: {format}"
                )
                promptMCQ = prompt.partial(format=parserMCQ.get_format_instructions())
                promptOpenEnded = prompt.partial(format=parserOpenEnded.get_format_instructions())
                promptTF = prompt.partial(format=parserTF.get_format_instructions())

                document_chainMCQ = create_stuff_documents_chain(llm, promptMCQ)
                document_chainOpenEnded = create_stuff_documents_chain(llm, promptOpenEnded)
                document_chainTF = create_stuff_documents_chain(llm, promptTF)

                retriever = vector.as_retriever()

                retrieval_chainMCQ = create_retrieval_chain(retriever, document_chainMCQ)
                retrieval_chainOpenEnded = create_retrieval_chain(retriever, document_chainOpenEnded)
                retrieval_chainTF = create_retrieval_chain(retriever, document_chainTF)

                quiz_questions = []
                for _ in range(num_quizzes):
                    quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainMCQ, retrieval_chainOpenEnded, retrieval_chainTF))

                st.session_state.quiz_questions = quiz_questions
                st.session_state.quiz_created = True
                st.success('퀴즈 생성이 완료되었습니다!')
                st.write(quiz_questions)

    if st.session_state.quiz_created:
        if st.button('퀴즈 제출하기'):
            db = connect_db()
            collection = db.quizzes
            quiz_data = {
                'quiz_type': quiz_type,
                'num_quizzes': num_quizzes,
                'quiz_questions': st.session_state.quiz_questions
            }
            result = collection.insert_one(quiz_data)
            st.session_state.quiz_created = False
            st.success(f'퀴즈가 성공적으로 제출되었습니다! (ID: {result.inserted_id})')
