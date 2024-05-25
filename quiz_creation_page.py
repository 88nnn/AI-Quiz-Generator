import streamlit as st
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from googletrans import Translator
from pydantic import BaseModel, Field
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS

# MongoDB URI
uri = "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp"

# Create a new MongoDB client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Initialize LLM and embeddings
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
embeddings = OpenAIEmbeddings()

# Create MongoDB Atlas Vector Search instance
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=uri,
    namespace="langchain_db.test",
    embedding=embeddings,
    index_name="vector_index_test"
)

# Initialize Google Translator
translator = Translator()

# Define the Pydantic models for the output parser
class CreateQuizMCQ(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")

class CreateQuizOpen(BaseModel):
    quiz: str = Field(description="The created problem")
    correct_answer: str = Field(description="The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The true or false option of the created problem")
    options2: str = Field(description="The true or false option of the created problem")
    correct_answer: str = Field(description="One of the options1 or options2")

# Initialize the output parsers
parser_mcq = PydanticOutputParser(pydantic_object=CreateQuizMCQ)
parser_open = PydanticOutputParser(pydantic_object=CreateQuizOpen)
parser_tf = PydanticOutputParser(pydantic_object=CreateQuizTF)

# Function to format documents into a single string
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Define the RAG prompt template
template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

# Construct a chain to answer questions using the retriever, formatter, and LLM
rag_chain = (
    {"context": RunnablePassthrough() | format_docs, "question": RunnablePassthrough()}
    | custom_rag_prompt
    | llm
    | output_parser
)

# Function to process text from various file types
def process_file(uploaded_file):
    if uploaded_file.type == "text/plain":
        text_content = uploaded_file.read().decode("utf-8")
    elif uploaded_file.type.startswith("image/"):
        image = Image.open(uploaded_file)
        text_content = pytesseract.image_to_string(image)
    elif uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(io.BytesIO(uploaded_file.read()))
        text_content = "".join([page.extract_text() for page in pdf_reader.pages])
    else:
        st.error("지원하지 않는 파일 형식입니다.")
        return None

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
    texts = text_splitter.create_documents([text_content])
    return texts

# Function to generate quiz based on quiz type
def generate_quiz(quiz_type, text_content, rag_chain):
    if quiz_type == "다중 선택 (객관식)":
        response = rag_chain.invoke({
            "context": text_content,
            "question": "Create one multiple-choice question focusing on important concepts."
        })
    elif quiz_type == "주관식":
        response = rag_chain.invoke({
            "context": text_content,
            "question": "Create one open-ended question focusing on important concepts."
        })
    elif quiz_type == "OX 퀴즈":
        response = rag_chain.invoke({
            "context": text_content,
            "question": "Create one true or false question focusing on important concepts."
        })
    return response

# Main function
def quiz_creation_page():
    st.title("AI 퀴즈 생성기")

    # 퀴즈 유형 선택
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

    # 퀴즈 개수 선택
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=5, step=1)

    # 파일 업로드 옵션 선택
    upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "이미지 파일"))

    # 파일 업로드
    uploaded_file = st.file_uploader("파일을 업로드하세요.", type=["txt", "pdf", "jpg", "jpeg", "png"])

    if uploaded_file:
        text_content = process_file(uploaded_file)

        if st.button('문제 생성 하기'):
            with st.spinner('퀴즈를 생성 중입니다...'):
                quiz_questions = []
                for _ in range(num_quizzes):
                    quiz_questions.append(generate_quiz(quiz_type, text_content, rag_chain))
                st.success('퀴즈 생성이 완료되었습니다!')
                st.write(quiz_questions)

if __name__ == "__main__":
    quiz_creation_page()
