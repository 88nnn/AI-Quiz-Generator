#quiz_creation_page.py

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.prompts.prompt import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain import hub
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders.image import UnstructuredImageLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader
import io
from langchain_community.document_loaders.recursive_url_loader import RecursiveUrlLoader
import chardet
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from pymongo import MongoClient
import pymongo

def connect_db():
    client = MongoClient('mongodb://username:password@host:port/')
    return client['your_database_name']

def insert_documents(collection_name, documents):
    db = connect_db()
    collection = db[collection_name]
    collection.insert_many(documents)

def vectorize_and_store(data, collection_name):
    embeddings = OpenAIEmbeddings(api_key='your_openai_api_key')
    vector_operations = []

    for document in data:
        text = document['text']
        vector = embeddings.embed_text(text)
        operation = UpdateOne({'_id': document['_id']}, {'$set': {'vector': vector.tolist()}})
        vector_operations.append(operation)

    db = connect_db()
    collection = db[collection_name]
    collection.bulk_write(vector_operations)

def search_vectors(collection_name, query_vector, top_k=10):
    db = connect_db()
    collection = db[collection_name]
    results = collection.aggregate([
        {
            '$search': {
                'vector': {
                    'query': query_vector,
                    'path': 'vector',
                    'cosineSimilarity': True,
                    'topK': top_k
                }
            }
        }
    ])
    return list(results)

def retrieve_results(user_query):
    # Create MongoDB Atlas Vector Search instance
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://username:password@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=YourApp",
        "database.collection",
        OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
        index_name="vector_index"
    )

    # Perform vector search based on user input
    response = vector_search.similarity_search_with_score(
        input=user_query, k=5, pre_filter={"page": {"$eq": 1}}
    )

    # Check if any results are found
    if not response:
        return None

    return response


examples = [
    {
        "Question": "Create one multiple-choice question focusing on important concepts, following the given format, referring to the following context, Please answer in KOREAN.",

        "CONTEXT": """
        {context}
        """,

        "FORMAT": """
        {
            plusQA: str = Field(description="The plus question and answer")
            quiz: str = Field(description="The created problem")
            options1: str = Field(description="The first option of the created problem")
            options2: str = Field(description="The second option of the created problem")
            options3: str = Field(description="The third option of the created problem")
            options4: str = Field(description="The fourth option of the created problem")
            correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")
        }
        """,

        "answer": """
{"plusQA": "추가 질문: 이 자료는 수학, 문학, 비문학, 과학 중 어느 종류야? 
중간 답변: 이 자료는 수학, 분야는 미적분입니다.
추가 질문: 미적분에 관한 format에 맞는 다양한 multiple-choice 문제를 생성합니다.", "quiz": "실수 전체의 집합에서 연속인 함수 f{\left(x \\right)}\가 모든 실수 x 대하여 f{\left(x \\right)}\ ≥ 0이고, x < 0일 때 f{\left(x \\right)}\ = (-x)*($e$^((x)^))이다.
모든 양수 t에 대하여 x에 대한 방정식 f{\letf(x \\right)}\ = t의 서로 다른
실근의 개수는 2이고, 이 방정식의 두 실근 중 작은 값을 g{\left(t \\right)}\,
큰 값을 h{\left(t \\right)}\라 하자.
두 함수 g{\left(t \\right)}\, h{\left(t \\right)}\는 모든 양수 t에 대하여
2g{\left(t \\right)}\ + h{\left(t \\right)}\ = k (k는 상수)
를 만족시킨다. \int\limits_{0}^{7} f{\left(x \\right)}\, dx = $e$() - 1일 때,f{\left(9 \\right)}\/{\left(8 \\right)}\ 의 값은?",
"options1": "1.) \\frac{3}{2}*$e$^5",
"options2": "2.) \\frac{4}{3}*$e$^7",
"options3": "3.) \\frac{5}{4}*$e$^9",
"options4": "4.) \\frac{6}{5}*$e$^11",
"correct_answer": "options4"}
""",
    },
    {
        "Question": "Create one open-ended question focusing on important concepts, following the given format, referring to the following context",

        "CONTEXT": """
        {context}
        """,

        "FORMAT": """
        {
            plusQA: str = Field(description="The plus question and answer")
            quiz: str = Field(description="The created problem")
            correct_answer: str = Field(description="correct_answer =The answer to the problem")
        }
        """,

        "answer": """
{"plusQA": "추가 질문: 이 자료는 수학, 문학, 비문학, 과학 중 어느 종류야?
중간 답변: 이 자료는 수학, 분야는 기하 입니다.
추가 질문: 미적분에 관한 format에 맞는 다양한 open-ended 문제를 생성합니다.",
"quiz": "좌표평면에 한 변의 길이가 4인 정삼각형 ABC가 있다. 선분 AB를 1 : 3으로 내분하는 점을 D , 선분 BC를 1 : 3으로 내분하는 점을 E, 선분 CA 를 1 : 3으로 내분하는 점을 F라 하자. 네 점 P , Q , R, X가 다음 조건을 만족시킨다. \"(가)  \left|\overset{\\rightarrow}{DP}\\right| = \left|\overset{\\rightarrow}{EQ}\\right = \left\overset{\\rightarrow}{FR}\\right = 1\" \"(나) \overset{\\rightarrow}{AX} = \overset{\\rightarrow}{PB} + \overset{\\rightarrow}{QC} + \overset{\\rightarrow}{RA}\" |[A,X]| 의 값이 최대일 때, 삼각형 PQR의 넓이를 S라 하자. 16S^2 의 값을 구하시오,
"correct_answer": "147"}
""",
    },
    {
        "Question": "Create one true or false question focusing on important concepts, following the given format, referring to the following context",

        "CONTEXT": """
        {context}
        """,

        "FORMAT": """
        {
            plusQA: str = Field(description="The plus question and answer")
            quiz: str = Field(description="The created problem")
            options1: str = Field(description="The true or false option of the created problem")
            options2: str = Field(description="The true or false option of the created problem")
            correct_answer: str = Field(description="One of the options1 or options2")
        }
        """,

        "answer": """
{"plusQA": "추가 질문: 이 자료는 수학, 문학, 비문학, 과학 중 어느 종류야?
중간 답변: 이 자료는 수학, 분야는 다항식입니다.
추가 질문: 미적분에 관한 format에 맞는 다양한 true or false 문제를 생성합니다.",
"quiz": " 다항식의 덧셈이나 뺄셈을 계산할 때에는 계수가 같은 문자의 차수를 더하거나 뺀 후 정리하면 된다. 이 문장은 참인가 거짓인가?",
"options1": "1.) 참",
"options2": "2.) 거짓",
"correct_answer": "options2"}
""",
    },
]

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


def process_text(text_area_content):
    text_content = st.text_area("텍스트를 입력하세요.")

    return text_content

# 파일 처리 함수
def process_file(uploaded_file, upload_option):

    uploaded_file = None
    text_area_content = None
    url_area_content = None
    selected_topic = None
    
    # # 파일 업로드 옵션 선택
    # upload_option = st.radio("입력 유형을 선택하세요", ("이미지 파일", "PDF 파일", "직접 입력", "URL", "토픽 선택"))

    # 선택된 옵션에 따라 입력 방식 제공
    if upload_option == "텍스트 파일":
        uploaded_file = st.file_uploader("텍스트 파일을 업로드하세요.", type=["txt"])
    elif upload_option == "이미지 파일":
        uploaded_file = st.file_uploader("이미지 파일을 업로드하세요.", type=["jpg", "jpeg", "png"])
    elif upload_option == "PDF 파일":
        uploaded_file = st.file_uploader("PDF 파일을 업로드하세요.", type=["pdf"])
    else:
        uploaded_file = None

    # 업로드된 파일 처리
    if uploaded_file is None:
        st.warning("파일을 업로드하세요.")
        return None

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
    if text_area_content is not None:
        text_content = process_file(uploaded_file, text_area_content) #?
    texts = text_splitter.create_documents([text_content])
    return texts

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

            # 파일 업로드 옵션 선택
            upload_option = st.radio("입력 유형을 선택하세요", ("PDF 파일", "텍스트 파일", "URL", "토픽 선택"))

            # 파일 업로드 옵션
            st.header("파일 업로드")
            uploaded_file = None
            text_content = None
            topic = None
            #uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])

            # if upload_option == "직접 입력":               
            #     text_input = st.text_area("텍스트를 입력하세요.")
            #     st.write(text_input)
                # text_content = text_input.load().encoding("utf-8", errors='ignore')
                
                # result = chardet.detect(text_input)
                # encoding = result['encoding']
                # text_content = text_input.decode(encoding)
          
                # try:
                #     text_content = text_input.encoding("utf-8")
                # except UnicodeDecodeError:
                #     # 오류 처리 코드 작성
                #     text_content = text_input.encoding("utf-8")

            
            if upload_option == "토픽 선택":
                topic = st.selectbox(
                   "토픽을 선택하세요",
                   ("수학", "문학", "비문학", "과학"),
                   index=None,
                   placeholder="토픽을 선택하세요",
                ) 

            elif upload_option == "URL":
                url_area_content = st.text_area("URL을 입력하세요.")
                loader = RecursiveUrlLoader(url=url_area_content)
                text_content = loader.load()
                
            else:
                text_content = process_file(uploaded_file, upload_option)
            

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
                        st.session_state['quiz_created'] = True

                if st.session_state.get('quiz_created', False):
                    if st.button('퀴즈 풀기'):
                        st.switch_page("pages/quiz_solve_page.py")

            elif topic is not None:
                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                        llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
                        embeddings = OpenAIEmbeddings()

                        # Rag
                        text_splitter = RecursiveCharacterTextSplitter()
                        # documents = text_splitter.split_documents(text_content)
                        # vector = FAISS.from_documents(documents, embeddings)

                        # # PydanticOutputParser 생성
                        # parseroub = PydanticOutputParser(pydantic_object=CreateQuizoub)
                        # parsersub = PydanticOutputParser(pydantic_object=CreateQuizsub)
                        # parsertf = PydanticOutputParser(pydantic_object=CreateQuizTF)

                        # prompt = PromptTemplate.from_template(
                        #     "{input}, Please answer in KOREAN."

                        #     "CONTEXT:"
                        #     "{context}."

                        #     "FORMAT:"
                        #     "{format}"
                        # )
                        # promptoub = prompt.partial(format=parseroub.get_format_instructions())
                        # promptsub = prompt.partial(format=parsersub.get_format_instructions())
                        # prompttf = prompt.partial(format=parsertf.get_format_instructions())

                        # document_chainoub = create_stuff_documents_chain(llm, promptoub)
                        # document_chainsub = create_stuff_documents_chain(llm, promptsub)
                        # document_chaintf = create_stuff_documents_chain(llm, prompttf)

                        # retriever = vector.as_retriever()

                        # retrieval_chainoub = create_retrieval_chain(retriever, document_chainoub)
                        # retrieval_chainsub = create_retrieval_chain(retriever, document_chainsub)
                        # retrieval_chaintf = create_retrieval_chain(retriever, document_chaintf)

                        # for i in range(num_quizzes):
                        #     quiz_questions.append(generate_quiz(quiz_type, text_content, retrieval_chainoub, retrieval_chainsub,retrieval_chaintf))
                        #     st.session_state['quizs'] = quiz_questions
                        # st.session_state.selected_page = "퀴즈 풀이"
                        # st.session_state.selected_type = quiz_type
                        # st.session_state.selected_num = num_quizzes

                        st.success('퀴즈 생성이 완료되었습니다!')
                        st.write(quiz_questions)
                        st.session_state['quiz_created'] = True

                if st.session_state.get('quiz_created', False):
                    if st.button('퀴즈 풀기'):
                        st.switch_page("pages/quiz_solve_page.py")


if __name__ == "__main__":
    quiz_creation_page()
