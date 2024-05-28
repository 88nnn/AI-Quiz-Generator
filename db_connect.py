
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pymongo, pprint
import langchain_mongodb
import langchain_openai
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo.server_api import ServerApi


# MongoDB URI
uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new MongoDB client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Database and collection references
db = client["db1"]
collection = db["PythonDatascienceinterview"]

# Load the PDF


# Define Pydantic models for quiz creation
class CreateQuizMC(BaseModel, ans):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
        if ans = 1:
            correct_answer: str = Field(description="One of the options1 or options2 or options3 or options4")
            if ans = 2:
                correct_answer: str = Field(description="Two of the options1 or options2 or options3 or options4")
                if ans = 3:
                    correct_answer: str = Field(description="Three of the options1 or options2 or options3 or options4")


class CreateQuizSubj(BaseModel):
    quiz = ("quiz =The created problem")
    correct_answer = ("correct_answer =The answer to the problem")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="True option")
    options2: str = Field(description="False option")
    correct_answer: str = Field(description="The correct answer")

# Function to retrieve results from the vector search
def retrieve_results(user_query):
    # Perform vector search based on user input
    response = vector_search.similarity_search_with_score(
        input=user_query, k=5, pre_filter={"page": {"$eq": 1}}
    )
    template = """
Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
Question: {question}
"""
custom_rag_prompt = PromptTemplate.from_template(template)

def format_docs(docs):
   return "\n\n".join(doc.page_content for doc in docs)
# Construct a chain to answer questions on your data
rag_chain = (
   { "context": retriever | format_docs, "question": RunnablePassthrough()}
   | custom_rag_prompt
   | llm
   | StrOutputParser()
)
    # Check if any results are found
question = user_query
response = rag_chain.invoke(question)
print("Question: " + question)
print("Answer: " + answer)
    if not response:
        return None

    return response

documents = retriever.get_relevant_documents(question)
print("\nSource documents:")
pprint.pprint(documents)

# Define topic lists
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

topic = [language, mathematic, social_science, natural_science, humanity, engineering, art]

# Function to select subtopics
def subtopic_select(selected_topics):
    sub_topics = []
    for topic in selected_topics:
        if topic == language:
            sub_topics.extend(language_topic)
        elif topic == mathematic:
            sub_topics.extend(mathematic_topic)
        elif topic == social_science:
            sub_topics.extend(social_science_topic)
        elif topic == natural_science:
            sub_topics.extend(natural_science_topic)
        elif topic == humanity:
            sub_topics.extend(humanity_topic)
        elif topic == engineering:
            sub_topics.extend(engineering_topic)
        elif topic == art:
            sub_topics.extend(art_topic)
    return sub_topics

# Function to create a quiz retrieval chain
def create_quiz_retrieval_chain(pages):
    # 퀴즈 유형 선택
    quiz_type = st.radio("생성할 퀴즈 유형을 선택하세요:", ["다중 선택 (객관식)", "주관식", "OX 퀴즈"])

    # 퀴즈 개수 선택
    num_quizzes = st.number_input("생성할 퀴즈의 개수를 입력하세요:", min_value=1, value=3, step=1)

    #정답 개수 선택
    ans = st.slider("생성할 퀴즈의 정답수를 입력하세요. 정답은 한 개 이상입니다:", min_value=1, value=1, step=1, max_value=3)
    st.text('제가 선택한 정답 개수는 {} 입니다.'.format(ans))
    
    # 입력 유형 선택
    upload_option = st.radio("입력 유형을 선택하세요", ("직접 입력", "PDF 파일", "토픽 선택"))
    st.header("파일 업로드")
    
    uploaded_file = None
    text_content = None
    topic = None
    
    uploaded_file = st.file_uploader("텍스트, 이미지, 또는 PDF 파일을 업로드하세요.", type=["txt", "jpg", "jpeg", "png", "pdf"])
    
    if upload_option == "직접 입력":
        text_input = st.text_area("텍스트를 입력하세요.")
        st.write(text_input)
        try:
            text_content = text_input.encode("utf-8")
        except UnicodeDecodeError:
            return None
    
    selected_topics = []
    if upload_option == "토픽 선택":
        selected_topics = st.multiselect(
            "토픽을 선택하세요",
            topic,
            index=None,
            placeholder="토픽을 선택하세요",
        )
    
    # Subtopics selection based on selected topics
    sub_topics = subtopic_select(selected_topics)

    if text_content is not None:
                if st.button('문제 생성 하기'):
                    with st.spinner('퀴즈를 생성 중입니다...'):
                            # Initialize LLM and embeddings
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    # Text splitter and document processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=20)
    documents = text_splitter.split_documents(data)
    vector = FAISS.from_documents(documents, embeddings)

    # Create PydanticOutputParser instances
    parser_mc = PydanticOutputParser(pydantic_object=CreateQuizMC)
    parser_subj = PydanticOutputParser(pydantic_object=CreateQuizSubj)
    parser_tf = PydanticOutputParser(pydantic_object=CreateQuizTF)

    # Create prompt templates
    prompt_template = PromptTemplate.from_template(
        "Question: {input}, Please answer in KOREAN."
        "CONTEXT:"
        "{context}."
        "FORMAT:"
        "{format}"
    )

    # Create partial prompts for different quiz types
    prompt_mc = prompt_template.partial(format=parser_mc.get_format_instructions())
    prompt_subj = prompt_template.partial(format=parser_subj.get_format_instructions())
    prompt_tf = prompt_template.partial(format=parser_tf.get_format_instructions())

    # Create document chains
    document_chain_mc = create_stuff_documents_chain(llm, prompt_mc)
    document_chain_subj = create_stuff_documents_chain(llm, prompt_subj)
    document_chain_tf = create_stuff_documents_chain(llm, prompt_tf)

    # Create retriever and retrieval chains
    retriever = vector.as_retriever()

    retrieval_chain_mc = create_retrieval_chain(retriever, document_chain_mc)
    retrieval_chain_subj = create_retrieval_chain(retriever, document_chain_subj)
    retrieval_chain_tf = create_retrieval_chain(retriever, document_chain_tf)

    quiz_questions = []
    for i in range(num_quizzes):
        if quiz_type == "다중 선택 (객관식)":
            quiz_questions.append(retrieval_chain_mc.run(input=text_content))
        elif quiz_type == "주관식":
            quiz_questions.append(retrieval_chain_subj.run(input=text_content))
        elif quiz_type == "OX 퀴즈":
            quiz_questions.append(retrieval_chain_tf.run(input=text_content))
    
    st.session_state['quizs'] = quiz_questions
    st.session_state.selected_page = "퀴즈 풀이"
    st.session_state.selected_type = quiz_type
    st.session_state.selected_num = num_quizzes

    st.success('퀴즈 생성이 완료되었습니다!')
    st.write(quiz_questions)
    st.session_state['quiz_created'] = True

# Main function to create the quiz creation page
def quiz_creation_page():
    st.title("Quiz Creation Page")
    create_quiz_retrieval_chain(pages=None)

