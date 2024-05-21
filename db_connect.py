from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define Pydantic models for quiz creation
class CreateQuizMC(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="The first option of the created problem")
    options2: str = Field(description="The second option of the created problem")
    options3: str = Field(description="The third option of the created problem")
    options4: str = Field(description="The fourth option of the created problem")
    correct_answer: str = Field(description="The correct answer")

class CreateQuizSubj(BaseModel):
    quiz: str = Field(description="The created problem")
    correct_answer: str = Field(description="The correct answer")

class CreateQuizTF(BaseModel):
    quiz: str = Field(description="The created problem")
    options1: str = Field(description="True option")
    options2: str = Field(description="False option")
    correct_answer: str = Field(description="The correct answer")

def retrieve_results(user_query):
    # Create MongoDB Atlas Vector Search instance
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        "sample_mflix.embedded_movies",
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

def create_quiz_retrieval_chain(pages):
    llm = ChatOpenAI(model="gpt-3.5-turbo-0125")
    embeddings = OpenAIEmbeddings()

    # Text splitter and document processing
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents(pages)
    vector = FAISS.from_documents(documents, embeddings)

    # Create PydanticOutputParser instances
    parser_mc = PydanticOutputParser(pydantic_object=CreateQuizMC)
    parser_subj = PydanticOutputParser(pydantic_object=CreateQuizSubj)
    parser_tf = PydanticOutputParser(pydantic_object=CreateQuizTF)

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

    return retrieval_chain_mc, retrieval_chain_subj, retrieval_chain_tf
