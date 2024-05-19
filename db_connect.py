# db_connect.py

from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_openai import OpenAIEmbeddings
from langchain.chains import combine_documents
from langchain.chains import create_retrieval_chain
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define Pydantic model for quiz input
class QuizInput(BaseModel):
    input_text: str = Field(description="사용자 입력 텍스트")

def retrieve_results(user_query):
    # MongoDB Atlas Vector Search 생성
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        "sample_mflix.embedded_movies",
        OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
        index_name="vector_index"
    )

    # 사용자 입력에 대한 프롬프트 템플릿 정의
    prompt_template = PromptTemplate.from_template(
        "{input}, Answer please."
        "context:"
        "{context}."
        "topic:"
        "{topic}."
        "characteristics:"
        "{characteristics}."
        "format:"
        "{format}"
    )

    # 프롬프트 생성
    prompt = prompt_template.partial(format="퀴즈 답변 형식을 지정하세요")

    # 사용자 입력과 (벡터 유사도)를 고려하여 검색
    #response = vector_search.invoke({"input": prompt.format(input=user_query)})
    response = vector_search.similarity_search_with_score(
    input=user_query, k=5, pre_filter={"page": {"$eq": 1}})

    return response
