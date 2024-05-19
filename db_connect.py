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

# Define function to process user input and retrieve results using RAG model
def retrieve_results(user_query):
    # Create MongoDB Atlas Vector Search
    vector_search = MongoDBAtlasVectorSearch.from_connection_string(
        "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0",
        "sample_mflix.embedded_movies",
        OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
        index_name="vector_index"
    )

    # Define prompt template
    prompt_template = PromptTemplate.from_template(
        "{input}, Please answer in KOREAN."
        "CONTEXT:"
        "{context}."
        "FORMAT:"
        "{format}"
    )

    # Parse input format
    parser = PydanticOutputParser(pydantic_object=QuizInput)
    prompt = prompt_template.partial(format=parser.get_format_instructions())

    # Generate prompt with user input
    response = vector_search.invoke({"input": prompt.format(input=user_query)})

    return response
