# db_connect.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chains import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# Define Pydantic model for quiz input
class QuizInput(BaseModel):
    input_text: str = Field(description="사용자 입력 텍스트")

# Define function to process user input and retrieve results using RAG model
def retrieve_results(user_query):
    # Preprocess user input
    text_splitter = RecursiveCharacterTextSplitter()
    documents = text_splitter.split_documents([{"page_content": user_query}])

    # Create RAG chain
    embeddings = OpenAIEmbeddings(model="gpt-3.5-turbo-0125")
    vector = FAISS.from_documents(documents, embeddings)
    document_chain = create_stuff_documents_chain(embeddings)
    retriever = vector.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

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
    response = retrieval_chain.invoke({"input": prompt.format(input=user_query)})

    return response
