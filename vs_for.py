import streamlit as st
from langchain_community.vectorstores import MongoDBAtlasVectorSearch
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnablePassthrough
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from googletrans import Translator

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

# MongoDB URI
uri = "mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"

# Create a new MongoDB client and connect to the server
client = MongoClient(uri, server_api=ServerApi('1'))

# Load the PDF
#loader = PyPDFLoader("https://query.prod.cms.rt.microsoft.com/cms/api/am/binary/RE4HkJP")
#data = loader.load()

# Create MongoDB Atlas Vector Search instance
vector_search = MongoDBAtlasVectorSearch.from_connection_string(
    connection_string=uri,
    namespace="langchain_db.test",
    embedding=OpenAIEmbeddings(model="gpt-3.5-turbo-0125"),
    index_name="vector_index_test"
)

# Initialize LLM
llm = ChatOpenAI(model="gpt-3.5-turbo-0125")

# Initialize Google Translator
translator = Translator()

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
    | PydanticOutputParser()
)

# Function to retrieve results from the vector search
def retrieve_results(user_query):
    responses = []
    languages = ["en", "ko"]

    # Perform vector search based on user input for original and translated queries
    for lang in languages:
        if lang == "en":
            translated_query = translator.translate(user_query, src='ko', dest='en').text
        else:
            translated_query = translator.translate(user_query, src='en', dest='ko').text
        
        for query in [user_query, translated_query]:
            documents = vector_search.similarity_search_with_score(
                input=query, k=5, pre_filter={"page": {"$eq": 1}}
            )

            # If no documents are found, continue to the next query
            if not documents:
                continue

            # Format the documents and run the RAG chain
            formatted_docs = format_docs(documents)
            response = rag_chain.invoke({"context": formatted_docs, "question": query})

            # Display the question and the generated answer
            st.text(f"Question ({lang}): " + query)
            st.text("Answer: " + response)

            # Collect responses
            responses.append((query, response))

    # Display the source documents for debugging purposes
    st.text("\nSource documents:")
    pprint.pprint(documents)

    return responses

# Streamlit interface
st.header("벡터 검색")
user_query = st.text_input("검색어를 입력하세요:", "")
if not user_query:
    user_query = "평점 높은 미국 영화"  # Default query

responses = retrieve_results(user_query)
st.write("Responses:", responses)
