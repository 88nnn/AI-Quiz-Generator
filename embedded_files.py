import streamlit as st
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts.prompt import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from pymongo import MongoClient
import sys

# Replace the placeholder data with your Atlas connection string. Be sure it includes
# a valid username and password! Note that in a production environment,
# you should not store your password in plain-text here.

try:
  client = pymongo.MongoClient(mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0)
  
# return a friendly error if a URI error is thrown 
except pymongo.errors.ConfigurationError:
  st.write(f"An Invalid URI host error was received. Is your Atlas host name correct in your connection string?")
  sys.exit(1)

# use a database named "myDatabase"
db = client.sample_mflix

# use a collection named "recipes"
my_collection = db["movies"]

recipe_documents = [{ "name": "elotes", "ingredients": ["corn", "mayonnaise", "cotija cheese", "sour cream", "lime"], "prep_time": 35 },
                    { "name": "loco moco", "ingredients": ["ground beef", "butter", "onion", "egg", "bread bun", "mushrooms"], "prep_time": 54 },
                    { "name": "patatas bravas", "ingredients": ["potato", "tomato", "olive oil", "onion", "garlic", "paprika"], "prep_time": 80 },
                    { "name": "fried rice", "ingredients": ["rice", "soy sauce", "egg", "onion", "pea", "carrot", "sesame oil"], "prep_time": 40 }]

# drop the collection in case it already exists
try:
  my_collection.drop()  

# return a friendly error if an authentication error is thrown
except pymongo.errors.OperationFailure:
  st.write(f"An authentication error was received. Are your username and password correct in your connection string?")
  sys.exit(1)

# INSERT DOCUMENTS
#
# You can insert individual documents using collection.insert_one().
# In this example, we're going to create four documents and then 
# insert them all with insert_many().

try: 
 result = my_collection.insert_many(recipe_documents)

# return a friendly error if the operation fails
except pymongo.errors.OperationFailure:
  st.write(f"An authentication error was received. Are you sure your database user is authorized to perform write operations?")
  sys.exit(1)
else:
  inserted_count = len(result.inserted_ids)
  st.write(f"I inserted %x documents." %(inserted_count))

  st.write(f"\n")

# FIND DOCUMENTS
#
# Now that we have data in Atlas, we can read it. To retrieve all of
# the data in a collection, we call find() with an empty filter. 

result = my_collection.find()

if result:    
  for doc in result:
    my_recipe = doc['name']
    my_ingredient_count = len(doc['ingredients'])
    my_prep_time = doc['prep_time']
    st.write(f"%s has %x ingredients and takes %x minutes to make." %(my_recipe, my_ingredient_count, my_prep_time))
    
else:
  st.write(f"No documents found.")

st.write(f"\n")

# We can also find a single document. Let's find a document
# that has the string "potato" in the ingredients list.
my_doc = my_collection.find_one({"ingredients": "potato"})

if my_doc is not None:
  st.write(f"A recipe which uses potato:")
  st.write(my_doc)
else:
  st.write(f"I didn't find any recipes that contain 'potato' as an ingredient.")
st.write(f"\n")

# UPDATE A DOCUMENT
#
# You can update a single document or multiple documents in a single call.
# 
# Here we update the prep_time value on the document we just found.
#
# Note the 'new=True' option: if omitted, find_one_and_update returns the
# original document instead of the updated one.

my_doc = my_collection.find_one_and_update({"ingredients": "potato"}, {"$set": { "prep_time": 72 }}, new=True)
if my_doc is not None:
  st.write(f"Here's the updated recipe:")
  st.write(my_doc)
else:
  st.write(f"I didn't find any recipes that contain 'potato' as an ingredient.")
st.write(f"\n")

# DELETE DOCUMENTS
#
# As with other CRUD methods, you can delete a single document 
# or all documents that match a specified filter. To delete all 
# of the documents in a collection, pass an empty filter to 
# the delete_many() method. In this example, we'll delete two of 
# the recipes.
#
# The query filter passed to delete_many uses $or to look for documents
# in which the "name" field is either "elotes" or "fried rice".

my_result = my_collection.delete_many({ "$or": [{ "name": "elotes" }, { "name": "fried rice" }]})
st.write(f"I deleted %x records." %(my_result.deleted_count))
st.write(f"\n")
"""
# 데이터베이스에서 퀴즈를 읽어오는 함수
def read_quiz_from_mongo():
    # MongoDB 연결 설정
    uri = 'mongodb+srv://acm41th:vCcYRo8b4hsWJkUj@cluster0.ctxcrvl.mongodb.net/sample_mflix?retryWrites=true&w=majority&appName=Cluster0'
    client = MongoClient(uri)
    
    # 데이터베이스 및 컬렉션 선택
    db = client.sample_mflix
    collection = db.quiz_collection  # 퀴즈를 저장하는 컬렉션 이름
    
    # MongoDB에서 퀴즈 문서들 조회
    quizzes = collection.find({})
    
    # 조회된 퀴즈 문서들 반환
    return quizzes

# 퀴즈 문서를 출력하는 함수
def display_quizzes(quizzes):
    for quiz in quizzes:
        st.write(f"Question: {quiz['question']}")
        st.write(f"Options:")
        for i, option in enumerate(quiz['options'], start=1):
            st.write(f"  {i}. {option}")
        st.write(f"Correct Answer: {quiz['correct_answer']}")
        st.write("\n")

# 메인 함수: Streamlit 앱 설정
def embedded_files():
    st.title("Quiz Display App")
    
    # MongoDB에서 퀴즈 읽어오기
    quizzes = read_quiz_from_mongo()
    
    # 퀴즈 출력
    if quizzes:
        display_quizzes(quizzes)
    else:
        st.write("No quizzes found.")
        
# 앱 실행
if __name__ == "__main__":
    embedded_files()
"""
