import time

def astra_vecsearch():
  st.header("아스트라 db 검색")
  query = st.text_input("검색어를 입력해주세요: ")
  if query is null: 
    query = "What were the compute requirements for training GPT 4"
    
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
  loader = PyPDFLoader("https://arxiv.org/pdf/2303.08774.pdf")
  data = loader.load()
  docs = text_splitter.split_documents(data)
  atlas_collection = "atlas_collection"
    
  ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
  ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
    
  vector_search = AstraDBVectorStore.from_documents(
    documents=docs,
    embedding=OpenAIEmbeddings(disallowed_special=()),
    collection_name=atlas_collection,
    api_endpoint=ASTRA_DB_API_ENDPOINT,
    token=ASTRA_DB_APPLICATION_TOKEN,
  )
    
  results = vector_search.similarity_search(query) 
  #print("result: ", results)
  #st,spinner() #뭐였지
  # 방법 1 progress bar 
  latest_iteration = st.empty()
  bar = st.progress(0)
  
  for i in range(100):
  # Update the progress bar with each iteration.
    latest_iteration.text(f'Iteration {i+1}')
    bar.progress(i + 1)
    time.sleep(0.05)
  # 0.05 초 마다 1씩증가
  
  st.balloons()
# 시간 다 되면 풍선 이펙트 보여주기 
  
  st.write(results)
