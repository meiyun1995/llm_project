import os
import tempfile
import streamlit as st

# Langchain imports
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings

from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Local dir to downloaded GPT4All models https://gpt4all.io/index.html
PATH = './gpt4all/ggml-model-gpt4all-falcon-q4_0.bin'

# Init LLM model and embedding
llm = GPT4All(model=PATH, verbose=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title('Generate Cover letter')

# Process the annual report you wish to analyse
st.header('Single File Upload')
uploaded_file = st.file_uploader('Upload a file',
                                 accept_multiple_files=False, 
                                 type=['pdf']
                                 )

temp_file_path = os.getcwd()
while uploaded_file is None:
    x = 1
        
if uploaded_file:
    # Save the uploaded file to a temporary location
    temp_dir = tempfile.TemporaryDirectory()
    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_file_path, "wb") as temp_file:
        temp_file.write(uploaded_file.read())

    loader = PyPDFLoader(temp_file_path)
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(pages)

    # Vector store
    db = Chroma.from_documents(texts, embeddings, collection_name='resume', persist_directory='db')

    prompt_template = """
    Develop a cover letter in a professional tone for a {position_title} position at {company_name},\n
    highlighting my accomplishments and how you can contribute to the company's goals.
    """

    position_title = st.text_input('What is the title of the job?')
    company_name = st.text_input("What is the company's name")

    prompt = PromptTemplate(
        input_variables=["position_title", "company_name"], 
        template=prompt_template
        )
    
    prompt.format(position_title=position_title, company_name=company_name)
    
    chain_type_kwargs = {"prompt": prompt}

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs =chain_type_kwargs,
        return_source_documents=True,
        verbose=False,
        
    )

    if position_title and company_name:
        response = qa(prompt)
        st.write(response['result']) 