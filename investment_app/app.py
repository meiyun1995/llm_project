import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA


PATH = '/Users/chuameiyun/gpt4all/ggml-model-gpt4all-falcon-q4_0.bin'

llm = GPT4All(model=PATH, verbose=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

st.title('💵 Your Investment Guide')

loader = PyPDFLoader('annualreport2223.pdf')
pages = loader.load_and_split()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
texts = text_splitter.split_documents(pages)

db = Chroma.from_documents(texts, embeddings, collection_name='annualreport')

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True,
    verbose=False,
)

prompt = st.text_input('Input your prompt here')

if prompt:
    # response = chain.run(prompt)
    response = qa(prompt)

    st.write(response['result'])