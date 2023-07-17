import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import Chroma

from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo
)


PATH = '/Users/chuameiyun/gpt4all/ggml-model-gpt4all-falcon-q4_0.bin'

llm = GPT4All(model=PATH, verbose=True)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
st.title('💵 Your Investment Guide')

loader = PyPDFLoader('annualreport2223.pdf')
pages = loader.load_and_split()
store = Chroma.from_documents(pages, embeddings, collection_name='annualreport')

vectorstore_info = VectorStoreInfo(
    name='annual_report',
    description='a banking annual report as pdf',
    vectorstore=store
)

toolkit = VectorStoreToolkit(vectorstore_info=vectorstore_info)

agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True
)

# prompt = PromptTemplate(input_variables=['question'],
#                         template = """Question: {question}

#                         A: Let's think step by step
#                         """
#                         )

# chain = LLMChain(prompt=prompt, llm=llm)

prompt = st.text_input('Input your prompt here')

if prompt:
    # response = chain.run(prompt)
    response = agent_executor.run(prompt)

    st.write(response)

    with st.expander('Document Similarity Search'):
        search = store.similarity_search_with_score(prompt)
        st.write(search[0][0].page_content)