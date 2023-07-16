import os
import streamlit as st
from langchain import PromptTemplate, LLMChain
from langchain.llms import GPT4All

from langchain.document_loaders import PyPDFLoader

PATH = '/Users/chuameiyun/gpt4all/ggml-model-gpt4all-falcon-q4_0.bin'

llm = GPT4All(model=PATH, verbose=True)
st.title('💵 Your Investment Guide')

prompt = PromptTemplate(input_variables=['question'],
                        template = """Question: {question}

                        A: Let's think step by step
                        """
                        )

chain = LLMChain(prompt=prompt, llm=llm)

prompt = st.text_input('Input your prompt here')

if prompt:
    response = chain.run(prompt)

    st.write(response)