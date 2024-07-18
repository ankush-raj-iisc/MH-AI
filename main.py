# -*- coding: utf-8 -*-
"""main.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1baeIM5oUAY3zf69g1lq18SYxaPO9FtSq
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

# Set OpenAI API key
os.environ["OPENAI_API_KEY"]="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"

class Query(BaseModel):
    question: str

app = FastAPI()

embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
db3 = Chroma(persist_directory="./embeddings", embedding_function=embeddings)

instruct_template = '''
You are a lawyer specialized in Indian law related to insider trading and SEBI.
Answer the question based on the following context.
The context you are given can be the snippet of the cases/judgements/ or the law itself.
Provide your answer based on the below context only.
{context}

Question : {question}
'''
PROMPT = PromptTemplate(template=instruct_template, input_variables=['context', 'question'])

llm = ChatOpenAI(temperature=0.9, model_name='gpt-3.5-turbo')
qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type='stuff',
    retriever=db3.as_retriever(search_kwargs={'k': 5}),
    chain_type_kwargs={"prompt": PROMPT},
)

@app.post("/query/")
async def get_answer(query: Query):
    try:
        print("Received Query:", query)
        result = qa.invoke({"query": query.question})
        print("QA Result:", result)
        return {"answer": result['result']}
    except Exception as e:
        print("Error:", str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return {"message": "Welcome to the FastAPI application!"}