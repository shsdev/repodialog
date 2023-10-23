#!/usr/bin/python
# -*- coding: utf-8 -*-
import os
import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.document_loaders import TextLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.schema import Document

from dotenv import load_dotenv

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')

st.set_page_config(layout="wide")

with st.sidebar:
    st.title('Repodialog')

llm = HuggingFaceHub(
    repo_id="google/flan-t5-xxl",
    model_kwargs={"temperature": 0.5},
    huggingfacehub_api_token="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"
)

    

def main():

    st.header("File upload")
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        
        store_name = pdf.name[:-4]

        if os.path.exists(f"{store_name}.pkl"):
            with open(f"{store_name}.pkl", "rb") as f:
                VectorStore = pickle.load(f)
            st.write('Embeddings loaded from disk')
        else: 

            embeddings = HuggingFaceEmbeddings()


            pdf_reader = PdfReader(pdf)

            pages = pdf_reader.pages
            num_pages = len(pages)
            text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500,
                        chunk_overlap=60,
                        length_function=len
            )
            list_of_documents = []
            for i in range(0, num_pages):
                pagenum = i+1
                page = pages[i]
                chunks = text_splitter.split_text(text=page.extract_text())
                for chunk in chunks:
                    list_of_documents.append(Document(page_content=chunk, metadata=dict(page=pagenum, filename=pdf.name)))
            VectorStore = FAISS.from_documents(embedding=embeddings, documents=list_of_documents)

            with open(f"{store_name}.pkl", "wb") as f:
                pickle.dump(VectorStore, f)
            st.write('Embeddings computation completed')

        query = st.text_input("Ask questions")
        
        if query:            
            chain = load_qa_chain(llm, chain_type="stuff")
            
            docs = VectorStore.similarity_search(query)
           
            answer = chain.run(input_documents=docs, question=query)
            st.write(answer)
           
            st.write(docs)
        

if __name__ == '__main__':
    main()