import os
from typing import Dict, Any, List, Optional, Literal
import streamlit as st
import tempfile
from langgraph.graph import Graph, END
from langchain_groq import ChatGroq
from langchain_community.document_loaders import (
    CSVLoader,
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
    UnstructuredFileLoader
)
from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter

def main():
    st.set_page_config(page_title="Doc Analyst", page_icon="📄")
    st.title("📄 Document Analysis Agent with LangGraph")
    st.markdown("Upload documents or ask questions - I'll analyze or find answers!")
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = ""
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, CSV, TXT)",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
    )
    
    if uploaded_files and not st.session_state.documents:
        with st.spinner("Processing documents..."):
            st.session_state.documents = process_uploaded_files(uploaded_files)
            st.success(f"Processed {len(uploaded_files)} file(s)")
    
    # Display chat
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask a question or request analysis"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Prepare workflow state
                state = {
                    "documents": st.session_state.documents,
                    "question": prompt if prompt else None,
                }
                
                # Execute workflow
                result = workflow.invoke(state)
                
                # Display appropriate response
                if "analysis" in result:
                    response = result["analysis"]
                elif "answer" in result:
                    response = result["answer"]
                else:
                    response = "I couldn't generate a response."
                
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()