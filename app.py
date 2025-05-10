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


# Configuration
MODEL_NAME = "mixtral-8x7b-32768"
TEMPERATURE = 0.1
MAX_TOKENS = 500
TAVILY_MAX_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 500


@st.cache_resource
def initialize_components():
    llm = ChatGroq(
        temperature=TEMPERATURE,
        model=MODEL_NAME,
        max_tokens=MAX_TOKENS,
        api_key=os.getenv('GROQ_API_KEY')
    )

    search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)

    return search_tool, llm

llm, search_tool = initialize_components()




def process_uploaded_files(uploaded_files: List) -> str:
    """Process uploaded files and return concatenated text."""
    full_text = ""
    
    for uploaded_file in uploaded_files:
        file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        try:
            if file_ext == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
            elif file_ext == '.csv':
                loader = CSVLoader(tmp_file_path)
            elif file_ext == '.docx':
                loader = Docx2txtLoader(tmp_file_path)
            elif file_ext == '.txt':
                loader = TextLoader(tmp_file_path)
            else:
                loader = UnstructuredFileLoader(tmp_file_path)
            
            docs = loader.load()
            full_text += "\n\n".join([doc.page_content for doc in docs])
        
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            os.unlink(tmp_file_path)
    
    # Simple chunking to manage context window
    if len(full_text) > CHUNK_SIZE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = text_splitter.split_text(full_text)
        return "\n\n[Continued...]\n\n".join(chunks[:3])  # Limit to first 3 chunks
    return full_text



# Creating Nodes









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