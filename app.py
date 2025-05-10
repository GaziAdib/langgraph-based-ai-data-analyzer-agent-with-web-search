import os
from typing import Dict, Any, List, Optional, Literal
import streamlit as st
import tempfile
from langchain_core.runnables import RunnablePassthrough
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
from langchain_core.prompts import ChatPromptTemplate
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





# Prompt templates
DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are an expert data analyst. Analyze these documents:

DOCUMENTS:
{documents}

INSTRUCTIONS:
1. Identify key information and patterns
2. Provide a detailed summary
3. Format in markdown with headings

ANALYSIS:
""")

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based on your knowledge.

QUESTION:
{question}

If you don't know, say "I don't know" - don't make up answers.
Format your response in markdown.

ANSWER:
""")

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
Answer using these web results:

QUESTION: {question}

RESULTS:
{search_results}

INSTRUCTIONS:
1. Synthesize the information
2. Cite sources when possible
3. If results don't help, say so

ANSWER:
""")





# Creating Nodes


def analyse_documents(state: Dict[str, any]) -> Dict[str, Any]:
    """Node for document analysis."""
    documents = state.get('documents', '')

    if not documents:
        return {"analysis": "No documents provided"}
    
    chain = (
        {'documents': RunnablePassthrough()}

        | DOCUMENT_ANALYSIS_PROMPT
        | llm
        | StrOutputParser()
    )

    analysis = chain.invoke(documents)
    return {"analysis": analysis}


def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node for answering from LLM's knowledge."""
    question = state.get("question", "")
    
    chain = (
        {"question": RunnablePassthrough()}
        | DIRECT_ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    
    answer = chain.invoke(question)
    return {"answer": answer}



def answer_from_web(state: Dict[str, any]) -> Dict[str, any]:
    """Node for answering from web search"""
    question = state.get('question', '')

    search_results = search_tool.invoke({'query': question})

    chain = (
        {
            "question": RunnablePassthrough(),
            "search_results": RunnablePassthrough()
        }
        | WEB_SEARCH_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = chain.invoke({
        "question": question,
        "search_results": search_results
    })
    return {"answer": answer}



def router(state: Dict[str, any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
    """Router node to decide the flow."""
    if state.get("documents") and not state.get("question"):
        return 'analyze_documents'
    elif state.get('question'):
        if "don't know" in state.get('answer', '').lower():
            return "answer_from_web"
        return "end"
    else:
        return "end"












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