import os
from typing import Dict, Any, List, Literal
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

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

st.set_page_config(page_title="Doc Analyst", page_icon="📄")
st.title("📄 Document Analysis Agent")


# Configuration
# llama3-70b-8192
MODEL_NAME = "llama3-70b-8192"
TEMPERATURE = 0.1
MAX_TOKENS = 100
TAVILY_MAX_RESULTS = 3
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

@st.cache_resource
def initialize_components():
    """Initialize components with proper error handling."""
    try:
        llm = ChatGroq(
            api_key=os.getenv('GROQ_API_KEY'), 
            temperature=TEMPERATURE,
            model_name=MODEL_NAME,
            max_tokens=MAX_TOKENS,
        )
        search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
        return llm, search_tool
    except Exception as e:
        st.error(f"Failed to initialize components: {str(e)}")
        return None, None

llm, search_tool = initialize_components()

def process_uploaded_files(uploaded_files: List) -> str:
    """Process uploaded files with improved error handling and comprehensive text extraction."""
    if not uploaded_files:
        return ""
    
    full_text = ""
    file_summaries = []
    
    for uploaded_file in uploaded_files:
        file_name = uploaded_file.name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            file_summary = f"--- File: {file_name} ---\n"
            
            if file_ext == '.pdf':
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()
                file_text = "\n\n".join([f"Page {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
            elif file_ext == '.csv':
                loader = CSVLoader(tmp_file_path)
                docs = loader.load()
                file_text = "\n\n".join([doc.page_content for doc in docs])
                # Add CSV structure info
                file_summary += f"CSV with approximately {len(docs)} rows\n"
            elif file_ext == '.docx':
                loader = Docx2txtLoader(tmp_file_path)
                docs = loader.load()
                file_text = "\n\n".join([doc.page_content for doc in docs])
            elif file_ext == '.txt':
                loader = TextLoader(tmp_file_path)
                docs = loader.load()
                file_text = "\n\n".join([doc.page_content for doc in docs])
            else:
                loader = UnstructuredFileLoader(tmp_file_path)
                docs = loader.load()
                file_text = "\n\n".join([doc.page_content for doc in docs])
            
            # Add content summary
            file_summary += f"Character count: {len(file_text)}\n"
            file_summary += f"Word count: approximately {len(file_text.split())}\n"
            file_summary += "---\n\n"
            
            file_summaries.append(file_summary)
            full_text += file_summary + file_text + "\n\n"
        
        except Exception as e:
            error_msg = f"Error processing {file_name}: {str(e)}"
            st.error(error_msg)
            file_summaries.append(f"--- File: {file_name} ---\nERROR: {str(e)}\n---\n\n")
        finally:
            if 'tmp_file_path' in locals():
                try:
                    os.unlink(tmp_file_path)
                except:
                    pass
    
    # Split if too large but preserve structure
    if len(full_text) > CHUNK_SIZE:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(full_text)
        
        # Always include file summaries at the beginning
        summary_section = "\n".join(file_summaries)
        processed_text = summary_section + "\n\n" + "\n\n[CHUNK BREAK]\n\n".join(chunks[:5])
        
        # Add note about truncation
        if len(chunks) > 5:
            processed_text += f"\n\n[NOTE: Document was truncated. {len(chunks)-5} more chunks available but not shown due to size constraints.]"
        
        return processed_text
    
    return full_text

# Prompt templates
DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
You are a document analysis expert. Analyze these documents thoroughly:

DOCUMENTS:
{documents}

Please provide a comprehensive analysis including:
1. A summary of each document identified 
2. Key insights and main points
3. Important data points, facts, or statistics
4. Relationships between documents (if multiple)
5. Potential implications or conclusions

Format your response in clear markdown with appropriate headers, bullet points, and emphasis.
If the document is technical, identify specialized terminology and explain it.
If data is present, summarize the key trends or findings.

Be thorough but concise. Focus on extracting maximum value from the provided content.
""")

DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based on your knowledge:

QUESTION:
{question}

If unsure, say "I don't know". Format in markdown.
""")

WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
Answer using these web results:

QUESTION: {question}

RESULTS:
{search_results}

Synthesize the information and cite sources if available.
""")

# LangGraph nodes
def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node for document analysis with enhanced functionality."""
    # Check if documents is a string or dict
    if isinstance(state, dict):
        documents = state.get("documents", "")
    else:
        # If state itself is a string (documents content)
        documents = state if state else ""
    
    if not documents:
        return {"analysis": "No documents provided for analysis."}
    
    # For debugging
    print(f"Document analysis called with content length: {len(documents)}")
    print(f"First 100 characters: {documents[:100]}")
    
    # Increase token limit for document analysis
    analysis_llm = ChatGroq(
        api_key=os.getenv('GROQ_API_KEY'), 
        temperature=0.2,  # Lower temperature for more factual analysis
        model_name=MODEL_NAME,
        max_tokens=1000,  # Increased for more comprehensive analysis
    )
    
    try:
        # Format the prompt directly here for better control
        formatted_prompt = f"""
You are a document analysis expert. Analyze these documents thoroughly:

DOCUMENTS:
{documents}

Please provide a comprehensive analysis including:
1. A summary of each document identified 
2. Key insights and main points
3. Important data points, facts, or statistics
4. Relationships between documents (if multiple)
5. Potential implications or conclusions

Format your response in clear markdown with appropriate headers, bullet points, and emphasis.
If the document is technical, identify specialized terminology and explain it.
If data is present, summarize the key trends or findings.

Be thorough but concise. Focus on extracting maximum value from the provided content.
"""
        
        # Create messages for the ChatGroq model
        messages = [{"role": "user", "content": formatted_prompt}]
        
        # Get response from LLM
        response = analysis_llm.invoke(messages)
        analysis = response.content
        
        # Add a helpful note about the analysis
        analysis += "\n\n---\n*Note: This analysis is based on the document content that was successfully processed. If you need more specific information, ask follow-up questions about particular aspects of the document.*"
        
        return {"analysis": analysis}
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        return {"analysis": f"Error analyzing documents: {str(e)}\n\nThe documents were loaded but could not be analyzed. You can try asking specific questions about the documents instead.\n\nError details: {error_details}"}

def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node for answering from LLM's knowledge."""
    # Check if state is a dictionary or directly the question
    if isinstance(state, dict):
        question = state.get("question", "")
    else:
        # If state itself is a string (question content)
        question = state if state else ""
    
    chain = (
        {"question": RunnablePassthrough()}
        | DIRECT_ANSWER_PROMPT
        | llm
        | StrOutputParser()
    )
    return {"answer": chain.invoke(question)}

def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
    """Node for answering from web search."""
    # Check if state is a dictionary or directly the question
    if isinstance(state, dict):
        question = state.get("question", "")
    else:
        # If state itself is a string (question content)
        question = state if state else ""
        
    try:
        search_results = search_tool.invoke({"query": question})
        chain = (
            {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
            | WEB_SEARCH_PROMPT
            | llm
            | StrOutputParser()
        )
        return {"answer": chain.invoke({"question": question, "search_results": search_results})}
    except Exception as e:
        return {"answer": f"Failed to perform web search: {str(e)}"}

def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
    """Router node to decide the flow."""
    # Check if state is a dictionary or a string
    if not isinstance(state, dict):
        # If state is a string, assume it's document content
        if state:
            return "analyze_documents"
        return "end"
    
    # Normal dictionary flow
    has_documents = "documents" in state and state["documents"]
    has_question = "question" in state and state["question"]
    
    if has_documents and not has_question:
        return "analyze_documents"
    elif has_question:
        has_answer = "answer" in state
        if has_answer and isinstance(state["answer"], str):
            # Check if the answer indicates uncertainty
            if any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
                return "answer_from_web"
        elif not has_answer:
            return "answer_from_knowledge"
        return "end"
    return "end"

@st.cache_resource
def create_workflow() -> Graph:
    """Create and compile the LangGraph workflow."""
    workflow = Graph()
    
    # Add all nodes first
    workflow.add_node("analyze_documents", analyze_documents)
    workflow.add_node("answer_from_knowledge", answer_from_knowledge)
    workflow.add_node("answer_from_web", answer_from_web)
    workflow.add_node("router", router)
    
    # Set entry point
    workflow.set_entry_point("router")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "router",
        router,
        {
            "analyze_documents": "analyze_documents",
            "answer_from_knowledge": "answer_from_knowledge",
            "answer_from_web": "answer_from_web",
            "end": END,
        },
    )
    
    # Add regular edges
    workflow.add_edge("analyze_documents", END)
    workflow.add_edge("answer_from_knowledge", "router")
    workflow.add_edge("answer_from_web", END)
    
    return workflow.compile()

# Initialize workflow
workflow = create_workflow()

def main():
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents" not in st.session_state:
        st.session_state.documents = ""
    if "file_info" not in st.session_state:
        st.session_state.file_info = ""
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents (PDF, DOCX, CSV, TXT)",
        type=["pdf", "docx", "csv", "txt"],
        accept_multiple_files=True,
    )
    
    # Process uploaded files if available
    if uploaded_files and not st.session_state.documents:
        with st.spinner("Processing documents..."):
            st.session_state.documents = process_uploaded_files(uploaded_files)
            st.session_state.file_info = f"Processed {len(uploaded_files)} file(s)"
            st.success(st.session_state.file_info)
    
    # Show file information and analyze button
    if st.session_state.documents:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info(f"📄 {st.session_state.file_info}")
        
        with col2:
            if st.button("Analyze Documents"):
                # Add user message
                st.session_state.messages.append({"role": "user", "content": "Please analyze these documents"})
                
                # Add assistant response
                with st.spinner("Analyzing documents... This may take a moment."):
                    try:
                        # Explicitly create document analysis request
                        result = analyze_documents(st.session_state.documents)
                        response = result.get("analysis", "No analysis could be generated.")
                        st.session_state.messages.append({"role": "assistant", "content": response})
                        
                    except Exception as e:
                        error_msg = f"An error occurred during analysis: {str(e)}"
                        st.error(error_msg)
                        st.session_state.messages.append({"role": "assistant", "content": error_msg})
                      
        
        # Add document management in sidebar
        st.sidebar.subheader("Document Management")
        if st.sidebar.button("Clear Uploaded Documents"):
            st.session_state.documents = ""
            st.session_state.file_info = ""
 

    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    
    # Chat input
    if prompt := st.chat_input("Ask a question or type 'analyze' to analyze documents"):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Check if user wants to analyze documents
                    if prompt.lower().strip() in ["analyze", "analyze documents", "summarize", "summary", "analyze document"]:
                        if st.session_state.documents:
                            # Directly analyze documents
                            result = analyze_documents(st.session_state.documents)
                            response = result.get("analysis", "No analysis could be generated.")
                        else:
                            response = "I don't see any uploaded documents to analyze. Please upload documents using the file uploader above."
                    
                    # Otherwise handle as a regular question
                    elif st.session_state.documents:
                        # Create state with both documents and question
                        state = {
                            "documents": st.session_state.documents,
                            "question": prompt
                        }
                        # Process through workflow
                        result = workflow.invoke(state)
                        response = result.get("analysis") or result.get("answer") or "I couldn't generate a response."
                    else:
                        # No documents, just answer the question
                        result = answer_from_knowledge(prompt)
                        if "I don't know" in result.get("answer", ""):
                            # Try web search if knowledge base fails
                            result = answer_from_web(prompt)
                        response = result.get("answer", "I couldn't generate a response.")
                    
                    # Display and save response
                    st.markdown(response)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                except Exception as e:
                    error_msg = f"An error occurred: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})

if __name__ == "__main__":
    if llm and search_tool:  # Only run if components initialized successfully
        main()
    else:
        st.error("Failed to initialize required components. Please check your API keys.")
















# import os
# from typing import Dict, Any, List, Literal
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="Doc Analyst", page_icon="📄")
# st.title("📄 Document Analysis Agent")


# # Configuration
# MODEL_NAME = "llama3-8b-8192"
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# @st.cache_resource
# def initialize_components():
#     """Initialize components with proper error handling."""
#     try:
#         llm = ChatGroq(
#             api_key=os.getenv('GROQ_API_KEY'), 
#             temperature=TEMPERATURE,
#             model_name=MODEL_NAME,
#             max_tokens=MAX_TOKENS,
#         )
#         search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
#         return llm, search_tool
#     except Exception as e:
#         st.error(f"Failed to initialize components: {str(e)}")
#         return None, None

# llm, search_tool = initialize_components()

# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files with improved error handling."""
#     if not uploaded_files:
#         return ""
    
#     full_text = ""
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             if 'tmp_file_path' in locals():
#                 try:
#                     os.unlink(tmp_file_path)
#                 except:
#                     pass
    
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])
#     return full_text

# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# Analyze these documents:

# DOCUMENTS:
# {documents}

# Provide a detailed summary with key insights in markdown format.
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge:

# QUESTION:
# {question}

# If unsure, say "I don't know". Format in markdown.
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# Synthesize the information and cite sources if available.
# """)

# # LangGraph nodes
# def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     # Check if documents is a string or dict
#     if isinstance(state, dict):
#         documents = state.get("documents", "")
#     else:
#         # If state itself is a string (documents content)
#         documents = state if state else ""
    
#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {"documents": RunnablePassthrough()}
#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"analysis": chain.invoke(documents)}

# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     # Check if state is a dictionary or directly the question
#     if isinstance(state, dict):
#         question = state.get("question", "")
#     else:
#         # If state itself is a string (question content)
#         question = state if state else ""
    
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"answer": chain.invoke(question)}

# def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from web search."""
#     # Check if state is a dictionary or directly the question
#     if isinstance(state, dict):
#         question = state.get("question", "")
#     else:
#         # If state itself is a string (question content)
#         question = state if state else ""
        
#     try:
#         search_results = search_tool.invoke({"query": question})
#         chain = (
#             {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
#             | WEB_SEARCH_PROMPT
#             | llm
#             | StrOutputParser()
#         )
#         return {"answer": chain.invoke({"question": question, "search_results": search_results})}
#     except Exception as e:
#         return {"answer": f"Failed to perform web search: {str(e)}"}

# def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     # Check if state is a dictionary or a string
#     if not isinstance(state, dict):
#         # If state is a string, assume it's document content
#         if state:
#             return "analyze_documents"
#         return "end"
    
#     # Normal dictionary flow
#     has_documents = "documents" in state and state["documents"]
#     has_question = "question" in state and state["question"]
    
#     if has_documents and not has_question:
#         return "analyze_documents"
#     elif has_question:
#         has_answer = "answer" in state
#         if has_answer and isinstance(state["answer"], str):
#             # Check if the answer indicates uncertainty
#             if any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
#                 return "answer_from_web"
#         elif not has_answer:
#             return "answer_from_knowledge"
#         return "end"
#     return "end"

# @st.cache_resource
# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow."""
#     workflow = Graph()
    
#     # Add all nodes first
#     workflow.add_node("analyze_documents", analyze_documents)
#     workflow.add_node("answer_from_knowledge", answer_from_knowledge)
#     workflow.add_node("answer_from_web", answer_from_web)
#     workflow.add_node("router", router)
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents": "analyze_documents",
#             "answer_from_knowledge": "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         },
#     )
    
#     # Add regular edges
#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")
#     workflow.add_edge("answer_from_web", END)
    
#     return workflow.compile()

# # Initialize workflow
# workflow = create_workflow()

# def main():
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Determine what path to take based on user input and available documents
#                     if st.session_state.documents:
#                         if prompt.lower().strip() in ["analyze", "analyze documents", "summarize", "summary"]:
#                             # Direct document analysis without questions
#                             result = analyze_documents(st.session_state.documents)
#                             response = result.get("analysis", "No analysis generated")
#                         else:
#                             # Question with documents context
#                             state = {
#                                 "documents": st.session_state.documents,
#                                 "question": prompt
#                             }
#                             result = workflow.invoke(state)
#                             # Extract result - could be analysis or answer
#                             response = result.get("analysis") or result.get("answer") or "I couldn't generate a response."
#                     else:
#                         # Just a question without document context
#                         result = answer_from_knowledge(prompt)
#                         if "I don't know" in result.get("answer", ""):
#                             # Try web search if knowledge base fails
#                             result = answer_from_web(prompt)
#                         response = result.get("answer", "I couldn't generate a response.")
                    
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     error_msg = f"An error occurred: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     if llm and search_tool:  # Only run if components initialized successfully
#         main()
#     else:
#         st.error("Failed to initialize required components. Please check your API keys.")


















# import os
# from typing import Dict, Any, List, Literal
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="Doc Analyst", page_icon="📄")
# st.title("📄 Document Analysis Agent")


# # Configuration
# MODEL_NAME = "llama3-8b-8192"
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# @st.cache_resource
# def initialize_components():
#     """Initialize components with proper error handling."""
#     try:
#         llm = ChatGroq(
#             api_key=os.getenv('GROQ_API_KEY'), 
#             temperature=TEMPERATURE,
#             model_name=MODEL_NAME,
#             max_tokens=MAX_TOKENS,
#         )
#         search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
#         return llm, search_tool
#     except Exception as e:
#         st.error(f"Failed to initialize components: {str(e)}")
#         return None, None

# llm, search_tool = initialize_components()

# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files with improved error handling."""
#     if not uploaded_files:
#         return ""
    
#     full_text = ""
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             if 'tmp_file_path' in locals():
#                 try:
#                     os.unlink(tmp_file_path)
#                 except:
#                     pass
    
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])
#     return full_text

# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# Analyze these documents:

# DOCUMENTS:
# {documents}

# Provide a detailed summary with key insights in markdown format.
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge:

# QUESTION:
# {question}

# If unsure, say "I don't know". Format in markdown.
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# Synthesize the information and cite sources if available.
# """)

# # LangGraph nodes
# def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     # Check if documents is a string or dict
#     if isinstance(state, dict):
#         documents = state.get("documents", "")
#     else:
#         # If state itself is a string (documents content)
#         documents = state if state else ""
    
#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {"documents": RunnablePassthrough()}
#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"analysis": chain.invoke(documents)}

# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     # Check if state is a dictionary or directly the question
#     if isinstance(state, dict):
#         question = state.get("question", "")
#     else:
#         # If state itself is a string (question content)
#         question = state if state else ""
    
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"answer": chain.invoke(question)}

# def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from web search."""
#     # Check if state is a dictionary or directly the question
#     if isinstance(state, dict):
#         question = state.get("question", "")
#     else:
#         # If state itself is a string (question content)
#         question = state if state else ""
        
#     try:
#         search_results = search_tool.invoke({"query": question})
#         chain = (
#             {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
#             | WEB_SEARCH_PROMPT
#             | llm
#             | StrOutputParser()
#         )
#         return {"answer": chain.invoke({"question": question, "search_results": search_results})}
#     except Exception as e:
#         return {"answer": f"Failed to perform web search: {str(e)}"}

# def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     # Check if state is a dictionary or a string
#     if not isinstance(state, dict):
#         # If state is a string, assume it's document content
#         if state:
#             return "analyze_documents"
#         return "end"
    
#     # Normal dictionary flow
#     has_documents = "documents" in state and state["documents"]
#     has_question = "question" in state and state["question"]
    
#     if has_documents and not has_question:
#         return "analyze_documents"
#     elif has_question:
#         has_answer = "answer" in state
#         if has_answer and isinstance(state["answer"], str):
#             # Check if the answer indicates uncertainty
#             if any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
#                 return "answer_from_web"
#         elif not has_answer:
#             return "answer_from_knowledge"
#         return "end"
#     return "end"

# @st.cache_resource
# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow."""
#     workflow = Graph()
    
#     # Add all nodes first
#     workflow.add_node("analyze_documents", analyze_documents)
#     workflow.add_node("answer_from_knowledge", answer_from_knowledge)
#     workflow.add_node("answer_from_web", answer_from_web)
#     workflow.add_node("router", router)
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents": "analyze_documents",
#             "answer_from_knowledge": "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         },
#     )
    
#     # Add regular edges
#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")
#     workflow.add_edge("answer_from_web", END)
    
#     return workflow.compile()

# # Initialize workflow
# workflow = create_workflow()

# def main():
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Determine what path to take based on user input and available documents
#                     if st.session_state.documents:
#                         if prompt.lower().strip() in ["analyze", "analyze documents", "summarize", "summary"]:
#                             # Direct document analysis without questions
#                             result = analyze_documents(st.session_state.documents)
#                             response = result.get("analysis", "No analysis generated")
#                         else:
#                             # Question with documents context
#                             state = {
#                                 "documents": st.session_state.documents,
#                                 "question": prompt
#                             }
#                             result = workflow.invoke(state)
#                             # Extract result - could be analysis or answer
#                             response = result.get("analysis") or result.get("answer") or "I couldn't generate a response."
#                     else:
#                         # Just a question without document context
#                         result = answer_from_knowledge(prompt)
#                         if "I don't know" in result.get("answer", ""):
#                             # Try web search if knowledge base fails
#                             result = answer_from_web(prompt)
#                         response = result.get("answer", "I couldn't generate a response.")
                    
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     error_msg = f"An error occurred: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     if llm and search_tool:  # Only run if components initialized successfully
#         main()
#     else:
#         st.error("Failed to initialize required components. Please check your API keys.")

























# import os
# from typing import Dict, Any, List, Literal
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="Doc Analyst", page_icon="📄")
# st.title("📄 Document Analysis Agent")


# # Configuration
# MODEL_NAME = "llama3-8b-8192"
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# @st.cache_resource
# def initialize_components():
#     """Initialize components with proper error handling."""
#     try:
#         llm = ChatGroq(
#             api_key=os.getenv('GROQ_API_KEY'), 
#             temperature=TEMPERATURE,
#             model_name=MODEL_NAME,
#             max_tokens=MAX_TOKENS,
#         )
#         search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
#         return llm, search_tool
#     except Exception as e:
#         st.error(f"Failed to initialize components: {str(e)}")
#         return None, None

# llm, search_tool = initialize_components()

# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files with improved error handling."""
#     if not uploaded_files:
#         return ""
    
#     full_text = ""
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             if 'tmp_file_path' in locals():
#                 try:
#                     os.unlink(tmp_file_path)
#                 except:
#                     pass
    
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])
#     return full_text

# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# Analyze these documents:

# DOCUMENTS:
# {documents}

# Provide a detailed summary with key insights in markdown format.
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge:

# QUESTION:
# {question}

# If unsure, say "I don't know". Format in markdown.
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# Synthesize the information and cite sources if available.
# """)

# # LangGraph nodes
# def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     documents = state.get("documents", "")
#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {"documents": RunnablePassthrough()}
#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"analysis": chain.invoke(documents)}

# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     question = state.get("question", "")
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"answer": chain.invoke(question)}

# def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from web search."""
#     question = state.get("question", "")
#     try:
#         search_results = search_tool.invoke({"query": question})
#         chain = (
#             {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
#             | WEB_SEARCH_PROMPT
#             | llm
#             | StrOutputParser()
#         )
#         return {"answer": chain.invoke({"question": question, "search_results": search_results})}
#     except Exception as e:
#         return {"answer": f"Failed to perform web search: {str(e)}"}

# def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     if state.get("documents") and not state.get("question"):
#         return "analyze_documents"
#     elif state.get("question"):
#         if "answer" in state and any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
#             return "answer_from_web"
#         elif "answer" not in state:
#             return "answer_from_knowledge"
#         return "end"
#     return "end"

# @st.cache_resource
# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow."""
#     workflow = Graph()
    
#     # Add all nodes first
#     workflow.add_node("analyze_documents", analyze_documents)
#     workflow.add_node("answer_from_knowledge", answer_from_knowledge)
#     workflow.add_node("answer_from_web", answer_from_web)
#     workflow.add_node("router", router)
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents": "analyze_documents",
#             "answer_from_knowledge": "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         },
#     )
    
#     # Add regular edges
#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")
#     workflow.add_edge("answer_from_web", END)
    
#     return workflow.compile()

# # Initialize workflow
# workflow = create_workflow()

# def main():
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Initialize state with documents as string
#                     state = {
#                         "documents": st.session_state.documents,
#                         "question": prompt if prompt else None
#                         # No need to initialize answer here
#                     }
                    
#                     # Run through workflow
#                     result = workflow.invoke(state)
                    
#                     # Handle the response
#                     if "analysis" in result:
#                         response = result["analysis"]
#                     elif "answer" in result:
#                         response = result["answer"]
#                     else:
#                         response = "I couldn't generate a response."
                    
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     error_msg = f"An error occurred: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     if llm and search_tool:  # Only run if components initialized successfully
#         main()
#     else:
#         st.error("Failed to initialize required components. Please check your API keys.")










# import os
# from typing import Dict, Any, List, Literal
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter

# st.set_page_config(page_title="Doc Analyst", page_icon="📄")
# st.title("📄 Document Analysis Agent")


# # Configuration
# MODEL_NAME = "llama3-8b-8192"
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# @st.cache_resource
# def initialize_components():
#     """Initialize components with proper error handling."""
#     try:
#         llm = ChatGroq(
#             api_key=os.getenv('GROQ_API_KEY'), 
#             temperature=TEMPERATURE,
#             model_name=MODEL_NAME,
#             max_tokens=MAX_TOKENS,
#         )
#         search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
#         return llm, search_tool
#     except Exception as e:
#         st.error(f"Failed to initialize components: {str(e)}")
#         return None, None

# llm, search_tool = initialize_components()

# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files with improved error handling."""
#     if not uploaded_files:
#         return ""
    
#     full_text = ""
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             if 'tmp_file_path' in locals():
#                 try:
#                     os.unlink(tmp_file_path)
#                 except:
#                     pass
    
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])
#     return full_text

# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# Analyze these documents:

# DOCUMENTS:
# {documents}

# Provide a detailed summary with key insights in markdown format.
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge:

# QUESTION:
# {question}

# If unsure, say "I don't know". Format in markdown.
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# Synthesize the information and cite sources if available.
# """)

# # LangGraph nodes
# def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     documents = state.get("documents", "")
#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {"documents": RunnablePassthrough()}
#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"analysis": chain.invoke(documents)}

# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     question = state.get("question", "")
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"answer": chain.invoke(question)}

# def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from web search."""
#     question = state.get("question", "")
#     try:
#         search_results = search_tool.invoke({"query": question})
#         chain = (
#             {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
#             | WEB_SEARCH_PROMPT
#             | llm
#             | StrOutputParser()
#         )
#         return {"answer": chain.invoke({"question": question, "search_results": search_results})}
#     except Exception as e:
#         return {"answer": f"Failed to perform web search: {str(e)}"}

# def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     if state.get("documents") and not state.get("question"):
#         return "analyze_documents"
#     elif state.get("question"):
#         if "answer" in state and any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
#             return "answer_from_web"
#         return "end"
#     return "end"

# @st.cache_resource
# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow."""
#     workflow = Graph()
    
#     # Add all nodes first
#     workflow.add_node("analyze_documents", analyze_documents)
#     workflow.add_node("answer_from_knowledge", answer_from_knowledge)
#     workflow.add_node("answer_from_web", answer_from_web)
#     workflow.add_node("router", router)
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents": "analyze_documents",
#             "answer_from_knowledge": "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         },
#     )
    
#     # Add regular edges
#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")
#     workflow.add_edge("answer_from_web", END)
    
#     return workflow.compile()

# # Initialize workflow
# workflow = create_workflow()

# def main():
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     # Initialize state with empty answer
#                     state = {
#                         "documents": st.session_state.documents,
#                         "question": prompt if prompt else None,
#                         "answer": ""  # Initialize answer field
#                     }
                    
#                     # First pass through workflow
#                     result = workflow.invoke(state)
                    
#                     # Handle the response
#                     if "analysis" in result:
#                         response = result["analysis"]
#                     elif "answer" in result:
#                         response = result["answer"]
#                     else:
#                         response = "I couldn't generate a response."
                    
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     error_msg = f"An error occurred: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     if llm and search_tool:  # Only run if components initialized successfully
#         main()
#     else:
#         st.error("Failed to initialize required components. Please check your API keys.")








# import os
# from typing import Dict, Any, List, Optional, Literal
# import streamlit as st
# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# # Configuration
# MODEL_NAME = "llama-3.1-8b-instant"
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200


# @st.cache_resource
# def initialize_components():
#     llm = ChatGroq(
#         api_key=os.getenv('GROQ_API_KEY'), 
#         temperature=TEMPERATURE,
#         model=MODEL_NAME,
#         max_tokens=MAX_TOKENS,
#     )

#     search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)

#     return search_tool, llm

# llm, search_tool = initialize_components()


# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files and return concatenated text."""
#     full_text = ""
    
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#             tmp_file.write(uploaded_file.getvalue())
#             tmp_file_path = tmp_file.name
        
#         try:
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             os.unlink(tmp_file_path)
    
#     # Simple chunking to manage context window
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])  # Limit to first 3 chunks
#     return full_text





# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# You are an expert data analyst. Analyze these documents:

# DOCUMENTS:
# {documents}

# INSTRUCTIONS:
# 1. Identify key information and patterns
# 2. Provide a detailed summary
# 3. Format in markdown with headings

# ANALYSIS:
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge.

# QUESTION:
# {question}

# If you don't know, say "I don't know" - don't make up answers.
# Format your response in markdown.

# ANSWER:
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# INSTRUCTIONS:
# 1. Synthesize the information
# 2. Cite sources when possible
# 3. If results don't help, say so

# ANSWER:
# """)





# # Creating Nodes


# def analyse_documents(state: Dict[str, any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     documents = state.get('documents', '')

#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {'documents': RunnablePassthrough()}

#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )

#     analysis = chain.invoke(documents)
#     return {"analysis": analysis}


# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     question = state.get("question", "")
    
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
    
#     answer = chain.invoke(question)
#     return {"answer": answer}



# def answer_from_web(state: Dict[str, any]) -> Dict[str, any]:
#     """Node for answering from web search"""
#     question = state.get('question', '')

#     search_results = search_tool.invoke({'query': question})

#     chain = (
#         {
#             "question": RunnablePassthrough(),
#             "search_results": RunnablePassthrough()
#         }
#         | WEB_SEARCH_PROMPT
#         | llm
#         | StrOutputParser()
#     )

#     answer = chain.invoke({
#         "question": question,
#         "search_results": search_results
#     })
#     return {"answer": answer}



# def router(state: Dict[str, any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     if state.get("documents") and not state.get("question"):
#         return 'analyze_documents'
#     elif state.get('question'):
#         if "don't know" in state.get('answer', '').lower():
#             return "answer_from_web"
#         return "end"
#     else:
#         return "end"



# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow."""
#     workflow = Graph()

#     workflow.add_node('analyze_documents', analyse_documents)
#     workflow.add_node('answer_from_knowledge', answer_from_knowledge)
#     workflow.add_node('answer_from_web', answer_from_web)

#     workflow.set_entry_point("router")

#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents" : "analyze_documents",
#             "answer_from_knowledge" : "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         }
#     )

#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")
#     workflow.add_edge("answer_from_web", END)
#     return workflow.compile()



# try:
#     workflow = create_workflow()
# except Exception as e:
#     st.error(f"Failed to create workflow: {str(e)}")


# # from langgraph.graph import Graph
# # workflow.get_graph().draw_mermaid_png()










# def main():
#     st.set_page_config(page_title="Doc Analyst", page_icon="📄")
#     st.title("📄 Document Analysis Agent with LangGraph")
#     st.markdown("Upload documents or ask questions - I'll analyze or find answers!")
    
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 # Prepare workflow state
#                 state = {
#                     "documents": st.session_state.documents,
#                     "question": prompt if prompt else None,
#                 }
                
#                 # Execute workflow
#                 result = workflow.invoke(state)
                
#                 # Display appropriate response
#                 if "analysis" in result:
#                     response = result["analysis"]
#                 elif "answer" in result:
#                     response = result["answer"]
#                 else:
#                     response = "I couldn't generate a response."
                
#                 st.markdown(response)
#                 st.session_state.messages.append({"role": "assistant", "content": response})

# if __name__ == "__main__":
#     main()



# import os
# from typing import Dict, Any, List, Literal
# import streamlit as st

# from dotenv import load_dotenv
# load_dotenv()

# import tempfile
# from langchain_core.runnables import RunnablePassthrough
# from langgraph.graph import Graph, END
# from langchain_groq import ChatGroq
# from langchain_community.document_loaders import (
#     CSVLoader,
#     PyPDFLoader,
#     Docx2txtLoader,
#     TextLoader,
#     UnstructuredFileLoader
# )
# from langchain_community.tools.tavily_search import TavilySearchResults
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain.text_splitter import RecursiveCharacterTextSplitter


# st.set_page_config(page_title="Doc Analyst", page_icon="📄")
# st.title("📄 Document Analysis Agent")



# # Configuration
# MODEL_NAME = "llama3-8b-8192"  # Updated to correct model name
# TEMPERATURE = 0.1
# MAX_TOKENS = 500
# TAVILY_MAX_RESULTS = 3
# CHUNK_SIZE = 1000
# CHUNK_OVERLAP = 200

# @st.cache_resource
# def initialize_components():
#     """Initialize components with proper error handling."""
#     try:
#         llm = ChatGroq(
#             api_key=os.getenv('GROQ_API_KEY'), 
#             temperature=TEMPERATURE,
#             model_name=MODEL_NAME,  # Changed from 'model' to 'model_name'
#             max_tokens=MAX_TOKENS,
#         )
#         search_tool = TavilySearchResults(max_results=TAVILY_MAX_RESULTS)
#         return llm, search_tool
#     except Exception as e:
#         st.error(f"Failed to initialize components: {str(e)}")
#         return None, None

# llm, search_tool = initialize_components()

# def process_uploaded_files(uploaded_files: List) -> str:
#     """Process uploaded files with improved error handling."""
#     if not uploaded_files:
#         return ""
    
#     full_text = ""
#     for uploaded_file in uploaded_files:
#         file_ext = os.path.splitext(uploaded_file.name)[1].lower()
        
#         try:
#             with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as tmp_file:
#                 tmp_file.write(uploaded_file.getvalue())
#                 tmp_file_path = tmp_file.name
            
#             if file_ext == '.pdf':
#                 loader = PyPDFLoader(tmp_file_path)
#             elif file_ext == '.csv':
#                 loader = CSVLoader(tmp_file_path)
#             elif file_ext == '.docx':
#                 loader = Docx2txtLoader(tmp_file_path)
#             elif file_ext == '.txt':
#                 loader = TextLoader(tmp_file_path)
#             else:
#                 loader = UnstructuredFileLoader(tmp_file_path)
            
#             docs = loader.load()
#             full_text += "\n\n".join([doc.page_content for doc in docs])
        
#         except Exception as e:
#             st.error(f"Error processing {uploaded_file.name}: {str(e)}")
#         finally:
#             if 'tmp_file_path' in locals():
#                 try:
#                     os.unlink(tmp_file_path)
#                 except:
#                     pass
    
#     if len(full_text) > CHUNK_SIZE:
#         text_splitter = RecursiveCharacterTextSplitter(
#             chunk_size=CHUNK_SIZE,
#             chunk_overlap=CHUNK_OVERLAP
#         )
#         chunks = text_splitter.split_text(full_text)
#         return "\n\n[Continued...]\n\n".join(chunks[:3])
#     return full_text

# # Prompt templates
# DOCUMENT_ANALYSIS_PROMPT = ChatPromptTemplate.from_template("""
# Analyze these documents:

# DOCUMENTS:
# {documents}

# Provide a detailed summary with key insights in markdown format.
# """)

# DIRECT_ANSWER_PROMPT = ChatPromptTemplate.from_template("""
# Answer the question based on your knowledge:

# QUESTION:
# {question}

# If unsure, say "I don't know". Format in markdown.
# """)

# WEB_SEARCH_PROMPT = ChatPromptTemplate.from_template("""
# Answer using these web results:

# QUESTION: {question}

# RESULTS:
# {search_results}

# Synthesize the information and cite sources if available.
# """)

# # LangGraph nodes
# def analyze_documents(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for document analysis."""
#     documents = state.get("documents", "")
#     if not documents:
#         return {"analysis": "No documents provided"}
    
#     chain = (
#         {"documents": RunnablePassthrough()}
#         | DOCUMENT_ANALYSIS_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"analysis": chain.invoke({"documents": documents})}

# def answer_from_knowledge(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from LLM's knowledge."""
#     question = state.get("question", "")
#     chain = (
#         {"question": RunnablePassthrough()}
#         | DIRECT_ANSWER_PROMPT
#         | llm
#         | StrOutputParser()
#     )
#     return {"answer": chain.invoke({"question": question})}

# def answer_from_web(state: Dict[str, Any]) -> Dict[str, Any]:
#     """Node for answering from web search."""
#     question = state.get("question", "")
#     try:
#         search_results = search_tool.invoke({"query": question})
#         chain = (
#             {"question": RunnablePassthrough(), "search_results": RunnablePassthrough()}
#             | WEB_SEARCH_PROMPT
#             | llm
#             | StrOutputParser()
#         )
#         return {"answer": chain.invoke({"question": question, "search_results": search_results})}
#     except Exception as e:
#         return {"answer": f"Failed to perform web search: {str(e)}"}

# def router(state: Dict[str, Any]) -> Literal["analyze_documents", "answer_from_knowledge", "answer_from_web", "end"]:
#     """Router node to decide the flow."""
#     if state.get("documents") and not state.get("question"):
#         return "analyze_documents"
#     elif state.get("question"):
#         if "answer" in state and any(phrase in state["answer"].lower() for phrase in ["don't know", "not sure", "no information"]):
#             return "answer_from_web"
#         return "end"
#     return "end"

# @st.cache_resource
# def create_workflow() -> Graph:
#     """Create and compile the LangGraph workflow with all nodes properly defined."""
#     workflow = Graph()
    
#     # Add all nodes first
#     workflow.add_node("analyze_documents", analyze_documents)
#     workflow.add_node("answer_from_knowledge", answer_from_knowledge)
#     workflow.add_node("answer_from_web", answer_from_web)
    
#     # Add router as a node
#     workflow.add_node("router", router)
    
#     # Set entry point
#     workflow.set_entry_point("router")
    
#     # Add conditional edges
#     workflow.add_conditional_edges(
#         "router",
#         router,
#         {
#             "analyze_documents": "analyze_documents",
#             "answer_from_knowledge": "answer_from_knowledge",
#             "answer_from_web": "answer_from_web",
#             "end": END,
#         },
#     )
    
#     # Add regular edges
#     workflow.add_edge("analyze_documents", END)
#     workflow.add_edge("answer_from_knowledge", "router")  # Loop back to router
#     workflow.add_edge("answer_from_web", END)
    
#     return workflow.compile()

# # Initialize workflow
# workflow = create_workflow()

# def main():
#     # Initialize session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = []
#     if "documents" not in st.session_state:
#         st.session_state.documents = ""
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents (PDF, DOCX, CSV, TXT)",
#         type=["pdf", "docx", "csv", "txt"],
#         accept_multiple_files=True,
#     )
    
#     if uploaded_files and not st.session_state.documents:
#         with st.spinner("Processing documents..."):
#             st.session_state.documents = process_uploaded_files(uploaded_files)
#             st.success(f"Processed {len(uploaded_files)} file(s)")
    
#     # Display chat
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])
    
#     # Chat input
#     if prompt := st.chat_input("Ask a question or request analysis"):
#         st.session_state.messages.append({"role": "user", "content": prompt})
        
#         with st.chat_message("user"):
#             st.markdown(prompt)
        
#         with st.chat_message("assistant"):
#             with st.spinner("Thinking..."):
#                 try:
#                     state = {
#                         "documents": st.session_state.documents,
#                         "question": prompt if prompt else None,
#                     }
                    
#                     # First pass through workflow
#                     result = workflow.invoke(state)
                    
#                     # Handle the response
#                     if "analysis" in result:
#                         response = result["analysis"]
#                     elif "answer" in result:
#                         response = result["answer"]
#                     else:
#                         response = "I couldn't generate a response."
                    
#                     st.markdown(response)
#                     st.session_state.messages.append({"role": "assistant", "content": response})
#                 except Exception as e:
#                     error_msg = f"An error occurred: {str(e)}"
#                     st.error(error_msg)
#                     st.session_state.messages.append({"role": "assistant", "content": error_msg})

# if __name__ == "__main__":
#     if llm and search_tool:  # Only run if components initialized successfully
#         main()
#     else:
#         st.error("Failed to initialize required components. Please check your API keys.")